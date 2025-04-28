import os
import json
import geopandas as gpd
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform

from data.data import load_and_preprocess, GWLDataset
from main import args
from model.GFN import GraphFourierNet
from utils.graph_build import build_station_graph
from utils.random_seed import set_random_seed

matplotlib.use("TkAgg")

# === Utility Functions ===

def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    gdf['id'] = gdf['id'].astype(str)
    return gdf

def build_pos_from_shapefile(station_shp, stations):
    gdf = load_shapefile(station_shp)
    stations_str = list(map(str, stations))
    pos = {}
    for idx, station_id in enumerate(stations_str):
        matched = gdf[gdf['id'] == station_id]
        if not matched.empty:
            lon, lat = matched.geometry.values[0].x, matched.geometry.values[0].y
            pos[idx] = (lon, lat)
    return pos

def map_station_to_region(station_shp, region_shp):
    stations_gdf = load_shapefile(station_shp)
    regions_gdf = load_shapefile(region_shp)
    stations_gdf = stations_gdf.set_geometry("geometry")
    regions_gdf = regions_gdf.set_geometry("geometry")
    joined = gpd.sjoin(stations_gdf, regions_gdf, how="left", predicate="within")
    station_to_region = dict(zip(joined['id_left'], joined['GWZ']))
    return station_to_region

# === Core Classes ===

class FixedGraphModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.static_edge_index = original_model.edge_index
        self.current_edges = self.static_edge_index
        if hasattr(self.original_model, 'dynamic_learner'):
            for param in self.original_model.dynamic_learner.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        saved = self.original_model.edge_index
        try:
            self.original_model.edge_index = self.current_edges
            return self.original_model(x)
        finally:
            self.original_model.edge_index = saved

class MonteCarloEdgeExplainer:
    def __init__(self, fixed_model, edge_index, device='cuda', num_samples=50):
        self.model = fixed_model
        self.edge_index = edge_index.to(device)
        self.device = device
        self.num_edges = self.edge_index.size(1)
        self.num_samples = num_samples

    def explain(self, x):
        x = x.to(self.device)
        shap_values = np.zeros(self.num_edges, dtype=np.float32)
        for e_idx in tqdm(range(self.num_edges), desc="MC Explaining"):
            sum_diff = 0.0
            for _ in range(self.num_samples):
                sub_mask = torch.rand(self.num_edges, device=self.device) < 0.5
                sub_mask[e_idx] = False
                self.model.current_edges = self.edge_index[:, sub_mask]
                with torch.no_grad():
                    base_val = self.model(x)[0].mean().item()
                
                with_edge = sub_mask.clone()
                with_edge[e_idx] = True
                self.model.current_edges = self.edge_index[:, with_edge]
                with torch.no_grad():
                    compare_val = self.model(x)[0].mean().item()

                sum_diff += (compare_val - base_val)

            shap_values[e_idx] = sum_diff / self.num_samples
        return shap_values

class MCGraphResult:
    def __init__(self, edge_index, edge_shap):
        self.edge_index = edge_index
        self.edge_shap = edge_shap

    def compute_node_contrib(self, num_nodes):
        node_contrib = np.zeros(num_nodes, dtype=np.float32)
        src = self.edge_index[0]
        dst = self.edge_index[1]
        for e_idx, val in enumerate(self.edge_shap):
            node_contrib[src[e_idx]] += val / 2
            node_contrib[dst[e_idx]] += val / 2
        return node_contrib

    def top_edges(self, k=10):
        idx_sort = np.argsort(-self.edge_shap)
        for i in range(min(k, len(idx_sort))):
            e = idx_sort[i]
            print(f"Edge {e}: shap={self.edge_shap[e]:.4f}, (src={self.edge_index[0, e]}, dst={self.edge_index[1, e]})")


# === Main Execution ===

def main():
    set_random_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join('../../data', args.data)
    stations = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.xlsx')]

    features, targets, code_map, full_df = load_and_preprocess(data_dir, stations)
    num_nodes = len(stations)
    train_set = GWLDataset(features, targets, mode='train', full_df=full_df, cfg=args)
    edge_index = build_station_graph(code_map, device)
    model = GraphFourierNet(num_nodes, edge_index, args).to(device)

    ckpt_path = f"../../data/{args.data}/best_model.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    fixed_model = FixedGraphModel(model).to(device)
    loader = DataLoader(train_set, batch_size=8, shuffle=False, collate_fn=lambda batch: (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])))
    batch_x, _ = next(iter(loader))
    sample_x = batch_x[:1]

    save_dir = f"../../data/{args.data}/mc_graph_shap_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mc_explain.json")

    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            out_data = json.load(f)
        edge_index_np = np.array(out_data['edge_index'])
        shap_values = np.array(out_data['shap_values'])
        node_contrib = np.array(out_data['node_contrib'])
    else:
        mc_explainer = MonteCarloEdgeExplainer(fixed_model, edge_index, device=device, num_samples=50)
        shap_values = mc_explainer.explain(sample_x)

        edge_index_np = edge_index.cpu().numpy()
        result = MCGraphResult(edge_index_np, shap_values)
        node_contrib = result.compute_node_contrib(num_nodes)

        out_data = {
            'edge_index': edge_index_np.tolist(),
            'shap_values': shap_values.tolist(),
            'node_contrib': node_contrib.tolist(),
        }
        with open(save_path, 'w') as f:
            json.dump(out_data, f, indent=2)

    result = MCGraphResult(edge_index_np, shap_values)
    result.top_edges(k=10)

if __name__ == "__main__":
    main()