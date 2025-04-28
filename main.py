import argparse
import os
import numpy as np
import torch.nn.functional as f
import torch
from torch import nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
from data.data import load_and_preprocess, GWLDataset
from model.GFN import GraphFourierNet
from utils.graph_build import build_station_graph
from utils.random_seed import set_random_seed

parser = argparse.ArgumentParser(description='Graph Fourier Net:')
parser.add_argument('--data', type=str, default='YRB', help='data set')
parser.add_argument('--seq_len', type=int, default=6, help='input length')
parser.add_argument('--pred_len', type=int, default=5, help='predict length')
parser.add_argument('--feat_dim', type=int, default=5, help='feature size')
parser.add_argument('--batch_size', type=int, default=8, help='input data batch size')
parser.add_argument('--epochs', type=int, default=200, help='train epochs')
parser.add_argument('--lr', type=float, default=0.5, help='learning epochs')
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--gat_hidden', type=int, default=256, help='gat dimensions')
parser.add_argument('--gat_heads', type=int, default=4, help='gat heads')
parser.add_argument('--gat_layers', type=int, default=1, help='gat layers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--fnn_embed_size', type=int, default=256)
parser.add_argument('--edge_keep_ratio', type=float, default=0.4)
args = parser.parse_args()


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    return x, y


def train():
    print(f'Training configs: {args}')
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, _, _ = model(x)
            loss = f.smooth_l1_loss(pred, y)
            for name, param in model.named_parameters():
                if 'attn' in name and param.grad is not None:
                    print(f"{name} grad mean: {param.grad.mean().item():.3e}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred, _, _ = model(x)
                val_loss += f.smooth_l1_loss(pred.view(-1), y.view(-1)).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        print(f'Epoch {epoch + 1:03d} | Train Loss: {avg_train:.7f} | Val Loss: {avg_val:.7f}')
        torch.save(model.state_dict(), f'data/{args.data}/pred_len={args.pred_len}_feat_dim={args.feat_dim}_best_model.pth')


def test():
    model.load_state_dict(torch.load(f'data/{args.data}/best_model.pth', weights_only=True))
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred, alphas, emb = model(x)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
    # 整合结果
    preds = np.concatenate(all_preds, axis=0)  # (num_samples, num_stations * pred_len)
    trues = np.concatenate(all_trues, axis=0)  # (num_samples, num_stations * pred_len)

    # 重塑为三维数组 (num_samples, num_stations, pred_len)
    preds_3d = preds.reshape(-1, test_set.num_stations, args.pred_len)
    trues_3d = trues.reshape(-1, test_set.num_stations, args.pred_len)

    # 反标准化
    preds_inv = np.zeros_like(preds_3d)
    trues_inv = np.zeros_like(trues_3d)

    for i in range(test_set.num_stations):
        station_pred = preds_3d[:, i, :].reshape(-1, 1)
        preds_inv[:, i, :] = test_set.target_scalers[i].inverse_transform(station_pred).reshape(-1, args.pred_len)

        station_true = trues_3d[:, i, :].reshape(-1, 1)
        trues_inv[:, i, :] = test_set.target_scalers[i].inverse_transform(station_true).reshape(-1, args.pred_len)


if __name__ == "__main__":
    set_random_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stations = [f.split('.')[0] for f in os.listdir(f'data/{args.data}') if f.endswith('.xlsx')]
    num_nodes = len(stations)
    features, targets, code_map, full_df = load_and_preprocess(f'data/{args.data}', stations)
    edge_index = build_station_graph(code_map, device)
    train_set = GWLDataset(features, targets, mode='train', full_df=full_df, cfg=args)
    val_set = GWLDataset(features, targets, feat_scalers=train_set.feat_scalers,
                         target_scalers=train_set.target_scalers, mode='val', full_df=full_df, cfg=args)
    test_set = GWLDataset(features, targets, feat_scalers=train_set.feat_scalers,
                          target_scalers=train_set.target_scalers, mode='test', full_df=full_df, cfg=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn)
    model = GraphFourierNet(num_nodes, edge_index, args).to(device)


    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.1)


    model.apply(weights_init)
    train()
    test()
