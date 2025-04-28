import torch
import torch.nn as nn
import torch.nn.functional as f
from model.FNN import FNN
from model.GAT import GATLayer


class DynamicGraphLearner(nn.Module):
    def __init__(self, num_nodes, hidden_size, cfg):
        super().__init__()
        self.num_nodes = num_nodes

        # Learnable node embeddings.
        # 可学习的节点嵌入。
        self.node_emb = nn.Embedding(num_nodes, hidden_size)

        # Static graph encoder using GAT.
        # 使用 GAT 的静态图编码器。
        self.static_gat = GATLayer(hidden_size, hidden_size, heads=1)

        # Edge similarity scoring network.
        # 边相似度评分网络。
        self.sim_fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False)
        )

        self.cfg = cfg
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters with Xavier uniform.
        # 使用 Xavier 均匀初始化参数。
        nn.init.xavier_uniform_(self.node_emb.weight)
        for layer in self.sim_fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, static_edge_index):
        """
        Generate a dynamic graph structure based on node embeddings.
        根据节点嵌入动态生成图结构。
        """
        node_ids = torch.arange(self.num_nodes, device=x.device)
        emb = self.node_emb(node_ids)

        # Get node embeddings through static GAT.
        # 通过静态 GAT 得到节点嵌入。
        emb, _, _ = self.static_gat(emb, static_edge_index)

        # Construct all possible node pairs (excluding self-loops).
        # 构造所有可能的节点对（排除自环）。
        all_nodes = torch.arange(self.num_nodes, device=x.device)
        candidate_src, candidate_dst = torch.meshgrid(all_nodes, all_nodes, indexing='ij')
        mask = candidate_src != candidate_dst  # Exclude self-loops. 排除自环。
        candidate_src = candidate_src[mask]
        candidate_dst = candidate_dst[mask]

        # Remove existing static edges from candidates.
        # 从候选边中排除已有的静态边。
        static_src, static_dst = static_edge_index
        static_set = set(zip(static_src.cpu().numpy(), static_dst.cpu().numpy()))

        candidate_pairs = zip(candidate_src.cpu().numpy(), candidate_dst.cpu().numpy())
        keep_mask = [tuple(pair) not in static_set for pair in candidate_pairs]
        candidate_src = candidate_src[keep_mask]
        candidate_dst = candidate_dst[keep_mask]

        # Compute similarity scores for candidate edges.
        # 为候选边计算相似度得分。
        src_emb = emb[candidate_src]
        dst_emb = emb[candidate_dst]
        pair_feat = torch.cat([src_emb, dst_emb], dim=-1)
        scores = torch.tanh(self.sim_fc(pair_feat)).squeeze()

        # Select top-k scored edges to form dynamic edges.
        # 选择得分最高的 top-k 边构成动态边。
        k = int(len(scores) * self.cfg.edge_keep_ratio)
        _, topk_indices = torch.topk(scores, k=k)
        dynamic_edges = torch.stack([candidate_src[topk_indices], candidate_dst[topk_indices]])

        # Combine static and dynamic edges, and remove duplicates.
        # 合并静态和动态边，并去重。
        combined_edges = torch.cat([static_edge_index, dynamic_edges], dim=1)
        combined_edges = combined_edges.unique(dim=1)

        return combined_edges, emb


class GraphFourierNet(nn.Module):
    def __init__(self, num_nodes, edge_index, cfg):
        super().__init__()
        self.num_nodes = num_nodes

        # Register static edge index as a buffer (not a parameter).
        # 注册静态边索引为 buffer（非参数）。
        self.register_buffer('edge_index', edge_index)

        self.cfg = cfg
        self.edge_index = edge_index

        # Module to learn dynamic edges.
        # 用于学习动态边结构的模块。
        self.dynamic_learner = DynamicGraphLearner(
            num_nodes=num_nodes,
            hidden_size=cfg.gat_hidden,
            cfg=cfg
        )

        # Fourier Graph Network for initial encoding.
        # FGN 模块用于初始时序编码。
        self.fnn = FNN(
            pre_length=cfg.pred_len,
            embed_size=cfg.fnn_embed_size,
            feature_size=cfg.feat_dim,
            seq_length=cfg.seq_len,
            hidden_size=cfg.gat_hidden
        )

        # Multi-layer GAT encoder.
        # 多层 GAT 编码器。
        self.gat = nn.ModuleList()
        for _ in range(cfg.gat_layers):
            self.gat.append(GATLayer(cfg.gat_hidden,
                                     cfg.gat_hidden // cfg.gat_heads,
                                     heads=cfg.gat_heads))

        # Final prediction decoder.
        # 最后的预测解码器。
        self.decoder = nn.Sequential(
            nn.Linear(cfg.gat_hidden, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.pred_len)
        )

    def forward(self, x):
        """
        Full forward pass through GraphFourierNet.
        执行 GraphFourierNet 的完整前向传播。
        :param x: Input tensor of shape [B, N, T, F] （输入张量：[批次, 节点, 时间步, 特征]）
        :return: Prediction, attention weights, and final node embeddings.
        返回：预测结果，注意力分数，节点嵌入。
        """
        batch_size, num_nodes, seq_len, feat_dim = x.size()

        # Flatten for FGN input.
        # 调整形状用于 FGN 输入。
        x = x.view(-1, seq_len, feat_dim)
        gat_input = self.fnn(x)
        gat_input = gat_input.view(batch_size, num_nodes, -1)

        # Learn dynamic edge structure.
        # 学习动态边结构。
        dynamic_edge_index, node_emb = self.dynamic_learner(gat_input, self.edge_index)

        # Combine static and dynamic edges for current batch.
        # 当前 batch 合并静态与动态边。
        combined_edges = torch.cat([self.edge_index, dynamic_edge_index], dim=1)
        batch_edge_index = torch.cat(
            [combined_edges for _ in range(batch_size)],
            dim=1
        )

        # GAT propagation over dynamic graph.
        # 在动态图上进行 GAT 传播。
        all_alphas = []
        x_gat = gat_input.view(-1, self.cfg.gat_hidden)

        for i, gat_layer in enumerate(self.gat):
            residual = x_gat
            x_gat, alpha, edges = gat_layer(x_gat, batch_edge_index)
            all_alphas.append((alpha.cpu(), edges.cpu()))

            # Add residual connection.
            # 添加残差连接。
            x_gat = x_gat + residual

            if i != self.cfg.gat_layers - 1:
                x_gat = f.elu(x_gat)
                x_gat = f.dropout(x_gat, p=self.cfg.dropout, training=self.training)

        # Reshape and decode.
        # 重塑输出并进行解码。
        x_out = x_gat.view(batch_size, num_nodes, -1)
        output = self.decoder(x_out)

        return output.view(batch_size, -1), all_alphas, node_emb
