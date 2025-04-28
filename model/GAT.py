import torch
import torch.nn as nn
import torch.nn.functional as f


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=1):
        super().__init__()
        self.heads = heads
        self.out_features = out_features

        # Layer normalization for stabilizing training.
        # 添加层归一化以稳定训练。
        self.norm = nn.LayerNorm(out_features * heads)

        # Residual connection gating parameter.
        # 残差连接的门控参数（可学习的残差权重）。
        self.res_gate = nn.Parameter(torch.zeros(1))

        # Linear transformation for input features.
        # 输入特征的线性变换。
        self.W = nn.Linear(in_features, out_features * heads, bias=False)

        # Attention mechanism for edge-level scoring.
        # 注意力机制，用于计算边的权重分数。
        self.attn = nn.Linear(2 * out_features, 1, bias=False)

        # Initialize weights.
        # 初始化权重。
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier uniform initialization for weight matrices.
        # 使用 Xavier 均匀初始化。
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn.weight)

    def grouped_softmax(self, alpha, dst):
        """
        Computes softmax over edges grouped by destination node.
        针对每个目标节点，按组进行 softmax 归一化注意力分数。
        """
        softmax_values = torch.zeros_like(alpha)
        for h in range(self.heads):
            alpha_h = alpha[:, h]
            unique_dst, inverse_indices = torch.unique(dst, return_inverse=True)

            # Get max value for numerical stability.
            # 为了数值稳定性，先减去每组的最大值。
            max_values = torch.zeros_like(unique_dst, dtype=alpha.dtype).scatter_reduce_(
                0, inverse_indices, alpha_h, reduce='amax', include_self=False
            )
            stable_alpha = alpha_h - max_values[inverse_indices]

            # Compute exp and normalized softmax.
            # 计算指数并归一化。
            exp_alpha = torch.exp(stable_alpha)
            sum_exp = torch.zeros_like(unique_dst, dtype=alpha.dtype).scatter_add_(
                0, inverse_indices, exp_alpha
            )
            softmax = exp_alpha / (sum_exp[inverse_indices] + 1e-8)
            softmax_values[:, h] = softmax

        return softmax_values

    def forward(self, x, edge_index):
        """
        Forward pass of GATLayer.
        GAT 层的前向传播。
        :param x: Node features (节点特征).
        :param edge_index: Edge list [2, E] with source and target indices (边的索引矩阵，包含源和目标).
        :return: Updated node features, attention scores, and edge index.
        返回更新后的节点特征、注意力权重和边索引。
        """
        residual = x  # Save residual connection. 保存残差。
        x = x.contiguous()
        n = x.size(0)

        # Linear projection and reshape into [N, heads, out_features].
        # 线性变换后重塑为 [节点数, 注意力头数, 输出维度]。
        h = self.W(x).view(n, self.heads, self.out_features)

        src, dst = edge_index  # Edge source and target. 边的起点和终点。
        h_src = h[src]
        h_dst = h[dst]

        # Concatenate source and target node embeddings to compute attention score.
        # 拼接源点和目标点特征，用于计算注意力分数。
        alpha = torch.cat([h_src, h_dst], dim=-1)
        alpha = f.leaky_relu(self.attn(alpha), 0.2)  # LeakyReLU activation. 使用 LeakyReLU 激活。
        alpha = alpha.squeeze(-1)  # Remove last dimension. 移除最后一维。

        # Apply grouped softmax to attention weights.
        # 对注意力分数按目标节点归一化（分组 softmax）。
        alpha = self.grouped_softmax(alpha, dst)

        # Attention-weighted message aggregation.
        # 加权聚合邻居信息。
        alpha = alpha.unsqueeze(-1)
        h_src = h[src]
        out = torch.zeros(n, self.heads, self.out_features, device=x.device)
        expanded_dst = dst.view(-1, 1, 1).expand(-1, self.heads, self.out_features)

        # Scatter messages to target nodes.
        # 将消息根据目标节点索引聚合。
        out.scatter_add_(0, expanded_dst, alpha * h_src)

        # Apply residual connection and normalization.
        # 应用残差连接与归一化。
        out = out.view(n, -1)
        out = self.norm(out + self.res_gate * residual)

        return out, alpha.squeeze(-1).permute(1, 0), edge_index
        # 返回节点特征、注意力权重（[heads, edges]）和边索引。
