import torch


def build_station_graph(code_mapping, device):
    """
    Build an undirected graph based on station code mapping.
    根据站点的编码构建无向图结构。

    Args:
        code_mapping (dict): Mapping from station id to station group code.
                             站点编号与对应的分组编码映射（如：相同线路的编码相同）。
        device (torch.device): The device on which the edge index tensor will be stored.
                               构建的边索引张量存放的设备（如：'cuda' 或 'cpu'）。

    Returns:
        torch.LongTensor: Edge index tensor of shape [2, num_edges], stored on the given device.
                          构建好的边索引张量（形状为 [2, 边数]）。
    """
    edge_index = [[], []]  # Edge list: [source_nodes, target_nodes] 边列表：源节点和目标节点。

    code_groups = {}  # Group station ids by their shared code.  根据编码将站点分组。

    for sid, code in code_mapping.items():
        code_groups.setdefault(code, []).append(sid)

    for group in code_groups.values():
        # Create undirected edges between consecutive stations in the same group.
        # 为同一组中相邻站点创建无向边。
        for i in range(len(group) - 1):
            edge_index[0].append(group[i])
            edge_index[1].append(group[i + 1])
            edge_index[0].append(group[i + 1])
            edge_index[1].append(group[i])

    # Remove duplicate edges and move to the target device.
    # 去除重复边并转换到指定设备。
    return torch.LongTensor(edge_index).unique(dim=1).to(device)
