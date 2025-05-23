import torch
from torch_geometric.data import Data


def create_adjacency_diffpool(num_nodes=10000, grid_size=100):
    
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int)

    
    for node in range(num_nodes):
        row, col = divmod(node, grid_size)

        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                neighbor_node = new_row * grid_size + new_col
                
                adjacency_matrix[node, neighbor_node] = 1
                adjacency_matrix[neighbor_node, node] = 1  

    return adjacency_matrix













