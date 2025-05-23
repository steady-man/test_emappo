import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import torch.optim as optim


class DiffPoolLayer(nn.Module):
    def __init__(self, input_features, output_features, num_nodes, target_nodes):
        super(DiffPoolLayer, self).__init__()
        self.sage_conv = DenseSAGEConv(input_features, output_features)
        self.assign_mat = nn.Linear(output_features, target_nodes)

    def forward(self, x, adj):
        x = F.relu(self.sage_conv(x, adj))
        s = torch.softmax(self.assign_mat(x), dim=1)
        x_pool, adj_pool, _, _ = dense_diff_pool(x, adj, s)
        return x_pool, adj_pool


class DiffPoolNet(nn.Module):
    def __init__(self, num_features, num_nodes, intermediate_nodes=1000, target_nodes=1, final_features=64):
        super(DiffPoolNet, self).__init__()
        self.pool1 = DiffPoolLayer(num_features, 128, num_nodes, intermediate_nodes)
        self.pool2 = DiffPoolLayer(128, final_features, intermediate_nodes, target_nodes)

    def forward(self, x, adj):
        x, adj = self.pool1(x, adj)
        x, adj = self.pool2(x, adj)
        return x


def process_batch_diffpool(model, batch_features, adj):
    batch_size, num_nodes, node_features = batch_features.shape

    
    pooled_features_list = []
    for i in range(batch_size):
        x_single = batch_features[i]
        x_pool = model(x_single, adj)
        for _ in range(27):
            pooled_features_list.append(x_pool)

    
    pooled_features = torch.stack(pooled_features_list)
    return torch.squeeze(pooled_features, dim=1)


if __name__ == '__main__':

    
    node_features = 6
    num_nodes = 6400
    intermediate_nodes = 1000
    target_nodes = 49
    final_features = 64

    batch_size = 12

    model = DiffPoolNet(node_features, num_nodes, intermediate_nodes, target_nodes, final_features)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  

    
    batch_features = torch.rand((batch_size, num_nodes, node_features))  
    adj = torch.rand((num_nodes, num_nodes))  
    labels = torch.randint(0, 2, (batch_size, target_nodes))  

    
    for epoch in range(10):  
        optimizer.zero_grad()  
        pooled_features = process_batch_diffpool(model, batch_features, adj)
        loss = F.cross_entropy(pooled_features.view(batch_size * target_nodes, -1), labels.view(-1))  
        loss.backward()  
        optimizer.step()  

        print(f"Epoch {epoch}, Loss: {loss.item()}")
