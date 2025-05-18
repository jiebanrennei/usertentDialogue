import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        deg_inv_sqrt = torch.pow(torch.sum(adj, dim=1), -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = torch.mul(torch.mul(adj, deg_inv_sqrt.unsqueeze(1)), deg_inv_sqrt.unsqueeze(0))
        x = torch.matmul(norm_adj, x)
        x = self.linear(x)
        x = torch.relu(x)
        return x

class GCN1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN1, self).__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, output_dim)
    def forward(self,x,adj):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        stacked_tensor = torch.stack((x[0], x[1]))
        emd = torch.mean(stacked_tensor, dim=0)
        stacked_tensor = torch.stack((emd, x[2]))
        emd = torch.mean(stacked_tensor, dim=0)
        return emd

