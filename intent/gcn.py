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

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self,x,adjC,adjC1,adjC2,length):

        pim = x
        semantic = x
        syntactic = x

        pim = self.conv1(pim, adjC)
        pim = self.conv2(pim, adjC)

        syntactic = self.conv1(syntactic, adjC1)
        syntactic = self.conv2(syntactic, adjC1)

        semantic = self.conv1(semantic, adjC2)
        semantic = self.conv2(semantic, adjC2)
        # stacked_tensor = torch.stack((pim, syntactic))
        # emd = torch.mean(stacked_tensor, dim=0)
        #
        # stacked_tensor = torch.stack((emd, semantic))
        # emd = torch.mean(stacked_tensor, dim=0)

        # x = syntactic[length:, :]
        # x = emd[length:, :]
        # return x
        pim = pim[length:, :]
        syntactic = syntactic[length:, :]
        semantic = semantic[length:, :]
        return pim,syntactic,semantic

