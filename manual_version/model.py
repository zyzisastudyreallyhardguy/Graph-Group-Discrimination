import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        # assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, out_channels)]
        # for _ in range(1, k-1):
        #     self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        # self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder

        self.lin = torch.nn.Linear(num_hidden, num_hidden)

    def forward(self, x_1: torch.Tensor,  x_2: torch. Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_1 = self.encoder(x_1, edge_index)
        h_2 = self.encoder(x_2, edge_index)

        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

    def embed(self, x, edge_index, sp_adj):
        h_1 = self.encoder(x, edge_index)

        h_2 = h_1.clone().squeeze(0)
        for i in range(5):
            h_2 = sp_adj @ h_2

        h_2 = h_2.unsqueeze(0)

        return h_1.detach(), h_2.detach()

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
