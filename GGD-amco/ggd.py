"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import torch
import torch.nn as nn
import math
from gcn import GCN
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv
import dgl.function as fn

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, k = 1):
        super(Encoder, self).__init__()
        self.g = g
        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'gcn':
            self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        elif gnn_encoder == 'sgc':
            self.conv = SGConv(in_feats, n_hidden, k=10, cached=True)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        if self.gnn_encoder == 'gcn':
            features = self.conv(features)
        elif self.gnn_encoder == 'sgc':
            features = self.conv(self.g, features)
        return features


# class Discriminator(nn.Module):
#     def __init__(self, n_hidden):
#         super(Discriminator, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
#         self.reset_parameters()
#
#     def uniform(self, size, tensor):
#         bound = 1.0 / math.sqrt(size)
#         if tensor is not None:
#             tensor.data.uniform_(-bound, bound)
#
#     def reset_parameters(self):
#         size = self.weight.size(0)
#         self.uniform(size, self.weight)
#
#     def forward(self, features, summary):
#         features = torch.matmul(features, torch.matmul(self.weight, summary))
#         return features


class GGD(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, proj_layers, gnn_encoder, num_hop):
        super(GGD, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, num_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, n_hidden))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features, labels, loss_func):
        h_1 = self.encoder(features, corrupt=False)
        h_2 = self.encoder(features, corrupt=True)

        # sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        # sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, labels)

        return loss

    def embed(self, features, g):
        h_1 = self.encoder(features, corrupt=False)

        feat = h_1.clone().squeeze(0)

        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(h_1.device).unsqueeze(1)
        for _ in range(10):
            feat = feat * norm
            g.ndata['h2'] = feat
            g.update_all(fn.copy_u('h2', 'm'),
                             fn.sum('m', 'h2'))
            feat = g.ndata.pop('h2')
            feat = feat * norm

        # for i in range(10):
        #     h_2 = adj @ h_2

        h_2 = feat.unsqueeze(0)

        return h_1.detach(), h_2.detach()

class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)
