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
from gcn_sp import GCN
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv
import dgl.function as fn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, k = 1):
        super(Encoder, self).__init__()
        self.g = g
        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'gcn':
            self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        elif gnn_encoder == 'sgc':
            self.conv = SGConv(in_feats, n_hidden, k=10, cached=True)

    def forward(self, blocks, corrupt=False):
        if corrupt:
            for block in blocks:
                block.ndata['feat']['_N'] = block.ndata['feat']['_N'][torch.randperm(block.num_src_nodes())]
        if self.gnn_encoder == 'gcn':
            features = self.conv(blocks)
        elif self.gnn_encoder == 'sgc':
            features = self.conv(self.g, blocks)
        return features

class GGD(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, proj_layers, gnn_encoder, num_hop):
        super(GGD, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, num_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, n_hidden))
        self.loss = nn.BCEWithLogitsLoss()
        self.graphconv = GraphConv(in_feats, n_hidden, weight=False, bias=False, activation=None)

    def forward(self, features, labels, loss_func):
        h_1 = self.encoder(features, corrupt=False)
        h_2 = self.encoder(features, corrupt=True)

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        lbl_1 = torch.ones(1, sc_1.shape[1])
        lbl_2 = torch.zeros(1, sc_1.shape[1])
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, lbl)

        return loss

    def embed(self, blocks):
        h_1 = self.encoder(blocks, corrupt=False)

        return h_1.detach()

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
