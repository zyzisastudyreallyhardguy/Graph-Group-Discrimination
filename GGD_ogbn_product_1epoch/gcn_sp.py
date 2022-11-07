"""
This code was copied from the GCN implementation in DGL examples.
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 bias = True,
                 weight=True):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.bns = torch.nn.ModuleList()
        self.res_linears = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, weight = weight, bias = bias, activation=activation))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum = 0.01))
        self.res_linears.append(torch.nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, weight=weight, bias=bias, activation=activation))
            # if i != (n_layers - 2):
            self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum = 0.01))
            self.res_linears.append(torch.nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        # self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum=0.01))
        self.res_linears.append(torch.nn.Identity())
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks):
        collect = []
        h = blocks[0].srcdata['feat']
        h = self.dropout(h)
        num_output_nodes = blocks[-1].num_dst_nodes()
        collect.append(h[:num_output_nodes])
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_res = h[:block.num_dst_nodes()]
            h = layer(block, h)
            # if l < len(self.layers) - 1:
            #     h = self.bns[l](h)
            h = self.dropout(h)
            collect.append(h[:num_output_nodes])
            h += self.res_linears[l](h_res)
        # return torch.cat(collect, -1)
        return collect[-1]

        # h = features
        # for i, layer in enumerate(self.layers):
        #     if i != 0:
        #         h = self.dropout(h)
        #     h = layer(self.g, h)
        # return h

