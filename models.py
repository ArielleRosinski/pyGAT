import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, GraphAttentionLayerV2


class GAT(nn.Module):
    def __init__(self, nfeat, nheads, nlayers, dropout, alpha, layer_type=GraphAttentionLayer, skip_connection=False):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        nheads = [1] + nheads
        self.gat_layers = []      

        for i in range(nlayers):
            self.gat_layers.append([])
            for j in range(nheads[i+1]): 
                layer = layer_type(
                    name='attention_layer_{}_head_{}'.format(i+1,j+1),
                    in_features=nfeat[i] * nheads[i], 
                    out_features=nfeat[i+1], 
                    dropout=dropout, 
                    alpha=alpha,
                    concat=True if i < nlayers - 1 else False,
                    skip_connection = skip_connection,
                )
                self.gat_layers[i].append(layer)
                self.add_module('attention_layer_{}_head_{}'.format(i+1,j+1), layer)

    def forward(self, x, adj):
        for i in range(len(self.gat_layers)):
            if i < len(self.gat_layers) - 1:
                x = torch.cat([att(x, adj) for att in self.gat_layers[i]], dim=1)
            else:
                x = torch.mean(torch.stack([att(x, adj) for att in self.gat_layers[i]], dim=1), dim=1)
        return x
