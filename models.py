import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, dropout, alpha, nheads, nlayers):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        nheads = [1] + nheads
        self.gat_layers = []

        for i in range(nlayers):
            self.gat_layers.append([])
            for j in range(nheads[i+1]): 
                layer = GraphAttentionLayer(
                    in_features=nfeat[i] * nheads[i], 
                    out_features=nfeat[i+1], 
                    dropout=dropout, 
                    alpha=alpha,
                    concat=True if i < nlayers - 1 else False,
                    skip_connection = True,
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
    
class GAT_PPI(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers):
        """Dense version of GAT."""
        super(GAT_PPI, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_1_{}'.format(i), attention)

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_1_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.out_att(x, adj)

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

