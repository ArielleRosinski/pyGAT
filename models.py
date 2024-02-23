import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, GraphMultiheadAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    
class MultiLayerGAT(nn.Module):
    def __init__(self, nfeat, nheads, dropout, alpha, nlayers):
        """Dense version of GAT."""
        super(MultiLayerGAT, self).__init__()
        self.dropout = dropout

        nheads = [1] + nheads  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(nlayers):
            layer = GraphMultiheadAttentionLayer(
                in_features=nfeat[i] * nheads[i],  # consequence of concatenation
                out_features=nfeat[i+1],
                n_heads=nheads[i+1],
                dropout=dropout,
                alpha=alpha,
                activation=F.elu if i < nlayers - 1 else None,  # last layer just outputs raw scores
                concat=True if i < nlayers - 1 else False,  # last GAT layer does mean avg, the others do concat
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, x, adj):
        x, _ = self.gat_net((x, adj))
        x = F.elu(x)
        return F.log_softmax(x, dim=1)


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

