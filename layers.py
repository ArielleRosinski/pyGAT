import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, skip_connection=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.skip_connection = skip_connection

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #self.cnt = 0

        if self.skip_connection:
            self.skip_projection = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.xavier_uniform_(self.skip_projection.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Add dropout to the inputs
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        #print(Wh1.size())

        # In the official repo, they add dropout after the linear projection
        Wh = F.dropout(Wh, self.dropout, training=self.training)

        e = self._prepare_attentional_mechanism_input(Wh)

        # Applying softmax and save the attention scores
        # The last epoch will be saved as the file is being continously overwritten
        # I once also tried to save the attention weights in every epoch,
        # then only preserve the one in the best epoch,
        # but I found that takes a lot of time and might not be necessary
        Wh_att = F.softmax(e,dim=1)
        #print(Wh_att.size())
        new_directory = './attention'
        os.makedirs(new_directory, exist_ok=True)
        path = './attention/attention.pt'
        torch.save(Wh_att, path)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        #print(attention)


        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        #print(h_prime)
        # Add skip connection
        if self.skip_connection:
            h_prime += torch.mm(h, self.skip_projection)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        #print(Wh2.size())
        # broadcast add
        e = Wh1 + Wh2.T
        ''''
        Wh_att = F.softmax(e, dim=1)
        #print(Wh_att)

        new_directory = './attention'
        os.makedirs(new_directory, exist_ok=True)
        path = f'./attention/attention.pt'
        torch.save(Wh_att, path)
        '''
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, skip_connection=False):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.skip_connection = skip_connection
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        if self.skip_connection:
            self.skip_projection = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.xavier_uniform_(self.skip_projection.data, gain=1.414)

        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        # Add dropout to the inputs
        h = F.dropout(input, self.dropout, training=self.training)
        # Apply linear projection
        Wh = torch.mm(h, self.W)
        # In the official repo, they add dropout after the linear projection
        Wh = F.dropout(Wh, self.dropout, training=self.training)
        # h: N x out
        assert not torch.isnan(Wh).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((Wh[edge[0, :], :], Wh[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = F.dropout(edge_e, self.dropout, training=self.training)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), Wh)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        # Add skip connection
        if self.skip_connection:
            h_prime += torch.mm(h, self.skip_projection)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayerV2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, skip_connection=False):
        super(GraphAttentionLayerV2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.skip_connection = skip_connection

        self.W = nn.Parameter(torch.empty(size=(2*in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.skip_connection:
            self.skip_projection = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.xavier_uniform_(self.skip_projection.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Apply dropout to inputs
        h = F.dropout(h, self.dropout, training=self.training)
        # Project features
        Wh1 = torch.matmul(h,self.W[:self.in_features, :]) #[N, F] @ [F , F'] --> [N, F']
        Wh2 = torch.matmul(h,self.W[self.in_features:, :])
        #print(Wh2.size())
        # Apply dropout to linear projections
        Wh1 = F.dropout(Wh1, self.dropout, training=self.training) # 2708*8 or 2708*7
        Wh2 = F.dropout(Wh2, self.dropout, training=self.training) # 2708*8 or 2708*7

        Wh1t = Wh1.unsqueeze(2)  #2708*8*1 or 2708*7*1
        Wh2t = Wh2.T             #8*2708 or 7*2708
        Wh2t = Wh2t.unsqueeze(0) #1*8*2708 or 1*7*2708

        e = Wh1t + Wh2t  # 2708*8*2708 or 2708*7*2708

        e = self.leakyrelu(e) # 2708*8*2708 or 2708*7*2708
        # dot product operation: i=2708, j=8, k=2708, b=1
        e = torch.einsum('ijk,jb->ik', e, self.a)
        #print(e.size())
        ''''
        # another way of implementing the dot product, same result
        sz = Wh1.size(dim=1)
        aa = self.a.repeat_interleave(2708, dim=0).view(1, sz, 2708).repeat(2708, 1, 1)
        #print(aa.size())
        e = e*aa
        #e = torch.matmul(e, self.a) # size 2708*1                        #[N=2708, F'=64] @ [F',1] --> [N, 1]
        #print(e.size())
        e = e.sum(dim=1)
        #print(e.size())
        #e = torch.squeeze(e,dim=1)
        #print(e.size())
        '''
        # Applying softmax and save the attention scores
        # The last epoch will be saved as the file is being continously overwritten
        # I once also tried to save the attention weights in every epoch,
        # then only preserve the one in the best epoch,
        # but I found that takes a lot of time and might not be necessary
        Wh_att = F.softmax(e, dim=1)
        new_directory = './attention'
        os.makedirs(new_directory, exist_ok=True)
        path = f'./attention/attentionv2.pt'
        torch.save(Wh_att, path)

        zero_vec = -9e15*torch.ones_like(e)
        #print(zero_vec.size())
        attention = torch.where(adj > 0, e, zero_vec)
        #print(attention.size())
        attention = F.softmax(attention, dim=1)


        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh2)

        # Add skip connection
        if self.skip_connection:
            h_prime += torch.mm(h, self.skip_projection)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
