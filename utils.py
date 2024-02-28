import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl.data import PubmedGraphDataset
from dgl import AddSelfLoop
from dgl import to_networkx


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if dataset == "cora" or dataset == "citeseer":
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    if dataset == "cora":
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

    elif dataset == "citeseer":
        idx = idx_features_labels[:, 0]
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),  dtype=np.dtype(str))
        edges_temp = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.dtype(str)).reshape(edges_unordered.shape)
        edges_temp[edges_temp == 'None'] = '-1'  
        edges_temp = edges_temp.astype(np.int32)
        rows_without_none = ~np.any(edges_temp == -1, axis=1)
        edges = edges_temp[rows_without_none]
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # idx_train = range(120)
        # idx_val = range(200, 700)
        # idx_test = range(800, 1800)
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    
    elif dataset == "pubmed":
        transform = AddSelfLoop()  # Add self-loops
        data = PubmedGraphDataset(transform=transform)
        g = data[0]  # Get the first graph object from the dataset
        features = g.ndata['feat']  # Node features
        labels = g.ndata['label']  # Node labels

        idx_train = torch.nonzero(g.ndata['train_mask'], as_tuple=False).squeeze()
        idx_val = torch.nonzero(g.ndata['val_mask'], as_tuple=False).squeeze()
        idx_test = torch.nonzero(g.ndata['test_mask'], as_tuple=False).squeeze()
        
        src, dst = g.edges()
        num_nodes = g.num_nodes()

        adj = sp.coo_matrix((torch.ones(src.shape[0]), (src.numpy(), dst.numpy())),
                                shape=(num_nodes, num_nodes),
                                dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))


    adj = torch.FloatTensor(np.array(adj.todense()))
   

    if dataset == "citeseer" or dataset == "cora":
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
    
    elif dataset == "pubmed":
        features = torch.FloatTensor(np.array(features))
    print("all good")

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels)
    correct = correct.sum()
    return correct / len(labels)

