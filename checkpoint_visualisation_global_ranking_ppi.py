from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import layers
from scipy.stats import entropy
from scipy.special import rel_entr
from tqdm import tqdm
import networkx as nx

from load_data_ppi import load_data_ppi
from models import GAT
from layers import GraphAttentionLayer, GraphAttentionLayerV2, SpGraphAttentionLayer, SpGraphAttentionLayerV2
from sklearn.metrics import f1_score

# PPI specific constants
PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 121

def compute_attention_matrix(edge_e, N):
    sparse_edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
    e_rowsum = torch.sparse.mm(sparse_edge_e, torch.ones(size=(N,1)))
    attention = sparse_edge_e.to_dense()/e_rowsum
    return attention.detach().numpy()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
#parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--dataset', type=str, default='ppi', choices=['ppi'], help='Dataset to use')
parser.add_argument('--model', type=str, default='GATv2_sparse', choices=['GAT_sparse', 'GAT', 'GATv2', 'GATv2_sparse'], help='GAT model version.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
#parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
#parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
#parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
#parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--batch_size', type=int, default=1, help='Number of graphs that are passed during training')

args = parser.parse_args()

if args.dataset == "ppi":
    gat_config = {
        "nlayers": 3,
        "nheads": [4, 4, 6],
        "nfeats": [PPI_NUM_INPUT_FEATURES, 256, 256, PPI_NUM_CLASSES],
        "skip_connection": True,
        "alpha": 0.2,
        "dropout": 0.0,
    }
    train_config = {
        "lr": 0.005,
        "weight_decay": 0.0,
    }
else:
    raise ValueError("Dataset not known")

args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Select a backend
if args.cuda:
    device = torch.device("cuda")
elif not args.no_cuda and torch.backends.mps.is_available() and args.model not in ['GAT_sparse','GATv2_sparse']:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load data - Use batch size 1
data_loader_train, data_loader_val, data_loader_test = load_data_ppi(args.batch_size, device)

# Select the appropiate layer type
layer_type = {
    'GAT_sparse': SpGraphAttentionLayer,
    'GAT': GraphAttentionLayer,
    'GATv2': GraphAttentionLayerV2,
    'GATv2_sparse': SpGraphAttentionLayerV2,
}[args.model]

# Create model
model = GAT(nfeat=gat_config["nfeats"], 
            nheads=gat_config["nheads"],
            nlayers=gat_config["nlayers"],
            dropout=gat_config["dropout"],
            alpha=gat_config["alpha"],
            layer_type=layer_type,
            skip_connection=gat_config["skip_connection"])

optimizer = optim.Adam(model.parameters(), 
                       lr=train_config["lr"], 
                       weight_decay=train_config["weight_decay"])

# Move the model to the appropiate device
model = model.to(device)

model_name_checkpoint = {
    "GAT_sparse": "GATsparse",
    "GATv2_sparse": "GATv2sparse",
}[args.model]

files = glob.glob(f'*_{args.dataset}_{model_name_checkpoint}.pkl')
print(files)
print(f'Using checkpoint saved at epoch: {files[0].split("_")[0]}')
checkpoint = files[0]
model.load_state_dict(torch.load(checkpoint, map_location=device))

visualisation_path = "./visualisations"
if not os.path.exists(visualisation_path):
    os.makedirs(visualisation_path)

visualisation_path += f"/{args.dataset}"
if not os.path.exists(visualisation_path):
    os.makedirs(visualisation_path)

model.eval()

for batch_idx, (features, gt_labels, adj) in enumerate(data_loader_test):
    # Record gradients w.r.t to features
    unnormalized_output = model(features, adj)

    edge = adj.nonzero().t()
    N = adj.shape[0]

    # Convert the adjacency matrix to a NetworkX graph
    G = nx.from_numpy_array(adj.detach().numpy())
    # Find cliques in the graph (fully connected subgraphs)
    cliques = list(nx.find_cliques(G))
    # Select the largest clique (fully connected subset of nodes)
    # Note: This selects one of the largest cliques if there are multiple
    largest_clique = max(cliques, key=len)

    for edge_e in layers.attention_weights.values():

        attention = compute_attention_matrix(edge_e, N)

        adversarial_indices = list(sorted(largest_clique))
        plt.figure(figsize=(10, 8))
        attention = attention[np.ix_(adversarial_indices, adversarial_indices)]
        plt.imshow(attention, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Add a colorbar to a plot
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Query Index')
        plt.ylabel('Key Index')
        plt.show()


        attention_sorted = []
        partial_orderings = dict()
        global_ranking_violations = 0
        adversarial_indices = []
        for row_idx, row in tqdm(enumerate(attention)):
            row_list = [(idx, att_coef) for idx, att_coef in enumerate(row) if att_coef > 0.0]
            row_list_sorted = list(sorted(row_list, key=lambda x: x[1], reverse=True))

            for i in range(len(row_list_sorted)):
                for j in range(i+1,len(row_list_sorted)):
                    order_str = f"{row_list_sorted[i][0]}>{row_list_sorted[j][0]}"
                    partial_orderings[order_str]=(i,j)

                    reverse_order_str = f"{row_list_sorted[j][0]}>{row_list_sorted[i][0]}"

                    if reverse_order_str in partial_orderings:
                        global_ranking_violations += 1
                        if global_ranking_violations < 5:
                            adversarial_indices += [
                                partial_orderings[reverse_order_str][0],
                                partial_orderings[reverse_order_str][1],
                                i,
                                j
                            ]

            attention_sorted.append(row_list_sorted)

        print(f'global ranking violations: {global_ranking_violations}')
        print(attention.shape)
        print(len(adversarial_indices))
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(attention[np.ix_(adversarial_indices, adversarial_indices)], cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Add a colorbar to a plot
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Query Index')
        plt.ylabel('Key Index')
        plt.show()

        adversarial_indices = list(sorted(adversarial_indices))
        plt.figure(figsize=(10, 8))
        attention = attention[np.ix_(adversarial_indices, adversarial_indices)]
        for i in range(len(adversarial_indices)):
            plt.plot(adversarial_indices, attention[i, :])
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Query Index')
        plt.ylabel('Key Index')
        plt.show()