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
from scipy.stats import entropy
from scipy.special import rel_entr
from tqdm import tqdm
import math

import matplotlib.pyplot as plt

import layers

from utils import load_data, accuracy
from layers import GraphAttentionLayer, GraphAttentionLayerV2, SpGraphAttentionLayer, SpGraphAttentionLayerV2
from models import GAT

# Cora specific constants
CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7
# Citeseer specific constants
CITESEER_NUM_INPUT_FEATURES = 3703
CITESEER_NUM_CLASSES = 6
# Cora specific constants
PUBMED_NUM_INPUT_FEATURES = 500
PUBMED_NUM_CLASSES = 3

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--f', type=str, help='Enables running in Jupyter.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
#parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'pubmed', 'citeseer'], help='Dataset to use')
parser.add_argument('--model', type=str, default='GAT_sparse', choices=['GAT_sparse', 'GAT', 'GATv2', 'GATv2_sparse'], help='GAT model version.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')  #10000
#parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
#parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).') #5e-4
#parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).') #0.6
#parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience') #100

args = parser.parse_args()

if args.dataset == "cora":
    gat_config = {
        "nlayers": 2,
        "nheads": [8, 1],
        "nfeats": [CORA_NUM_INPUT_FEATURES, 8, CORA_NUM_CLASSES],
        "skip_connection": False,
        "alpha": 0.2,
        "dropout": 0.6,
    }
    train_config = {
        "lr": 0.005,
        "weight_decay": 5e-4,
    }
elif args.dataset == "citeseer":
    gat_config = {
        "nlayers": 2,
        "nheads": [8, 1],
        "nfeats": [CITESEER_NUM_INPUT_FEATURES, 8, CITESEER_NUM_CLASSES],
        "skip_connection": False,
        "alpha": 0.2,
        "dropout": 0.6,
    }
    train_config = {
        "lr": 0.005,
        "weight_decay": 5e-4,
    }
elif args.dataset == "pubmed":
    gat_config = {
        "nlayers": 2,
        "nheads": [8, 8],
        "nfeats": [PUBMED_NUM_INPUT_FEATURES, 8, PUBMED_NUM_CLASSES],
        "skip_connection": False,
        "alpha": 0.2,
        "dropout": 0.6,
    }
    train_config = {
        "lr": 0.01,
        "weight_decay": 0.001,
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

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)

print(f'Number of nodes: {features.shape[0]}')
print(f'Number of edges: {(torch.count_nonzero(adj).item()-features.shape[0])//2}')
print(f'Number of input features: {features.shape[1]}')
print(f'Number of classes: {torch.unique(labels).shape[0]}')
print(f'Number of training nodes: {idx_train.shape[0]}')
print(f'Number of validation nodes: {idx_val.shape[0]}')
print(f'Number of test nodes: {idx_test.shape[0]}')

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

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

# Select a backend
if args.cuda:
    device = torch.device("cuda")
elif not args.no_cuda and torch.backends.mps.is_available() and args.model not in ['GAT_sparse','GATv2_sparse']:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Move tensor to appropiate device
model = model.to(device)
features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

model_name_checkpoint = {
    "GAT_sparse": "GATsparse",
    "GATv2_sparse": "GATv2sparse",
}[args.model]

files = glob.glob(f'*_{args.dataset}_{model_name_checkpoint}.pkl')
print(files)
print(f'Using checkpoint saved at epoch: {files[0].split("_")[0]}')
checkpoint = files[0]
model.load_state_dict(torch.load(checkpoint, map_location=device))

model.eval()
output = model(features, adj)

edge = adj.nonzero().t()
N = adj.shape[0]

node_degrees = np.bincount(edge[0].detach().numpy(), minlength=N)
# Uniform entropies for each node
uniform_entropies = np.array([np.log(node_degree) if node_degree > 0 else 0. for node_degree in node_degrees])

visualisation_path = "./visualisations"
if not os.path.exists(visualisation_path):
    os.makedirs(visualisation_path)

visualisation_path += f"/{args.dataset}"
if not os.path.exists(visualisation_path):
    os.makedirs(visualisation_path)

visualisation_path += f"/{args.model}"
if not os.path.exists(visualisation_path):
    os.makedirs(visualisation_path)

# Diagram for uniform entropies
plt.figure(figsize=(10, 8))
plt.hist(uniform_entropies, color='orange', alpha=0.5, label='Uniform Entropy', bins=10)

# Adding labels and title
plt.xlabel('Entropy Bin')
plt.ylabel('# of nodes')
plt.title('Uniform entropies histogram')
plt.legend(loc='upper right')

# Display the plot
plt.savefig(f"{visualisation_path}/uniform_entropy_histogram.png", dpi=300)
plt.close()

#######
### Compute KL divergences between heads
#######

def compute_attention_matrix(edge_e, N):
    sparse_edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
    e_rowsum = torch.sparse.mm(sparse_edge_e, torch.ones(size=(N,1)))
    attention = sparse_edge_e.to_dense()/e_rowsum
    return attention.detach().numpy()

"""
attention_layers = []
for layer_name in layers.attention_weights.keys():
    layer_num = int(layer_name.split("_")[2])
    if len(attention_layers) < layer_num:
        attention_layers.append([])
    attention_layers[layer_num-1].append(layer_name)

# Number of heads to show
NUM_HEADS_PLOT = 5

# Iterate over attention layers
for layer_idx, layer_head_names in enumerate(attention_layers):
    NUM_HEADS_PLOT = min(NUM_HEADS_PLOT, len(layer_head_names))
    kl_divergences = []

    for i in tqdm(range(len(layer_head_names))):
        attention_i = compute_attention_matrix(layers.attention_weights[layer_head_names[i]], N)
        for j in range(i, len(layer_head_names)):
            attention_j = compute_attention_matrix(layers.attention_weights[layer_head_names[j]], N)
            # Compute KL divergences between the rows of each attention head
            kl_divergence = np.array([sum(rel_entr(row_1, row_2)) for row_1, row_2 in zip(attention_i, attention_j) if math.isfinite(sum(rel_entr(row_1, row_2)))])
            # Add indices and mean KL divergences
            kl_divergences.append((i, j, np.mean(kl_divergence), kl_divergence))
        
    # Sort by mean KL divergences from max to min
    kl_divergences = list(sorted(kl_divergences, key=lambda x: x[2], reverse=True))

    plt.figure(figsize=(10, 8))
    for i in range(NUM_HEADS_PLOT):
        plt.hist(kl_divergences[i][3], alpha=0.5, label=f'KL divergences between heads {kl_divergences[i][0]+1} and {kl_divergences[i][1]+1}', bins=10)

    # Adding labels and title
    plt.xlabel('KL Divergence Bin')
    plt.ylabel('# of nodes')
    plt.legend(loc='upper right')
    plt.title(f'Layer {layer_idx+1}')

    # Display the plot
    plt.savefig(f"{visualisation_path}/layer_{layer_idx+1}_kl_divergences_histogram.png", dpi=300)
    plt.close()
"""

#######
### End of code to compute KL divergences between heads
#######

for module_name, edge_e in layers.attention_weights.items():
    sparse_edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
    e_rowsum = torch.sparse.mm(sparse_edge_e, torch.ones(size=(N,1)))

    attention = sparse_edge_e.to_dense()/e_rowsum
    assert torch.allclose(torch.ones(size=(N,)), attention.sum(dim=1))

    # Detach to numpy
    attention = attention.detach().numpy()

    entropy_attention = np.array([entropy(row) for row in attention])
    # HEAT
    plt.figure(figsize=(10, 8))
    plt.hist(entropy_attention, color='blue', alpha=0.5, label='Attention Entropy', bins=10)

    # Adding labels and title
    plt.xlabel('Entropy Bin')
    plt.ylabel('# of nodes')
    plt.title(module_name)
    plt.legend(loc='upper right')

    # Display the plot
    plt.savefig(f"{visualisation_path}/{module_name}_entropy_histogram.png", dpi=300)
    plt.close()

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.hist(uniform_entropies, alpha=0.5, label='Uniform entropy')
    plt.hist(entropy_attention, alpha=0.5, label='Entropy attention', bins=10)

    # Adding labels and title
    plt.xlabel('Entropy Bin')
    plt.ylabel('# of nodes')
    plt.title(module_name)
    plt.legend(loc='upper right')

    # Display the plot
    # Display the plot
    plt.savefig(f"{visualisation_path}/{module_name}_vs_uniform_histogram.png", dpi=300)
    plt.close()

