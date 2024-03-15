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
import math

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
parser.add_argument('--model', type=str, default='GAT_sparse', choices=['GAT_sparse', 'GAT', 'GATv2', 'GATv2_sparse'], help='GAT model version.')
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
    unnormalized_output = model(features, adj)

    edge = adj.nonzero().t()
    N = adj.shape[0]

    ######
    # Code to compute KL divergences between attention heads
    ######
    attention_layers = []
    for layer_name in layers.attention_weights.keys():
        layer_num = int(layer_name.split("_")[2])
        if len(attention_layers) < layer_num:
            attention_layers.append([])
        attention_layers[layer_num-1].append(layer_name)

    node_degrees = np.bincount(edge[0].detach().numpy(), minlength=N)
    # Uniform entropies for each node
    uniform_entropies = np.array([np.log(node_degree) if node_degree > 0 else 0. for node_degree in node_degrees])


    for layer_num, layer_name_list in enumerate(attention_layers):
        plt.figure(figsize=(10, 8))
        plt.hist(uniform_entropies, color='orange', alpha=0.3, label='Uniform entropy', bins=20)
        for model_name, color in zip(['GATv2_sparse', 'GAT_sparse'],['red','blue']):
            entropy_attention = None
            for layer_name in layer_name_list:
                attention = np.load(f"{visualisation_path}/{model_name}/graph_{batch_idx}_{layer_name}_attention_coefficients.npz")["arr_0"]
                entropy_attention_layer = np.array([entropy(row) for row in attention])
                entropy_attention = entropy_attention_layer if entropy_attention is None else entropy_attention_layer + entropy_attention
            # Mean attention
            entropy_attention /= len(layer_name_list)

            plt.hist(entropy_attention, color=color, alpha=0.3, label=f'{model_name .split("_")[0]} Attention Entropy', bins=20)
        
        # Adding labels and title
        plt.xlabel('Entropy Bin', fontsize=15)
        plt.ylabel('# of nodes',  fontsize=15)
        # plt.title(f"Layer {layer_num+1}",  fontsize=15)
        # Setting the font size for the tick labels on both axes
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(loc='upper right', fontsize=15)

        ax = plt.gca()  # Get the current Axes instance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Display the plot
        plt.savefig(f"{visualisation_path}/graph_{batch_idx+1}_layer_{layer_num+1}_entropy_histogram.png", dpi=300)
        plt.close()