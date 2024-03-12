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
from torch_scatter import scatter_sum
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


from load_data_ppi import load_data_ppi
from models import GAT
from layers import GraphAttentionLayer, GraphAttentionLayerV2, SpGraphAttentionLayer, SpGraphAttentionLayerV2
from sklearn.metrics import f1_score

# PPI specific constants
PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 121

def compute_attention_matrix(edge, edge_e, N):
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
model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

visualisation_path = "./visualisations"
if not os.path.exists(visualisation_path):
    os.makdirs(visualisation_path)

visualisation_path += f"/{args.dataset}"
if not os.path.exists(visualisation_path):
    os.makedirs(visualisation_path)

model.eval()

for batch_idx, (features, gt_labels, adj) in enumerate(data_loader_test):
    # Iterate over nodes
    N = adj.shape[0]
    edge = adj.nonzero().t()
    features.requires_grad_()
    unnormalized_output = model(features, adj)
    # shape 
    temperature = 0.5
    soft_classes = torch.matmul(torch.softmax(unnormalized_output/temperature, dim=1), torch.arange(0, 121, dtype=torch.float))

    normalised_grad_norms_list = []
    for i in tqdm(range(N)):
        # Retrieve the soft class for the node i
        soft_class_i = soft_classes[i]
        # Mask to get the neighbours of node i
        mask = edge[0, :] == i
        # Extract neighbours
        neighbours_idxs = edge[1, mask]
        # unnormalised_grad_norms = torch.abs(torch.rand(len(neighbours_idxs)))
        # normalised_grad_norms = unnormalised_grad_norms / unnormalised_grad_norms.sum()
        # normalised_grad_norms_list.append(normalised_grad_norms)
        # Retrieve features gradients
        soft_class_i.backward(retain_graph=True)
        # Obtain gradients for the neighbours
        grads_features = features.grad.data[neighbours_idxs, :]
        unnormalised_grad_norms = torch.norm(grads_features, p=2, dim=1)
        normalised_grad_norms = unnormalised_grad_norms / torch.sum(unnormalised_grad_norms)
        breakpoint()
        assert abs(normalised_grad_norms.sum().item() - 1.0) < 1e-2
        # Save normalised grad norms
        normalised_grad_norms_list.append(normalised_grad_norms)
        # Reset gradients
        features.grad = None

    normalised_grad_norms_list = torch.cat(normalised_grad_norms_list, dim=0).view(-1, 1)
    
    normalised_attention_dict = {}
    # Get attentions for last head to get correlations
    for layer_name, edge_e in layers.attention_weights.items():

        e_rowsum = scatter_sum(edge_e, edge[0, :])
        # Sparse attention
        attention = edge_e / e_rowsum[edge[0, :]]

        normalised_attention_dict[layer_name] = []
        
        # Iterate over each index in the range of N
        for i in tqdm(range(N)):
            # Find the mask where indexes match the current index
            mask = edge[0, :] == i
            assert abs(attention[mask].sum().item() - 1.0) < 1e-2
            # Select the elements from E where the mask is True and convert to a list
            normalised_attention_dict[layer_name].append(attention[mask])

        current_layer = 'attention_layer_3_head_6'
        normalised_attention_list = normalised_attention_dict['attention_layer_3_head_6']

        data_x = torch.cat(normalised_attention_list, dim=0).view(-1, 1)
        data_y = torch.cat(normalised_grad_norms_list, dim=0).view(-1, 1)
        data = torch.hstack((data_x, data_y)).detach().numpy()

        np.savez('correlations_grads.npz', data)
        # data = np.array([[normalised_attention_list.tolist(), normalised_grad_tensor.tolist()] for normalised_grad_tensor, normalised_attention_tensor in zip(normalised_grad_norms_list[:1000], normalised_attention_list[:1000])])
        # Scatter plot
        # Splitting the data into x and y components for regression
        x = data[:, 0].reshape(-1, 1)  # Feature matrix (needs to be 2D for sklearn)
        
        y = data[:, 1]  # Target variable

        # Perform linear regression
        model = LinearRegression().fit(x, y)

        # Predict y values
        y_pred = model.predict(x)

        # Calculate R^2 coefficient
        r2 = r2_score(y, y_pred)

        # Plotting the scatter plot
        plt.scatter(x, y, color='blue', s=8, label='Data points')

        # Plotting the regression line
        plt.plot(x, y_pred, color='green', linewidth=1, label=f'Regression line\n$R^2={r2:.2f}$')

        # Adding labels and title
        plt.xlabel('Attention Coefficients')
        plt.ylabel('Normalised Gradients')
        plt.title('Scatter Plot Attention Coefficients vs Normalised Gradients with $R^2$')
        plt.legend()

        plt.show()



