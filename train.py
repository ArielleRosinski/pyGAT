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

from utils import load_data, accuracy
from layers import GraphAttentionLayer, GraphAttentionLayerV2, SpGraphAttentionLayer
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
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
#parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'pubmed', 'citeseer'], help='Dataset to use')
parser.add_argument('--model', type=str, default='GAT', choices=['GAT_sparse', 'GAT', 'GATv2'], help='GAT model version.')
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
elif not args.no_cuda and torch.backends.mps.is_available() and args.model != 'GAT_sparse':
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

def compute_log_logits(output):
    return F.log_softmax(F.elu(output), dim=1)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = compute_log_logits(model(features, adj))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = compute_log_logits(model(features, adj))

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = compute_log_logits(model(features, adj))
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

print("start training")
# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}_{}.pkl'.format(epoch, args.dataset))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break


    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('_')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

                
files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('_')[0])
    if epoch_nb > best_epoch:
        os.remove(file)



print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}_{}.pkl'.format(best_epoch, args.dataset))) 

# Testing
compute_test()
