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

from load_data_ppi import load_data_ppi
from models import GAT
from layers import GraphAttentionLayer, GraphAttentionLayerV2, SpGraphAttentionLayer
from sklearn.metrics import f1_score

# PPI specific constants
PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 121

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
#parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--dataset', type=str, default='ppi', choices=['ppi'], help='Dataset to use')
parser.add_argument('--model', type=str, default='GAT_sparse', choices=['GAT_sparse', 'GAT', 'GATv2'], help='GAT model version.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
#parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
#parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
#parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
#parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--batch_size', type=int, default=2, help='Number of graphs that are passed during training')

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
#print(torch.cuda.is_available())
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
elif not args.no_cuda and torch.backends.mps.is_available() and args.model != 'GAT_sparse':
    device = torch.device("mps")
else:
    device = torch.device("cpu")
#print(device)
# Load data 
data_loader_train, data_loader_val, data_loader_test = load_data_ppi(args.batch_size, device)

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

# Move the model to the appropiate device
model = model.to(device)

def compute_f1_score(unnormalized_output, gt_labels):
    pred = (unnormalized_output > 0).float().cpu().numpy()
    gt = gt_labels.cpu().numpy()
    micro_f1_score = f1_score(gt, pred, average='micro')
    return micro_f1_score

def train(epoch):
    t = time.time()
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    model.train()

    for batch_idx, (features, gt_labels, adj) in enumerate(data_loader_train):        
        unnormalized_output = model(features, adj)
        loss_train = loss_fn(unnormalized_output, gt_labels)
        f1_train = compute_f1_score(unnormalized_output, gt_labels)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        print('[Train] Epoch: {:04d}'.format(epoch+1),
            'Batch: {:04d}'.format(batch_idx+1),
            'loss_train: {:.4f}'.format(loss_train.data.item()),
            'f1_train: {:.4f}'.format(f1_train),
            'time: {:.4f}s'.format(time.time() - t))
        
        t = time.time()

    model.eval()

    loss_val_list = []
    for batch_idx, (features, gt_labels, adj) in enumerate(data_loader_val):
        unnormalized_output = model(features, adj)
        loss_val = loss_fn(unnormalized_output, gt_labels)
        f1_val = compute_f1_score(unnormalized_output, gt_labels)

        print('[Val] Epoch: {:04d}'.format(epoch+1),
            'Batch: {:04d}'.format(batch_idx+1),
            'loss_val: {:.4f}'.format(loss_val.data.item()),
            'f1_val: {:.4f}'.format(f1_val),
            'time: {:.4f}s'.format(time.time() - t))
        
        t = time.time()

        loss_val_list.append(loss_val.data.item())

    return sum(loss_val_list)/len(loss_val_list)


def compute_test():
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    loss_test_list = []
    unnormalized_output_list = []
    gt_labels_list = []

    for batch_idx, (features, gt_labels, adj) in enumerate(data_loader_test):
        unnormalized_output = model(features, adj)
        loss_test = loss_fn(unnormalized_output, gt_labels)

        loss_test_list.append(loss_test.data.item())
        unnormalized_output_list.append(unnormalized_output)
        gt_labels_list.append(gt_labels)

    loss_test = sum(loss_test_list)/len(loss_test_list)
    f1_test = compute_f1_score(torch.cat(unnormalized_output_list, dim=0), torch.cat(gt_labels_list, dim=0))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(f1_test))

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
model.load_state_dict(torch.load('{}_ppi.pkl'.format(best_epoch)))

# Testing
compute_test()