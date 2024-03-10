import pickle
import torch
import matplotlib.pyplot as plt
import numpy
import os
from utils import load_data, accuracy
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
import models
from layers import GraphAttentionLayer, GraphAttentionLayerV2, SpGraphAttentionLayer, SpGraphAttentionLayerV2
from sklearn.manifold import TSNE
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset='cora')

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

device = torch.device("cuda")

features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)

#print(features.size())

h = features


CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7

gat_config = {
    "nlayers": 2,
    "nheads": [8, 1],
    "nfeats": [CORA_NUM_INPUT_FEATURES, 8, CORA_NUM_CLASSES],
    "skip_connection": False,
    "alpha": 0.2,
    "dropout": 0.6,
}
# inputs = torch.randn(100, 50).cuda()
# adj = torch.randn(100, 100).cuda()
model = models.GAT(nfeat=gat_config["nfeats"],
            nheads=gat_config["nheads"],
            nlayers=gat_config["nlayers"],
            dropout=gat_config["dropout"],
            alpha=gat_config["alpha"],
            layer_type= SpGraphAttentionLayer,
            skip_connection=gat_config["skip_connection"])



model = model.cuda()

model.load_state_dict(torch.load('./294_cora_GATsparse.pkl'))

with torch.no_grad():
    # Step 3: Run predictions and collect the high dimensional data
    all_nodes_unnormalized_scores = model(h, adj)  # shape = (N, num of classes)
    #print(np.size(all_nodes_unnormalized_scores))
    all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

sum_row = numpy.max(numpy.absolute(all_nodes_unnormalized_scores), axis=1)
#print(s)
idx = np.where(sum_row == 0)[0]
sum_row[idx] = 0.1

#all_nodes_normalized_scores = all_nodes_unnormalized_scores/s[:,numpy.newaxis]
all_nodes_normalized_scores = numpy.divide(all_nodes_unnormalized_scores, sum_row[:, None])
#print(all_nodes_normalized_scores)
node_labels = labels.cpu()
num_classes = CORA_NUM_CLASSES

# Feel free to experiment with perplexity it's arguable the most important parameter of t-SNE and it basically
# controls the standard deviation of Gaussians i.e. the size of the neighborhoods in high dim (original) space.
# Simply put the goal of t-SNE is to minimize the KL-divergence between joint Gaussian distribution fit over
# high dim points and between the t-Student distribution fit over low dimension points (the ones we're plotting)
# Intuitively, by doing this, we preserve the similarities (relationships) between the high and low dim points.
# This (probably) won't make much sense if you're not already familiar with t-SNE, God knows I've tried. :P
t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(all_nodes_normalized_scores)
#print(t_sne_embeddings)
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
for class_id in range(num_classes):
    # We extract the points whose true label equals class_id and we color them in the same way, hopefully
    # they'll be clustered together on the 2D chart - that would mean that GAT has learned good representations!
    plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=20, color=cora_label_to_color_map[class_id], edgecolors='black', linewidths=0.2)
plt.show()
plt.savefig('t-sne.png')