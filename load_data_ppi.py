import json
import os
import enum

# Visualization related imports
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import igraph as ig
import zipfile

# Main computation libraries
import numpy as np
import scipy.sparse as sp

# Deep learning related imports
import torch
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Dataset

# Function to process adjacency matrix
from utils import normalize_adj

# We'll be dumping and reading the data from this directory
DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')
PPI_PATH = os.path.join(DATA_DIR_PATH, 'ppi')
PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'  # preprocessed PPI data from Deep Graph Library

#
# PPI specific constants
#
PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 121

# First let's define this simple function for loading PPI's graph data
def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)

    return data

class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).

    """
    def __init__(self, node_features_list, node_labels_list, adjacency_matrix_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, adjacency_matrix_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it

    """
    def __init__(self, node_features_list, node_labels_list, adjacency_matrix_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.adjacency_matrix_list = adjacency_matrix_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.node_features_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.node_features_list[idx], self.node_labels_list[idx], self.adjacency_matrix_list[idx]


def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    the scatter functions in the implementation 3 will fail.

    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """

    node_features_list, node_labels_list, adjacency_matrix_list = zip(*batch)

    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    adjacency_matrix = torch.block_diag(*adjacency_matrix_list)

    return node_features, node_labels, adjacency_matrix

def load_data_ppi(batch_size, device):
    # Instead of checking PPI in, I'd rather download it on-the-fly the first time it's needed (lazy execution ^^)
    if not os.path.exists(PPI_PATH):  # download the first time this is ran
        os.makedirs(PPI_PATH)

        # Step 1: Download the ppi.zip (contains the PPI dataset)
        zip_tmp_path = os.path.join(PPI_PATH, 'ppi.zip')
        download_url_to_file(PPI_URL, zip_tmp_path)

        # Step 2: Unzip it
        with zipfile.ZipFile(zip_tmp_path) as zf:
            zf.extractall(path=PPI_PATH)
        print(f'Unzipping to: {PPI_PATH} finished.')

        # Step3: Remove the temporary resource file
        os.remove(zip_tmp_path)
        print(f'Removing tmp file {zip_tmp_path}.')

    # Collect train/val/test graphs here
    adjacency_matrix_list = []
    node_features_list = []
    node_labels_list = []

    # Dynamically determine how many graphs we have per split (avoid using constants when possible)
    num_graphs_per_split_cumulative = [0]

    # Small optimization "trick" since we only need test in the playground.py
    splits = ['train', 'valid', 'test']

    for split in splits:
        # PPI has 50 features per node, it's a combination of positional gene sets, motif gene sets,
        # and immunological signatures - you can treat it as a black box (I personally have a rough understanding)
        # shape = (NS, 50) - where NS is the number of (N)odes in the training/val/test (S)plit
        # Note: node features are already preprocessed
        node_features = np.load(os.path.join(PPI_PATH, f'{split}_feats.npy'))

        # PPI has 121 labels and each node can have multiple labels associated (gene ontology stuff)
        # SHAPE = (NS, 121)
        node_labels = np.load(os.path.join(PPI_PATH, f'{split}_labels.npy'))

        # Graph topology stored in a special nodes-links NetworkX format
        nodes_links_dict = json_read(os.path.join(PPI_PATH, f'{split}_graph.json'))
        # PPI contains undirected graphs with self edges - 20 train graphs, 2 validation graphs and 2 test graphs
        # The reason I use a NetworkX's directed graph is because we need to explicitly model both directions
        # because of the edge index and the way GAT implementation #3 works
        collection_of_graphs = nx.DiGraph(json_graph.node_link_graph(nodes_links_dict))
        # For each node in the above collection, ids specify to which graph the node belongs to
        graph_ids = np.load(os.path.join(PPI_PATH, F'{split}_graph_id.npy'))
        num_graphs_per_split_cumulative.append(num_graphs_per_split_cumulative[-1] + len(np.unique(graph_ids)))

        # Split the collection of graphs into separate PPI graphs
        for graph_id in range(np.min(graph_ids), np.max(graph_ids) + 1):
            mask = graph_ids == graph_id  # find the nodes which belong to the current graph (identified via id)
            graph_node_ids = np.asarray(mask).nonzero()[0]
            graph = collection_of_graphs.subgraph(graph_node_ids)  # returns the induced subgraph over these nodes
            print(f'Loading {split} graph {graph_id} to CPU. '
                    f'It has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')
            
            # shape = (N, N) - an entry of 1 in position (i,j) means that there is a connection between nodes 
            adjacency_matrix = nx.adjacency_matrix(graph)
            # row-normalize sparse adjacency matrix
            adjacency_matrix = normalize_adj(adjacency_matrix + sp.eye(adjacency_matrix.shape[0]))
            # obtain dense representation
            adjacency_matrix = torch.tensor(np.array(adjacency_matrix.todense()), dtype=torch.float).to(device)
            adjacency_matrix_list.append(adjacency_matrix)

            # Verify that the adjacency matrix is symmetric
            assert torch.allclose(adjacency_matrix, adjacency_matrix.t())

            # shape = (N, 50) - where N is the number of nodes in the graph
            node_features_list.append(torch.tensor(node_features[mask], dtype=torch.float).to(device))
            # shape = (N, 121), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
            node_labels_list.append(torch.tensor(node_labels[mask], dtype=torch.float).to(device))

    #
    # Prepare graph data loaders
    #

    data_loader_train = GraphDataLoader(
        node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
        node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
        adjacency_matrix_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
        batch_size=batch_size,
        shuffle=True
    )

    data_loader_val = GraphDataLoader(
        node_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
        node_labels_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
        adjacency_matrix_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
        batch_size=batch_size,
        shuffle=False  # no need to shuffle the validation and test graphs
    )

    data_loader_test = GraphDataLoader(
        node_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
        node_labels_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
        adjacency_matrix_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
        batch_size=batch_size,
        shuffle=False
    )

    return data_loader_train, data_loader_val, data_loader_test