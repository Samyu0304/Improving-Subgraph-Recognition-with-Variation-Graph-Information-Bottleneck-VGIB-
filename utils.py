"""Data reading utils."""

import json
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from texttable import Texttable

def hierarchical_graph_reader(path):
    """
    Reading the macro-level graph from disk.
    :param path: Path to the edge list.
    :return graph: Hierarchical graph as a NetworkX object.
    """
    edges = pd.read_csv(path).values.tolist()
    graph = nx.from_edgelist(edges)
    return graph

def graph_level_reader(path):
    """
    Reading a single graph from disk.
    :param path: Path to the JSON file.
    :return data: Dictionary of data.
    """
    data = json.load(open(path))
    return data

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

class GraphDatasetGenerator(object):
    """
    Creating an in memory version of the graphs.
    :param path: Folder with json files.
    """
    def __init__(self, path):
        self.path = path
        self._enumerate_graphs()
        self._count_features_and_labels()
        self._create_target()
        self._create_dataset()

    def _enumerate_graphs(self):
        """
        Listing the graph files and creating the respective label and feature maps.
        """
        graph_count = int(len(glob.glob(self.path + "*.json")))  # qed use 0.7 dataset for memory bottleneck
        labels = set()
        features = set()
        self.graphs = []
        for index in  tqdm(range(graph_count)):
            graph_file = self._concatenate_name(index)
            data = graph_level_reader(graph_file)
            self.graphs.append(data)
            labels = labels.union(set([data["label"]]))
            features = features.union(set([val for k, v in data["features"].items() for val in v]))
        self.label_map = {v: i for i, v in enumerate(labels)}
        self.feature_map = {v: i for i, v in enumerate(features)}

    def _count_features_and_labels(self):
        """
        Counting the number of unique features and labels.
        """
        self.number_of_features = len(self.feature_map)
        self.number_of_labels = len(self.label_map)

    def _transform_edges(self, raw_data):
        """
        Transforming an edge list from the data dictionary to a tensor.
        :param raw_data: Dictionary with edge list.
        :return : Edge list matrix.
        """
        edges = [[edge[0], edge[1]] for edge in raw_data["edges"]]
        edges = edges + [[edge[1], edge[0]] for edge in raw_data["edges"]]
        if torch.cuda.is_available():
            return torch.t(torch.LongTensor(edges)).cuda()
        else:
            return torch.t(torch.LongTensor(edges))

    def _concatenate_name(self, index):
        """
        Creating a file name from an index.
        :param index: Graph index.
        :return : File name.
        """
        return self.path + str(index) + ".json"

    def _transform_features(self, raw_data):
        """
        Creating a feature matrix from the raw data.
        :param raw_data: Dictionary with features.
        :return feature_matrix: FloatTensor of features.
        """
        number_of_nodes = len(raw_data["features"])
        feature_matrix = np.zeros((number_of_nodes, self.number_of_features))
        index_1 = [int(n) for n, feats in raw_data["features"].items() for f in feats]
        index_2 = [int(self.feature_map[f]) for n, feats in raw_data["features"].items() for f in feats]
        feature_matrix[index_1, index_2] = 1.0
        if torch.cuda.is_available():

            feature_matrix = torch.FloatTensor(feature_matrix).cuda()
        else:
            feature_matrix = torch.FloatTensor(feature_matrix)
        return feature_matrix

    def _data_transform(self, raw_data):
        """
        Creating a dictionary with the edge list matrix and the features matrix.
        """
        clean_data = dict()
        clean_data["edges"] = self._transform_edges(raw_data)
        clean_data["features"] = self._transform_features(raw_data)
        clean_data["smiles"] = raw_data["smiles"]
        clean_data["label"] = raw_data["label"]
        return clean_data

    def _create_target(self):
        """
        Creating a target vector.
        """
        self.target = [graph["label"] for graph in self.graphs]
        if torch.cuda.is_available():

            self.target = torch.LongTensor(self.target).cuda()
        else:
            self.target = torch.LongTensor(self.target)

    def _create_dataset(self):
        """
        Creating a list of dictionaries with edge list matrices and feature matrices.
        """
        self.graphs = [self._data_transform(graph) for graph in self.graphs]





if __name__ == '__main__':
    import torch
    from torch_geometric.nn import GCNConv, GATConv
    import torch.nn.functional as F
    new_adj = torch.Tensor([[1,2],[3,4]])
    print(torch.norm(new_adj, p = 1))

