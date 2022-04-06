"""Convolutional layers."""
import torch
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
import torch.nn.functional as F
import torch.nn as nn
from rdkit import Chem
from torch_geometric.utils import to_dense_adj


class SAGE(torch.nn.Module):
    """
    SAGE layer class.
    """
    def __init__(self, args, number_of_features):
        """
        Creating a SAGE layer.
        :param args: Arguments object.
        :param number_of_features: Number of node features.
        """
        super(SAGE, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self._setup()
        self.mseloss = torch.nn.MSELoss()
        self.relu = torch.nn.ReLU()
        self.subgraph_const = self.args.subgraph_const

    def _setup(self):
        """
        Setting up upstream and pooling layers.
        """
        if self.args.gnn == 'GCN':
            self.graph_convolution_1 = GCNConv(self.number_of_features, self.args.first_gcn_dimensions)
            self.graph_convolution_2 = GCNConv(self.args.first_gcn_dimensions, self.args.second_gcn_dimensions)

        elif self.args.gnn == 'GIN':
            self.graph_convolution_1 = GINConv(
                nn.Sequential(
                    nn.Linear(self.number_of_features, self.args.first_gcn_dimensions),
                    nn.ReLU(),
                    nn.Linear(self.args.first_gcn_dimensions, self.args.first_gcn_dimensions),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.args.first_gcn_dimensions),
                ), train_eps=False)

            self.graph_convolution_2 = GINConv(
                nn.Sequential(
                    nn.Linear(self.args.first_gcn_dimensions, self.args.second_gcn_dimensions),
                    nn.ReLU(),
                    nn.Linear(self.args.second_gcn_dimensions, self.args.second_gcn_dimensions),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.args.second_gcn_dimensions),
                ), train_eps=False)

        elif self.args.gnn == 'GAT':
            self.graph_convolution_1 = GATConv(self.number_of_features, self.args.first_gcn_dimensions, heads=2)
            self.graph_convolution_2 = GATConv(2 * self.args.first_gcn_dimensions, self.args.second_gcn_dimensions)

        elif self.args.gnn == 'SAGE':
            self.graph_convolution_1 = SAGEConv(self.number_of_features, self.args.first_gcn_dimensions)
            self.graph_convolution_2 = SAGEConv(self.args.first_gcn_dimensions, self.args.second_gcn_dimensions)

        self.fully_connected_1 = torch.nn.Linear(self.args.second_gcn_dimensions,
                                                 self.args.first_dense_neurons)

        self.fully_connected_2 = torch.nn.Linear(self.args.first_dense_neurons,
                                                 self.args.second_dense_neurons)

    def forward(self, data):
        """
        Making a forward pass with the graph level data.
        :param data: Data feed dictionary.
        :return graph_embedding: Graph level embedding.
        :return penalty: Regularization loss.
        """
        edges = data["edges"]
        epsilon = 0.0000001

        features = data["features"]
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)
        num_nodes = node_features_2.size()[0]

        #this part is used to add noise
        node_feature = node_features_2
        static_node_feature = node_feature.clone().detach()
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)

        #this part is used to generate assignment matrix
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature))
        assignment = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)

        gumbel_assignment = self.gumbel_softmax(assignment)

        #This is the graph embedding
        graph_feature = torch.sum(node_feature, dim = 0, keepdim=True)

        #add noise to the node representation
        node_feature_mean = node_feature_mean.repeat(num_nodes,1)

        #noisy_graph_representation
        lambda_pos = gumbel_assignment[:,0].unsqueeze(dim = 1)
        lambda_neg = gumbel_assignment[:,1].unsqueeze(dim = 1)

        #print(assignment[:0],lambda_pos)

        #this is subgraph embedding
        subgraph_representation = torch.sum(lambda_pos * node_feature, dim = 0, keepdim=True)

        noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std

        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        noisy_graph_feature = torch.sum(noisy_node_feature, dim = 0, keepdim=True)

        KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std+epsilon) ** 2) + \
                    torch.sum(((noisy_node_feature_mean -node_feature_mean)/(node_feature_std + epsilon))**2, dim = 0) #+\
        #            torch.log(node_feature_std / (noisy_node_feature_std + epsilon) + epsilon)
        KL_Loss = torch.mean(KL_tensor)

        if torch.cuda.is_available():
            EYE = torch.ones(2).cuda()
            Pos_mask = torch.FloatTensor([1,0]).cuda()
        else:
            EYE = torch.ones(2)
            Pos_mask = torch.FloatTensor([1, 0])

        Adj = to_dense_adj(edges)[0]
        Adj.requires_grad = False
        new_adj = torch.mm(torch.t(assignment),Adj)
        new_adj = torch.mm(new_adj,assignment)

        normalize_new_adj = F.normalize(new_adj, p=1, dim=1)

        norm_diag = torch.diag(normalize_new_adj)
        pos_penalty = self.mseloss(norm_diag, EYE)
        #cal preserve rate
        preserve_rate = torch.sum(assignment[:,0] > 0.5) / assignment.size()[0]

        return graph_feature, noisy_graph_feature, subgraph_representation, pos_penalty, KL_Loss, preserve_rate


    def return_att(self,data):
        edges = data["edges"]
        features = data["features"]
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)

        abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))
        attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)

        return attention


    def gumbel_softmax(self, prob):

        return F.gumbel_softmax(prob, tau = 1, dim = -1)




class Subgraph(torch.nn.Module):

    def __init__(self, args, number_of_features):
        super(Subgraph, self).__init__()

        self.args = args
        self.number_of_features = number_of_features
        self._setup()
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
        self.bce_criterion = torch.nn.BCELoss(reduction='mean')
        self.relu = torch.nn.ReLU()



    def _setup(self):

        self.graph_level_model = SAGE(self.args, self.number_of_features)
        if self.args.unsupervised:
            self.classify = torch.nn.Sequential(torch.nn.Linear(self.args.second_gcn_dimensions, self.args.cls_hidden_dimensions), torch.nn.ReLU(),torch.nn.Linear(self.args.cls_hidden_dimensions, 1), torch.nn.Sigmoid())
        else:
            self.classify = torch.nn.Sequential(torch.nn.Linear(self.args.second_gcn_dimensions, self.args.cls_hidden_dimensions), torch.nn.ReLU(),torch.nn.Linear(self.args.cls_hidden_dimensions, 1), torch.nn.ReLU())

    def forward(self, graphs):

        embeddings = []
        positive = []
        negative = []
        subgraph = []
        noisy_graph = []

        labels = []

        positive_penalty = 0
        preserve_rate = 0
        KL_Loss = 0
        for graph in graphs:
            embedding, noisy, subgraph_emb, pos_penalty, kl_loss, one_preserve_rate = self.graph_level_model(graph)

            embeddings.append(embedding)
            positive.append(noisy)
            subgraph.append(subgraph_emb)
            noisy_graph.append(noisy)
            positive_penalty += pos_penalty
            KL_Loss += kl_loss
            preserve_rate += one_preserve_rate
            labels.append(graph["label"])

        embeddings = torch.cat(tuple(embeddings),dim = 0)
        positive = torch.cat(tuple(positive),dim = 0)
        subgraph = torch.cat(tuple(subgraph),dim = 0)
        noisy_graph = torch.cat(tuple(noisy_graph),dim=0)

        labels = torch.FloatTensor(labels).view(-1,1)

        positive_penalty = positive_penalty/len(graphs)
        KL_Loss = KL_Loss / len(graphs)
        preserve_rate = preserve_rate / len(graphs)

        if self.args.unsupervised:
            cls_loss = self.unsupervise_classify_loss(embeddings, positive, negative)
        else:
            cls_loss = self.supervise_classify_loss(embeddings, positive, subgraph, labels)

        return embeddings, positive, noisy_graph, KL_Loss, cls_loss, self.args.con_weight * positive_penalty, preserve_rate



    def supervise_classify_loss(self,embeddings,positive,subgraph,labels):
        data = torch.cat((embeddings, positive), dim=0)

        labels = torch.cat((labels,labels),dim = 0)
        if torch.cuda.is_available():
            labels = labels.cuda()
        pred = self.classify(data)
        loss = self.mse_criterion(pred,labels)

        return loss



    def assemble(self, graphs):

        all_index_pair = []

        for graph in graphs:

            smiles = graph['smiles']

            attention = self.graph_level_model.return_att(graph)


            _,ind = torch.max(attention,1)
            ind = ind.tolist()
            pos_ind = [i for i,j in enumerate(ind) if j == 0]

            decomposed_cluster, decomposed_subgraphs = self.decompose_cluster(smiles,pos_ind)

            smiles_ind_pairs = {'smiles':smiles,'ind':decomposed_cluster,'subgraphs':decomposed_subgraphs}

            all_index_pair.append(smiles_ind_pairs)

        return all_index_pair


    def gumbel_softmax(self, prob):

        return F.gumbel_softmax(prob, dim = -1)


    def get_nei(self, atom_ind, edges):
        nei = []
        for bond in edges:
            if atom_ind in bond:
                nei.extend(bond)
        nei.remove(atom_ind)
        nei = list(set(nei))

        return nei

    def decompose_cluster(self,smiles, ind):

        mol = Chem.MolFromSmiles(smiles)

        all_cluster = []
        all_subgraphs = []

        edge = []
        for bond in mol.GetBonds():
            edge.append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])

        for i in ind:
            cluster = [i]
            ind.remove(i)
            nei = self.get_nei(i,edge)
            valid_nei = list(set(nei).intersection(set(ind)))
            while valid_nei != [] :

                cluster.extend(valid_nei)

                new_nei = []
                for j in valid_nei:
                    ind.remove(j)
                    new_nei.extend(self.get_nei(j,edge))

                valid_nei = list(set(new_nei).intersection(set(ind)))

            subgraph = self.get_subgraph_with_idx(smiles,cluster)
            all_subgraphs.append(subgraph)

            all_cluster.append(cluster)

        return all_cluster, all_subgraphs



    def get_subgraph_with_idx(self,smiles,ind):
        mol = Chem.MolFromSmiles(smiles)

        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            ring = list(ring)

            if set(ind) >= set(ring):
                pass
            else:
                for atom_ind in ring:
                    broke_atom = mol.GetAtomWithIdx(atom_ind)
                    broke_atom.SetIsAromatic(False)

        edit_mol = Chem.EditableMol(mol)
        del_ind = sorted(list(set(range(mol.GetNumAtoms())) - set(ind)))[::-1]
        for idx in del_ind:
            edit_mol.RemoveAtom(idx)
        new_mol = edit_mol.GetMol()

        subgraph = Chem.MolToSmiles(new_mol)

        if subgraph:
            return subgraph
        else:
            return None



if __name__ == '__main__':
    from rdkit import Chem
    import torch

    from torch_geometric.utils import to_dense_adj

    edge = [[0,1,1,1,2,3,0,4],[1,0,2,3,1,1,4,0]]
    edge = torch.LongTensor(edge)
    batch_id = torch.LongTensor([0,0,1,1,1])
    all_edge = to_dense_adj(edge)[0]
    print(all_edge)
    st = 0
    end = 2
    edge = all_edge[st:end,st:end]
    print(edge)



















