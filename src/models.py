import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import igraph
import pdb
import copy
from torch_geometric.data import Data

from batch import Batch

'''
    Circuit-SAT.
    The model used in this paper named Deep-Gated DAG Recursive Neural Networks (DG-DARGNN).
'''
class DGDAGRNN(nn.Module):
    '''
    The implemnetation of DGDAGRNN with Pytorch Geometric.
    Attributes:
        max_n (integer) - The maximum number of vertices of graph dataset
        nvt (integer, default: 3) - # vertex types.
        vhs (integer, default: 100) - the size of hidden state of nodes.
        chs (integer, default: 30) - the size of hidden state of classifier.
        temperature (float, default: 5.0) - the initial value of temperature for soft MIN.
        kstep (float, default: 10.0) - the value of k in soft step function.
        num_rounds (integer, default: 10) - # GRU iterations. 
    '''
    def __init__(self, max_n, nvt=3, vhs=100, chs=30, temperature=5.0, kstep=10.0, nrounds=10):
        super(DGDAGRNN, self).__init__()
        self.max_n = max_n # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.vhs = vhs  # hidden state size of each vertex
        self.chs = chs  # hidden state size of classifier
        self.nrounds = nrounds
        self.temperature = temperature
        self.kstep = kstep
        self.num_layers = 2 # one forward and one backword

        self.device = None

        # 0. GRU-related
        self.grue_forward = nn.GRUCell(self.nvt, self.vhs)  # encoder GRU
        self.grue_backward = nn.GRUCell(self.vhs, self.vhs)  # backward encoder GRU
        self.projector = nn.Linear(self.vhs, self.nvt)

        # 1. classifer-related
        self.literal_classifier = nn.Sequential(
                nn.Linear(self.hs, self.chs),
                nn.ReLU(),
                nn.Linear(self.chs, 1),
                nn.Sigmoid()
            )
        

        # 2. gate-related, aggregate
        num_rels = 1    # num_relationship
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vhs, self.vhs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.vhs, self.vhs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vhs, self.vhs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.vhs, self.vhs, bias=False), 
                )
            
        self.node_aggr_forward = GatedSumConv(self.vhs, num_rels, mapper=self.mapper_forward, gate=self.gate_forward)
        self.node_aggr_backward = GatedSumConv(self.vhs, num_rels, mapper=self.mapper_backward, gate=self.gate_backward, reverse=True)

        # 3. evaluator
        self.soft_evaluator = SoftEvaluator()
        self.hard_evaluator = HardEvaluator()

        # 4. loss function
        self.sat_loss = smooth_step()



    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def smooth_step(self, inputs, k):
        output = torch.pow(1-inputs, k) / (torch.pow(1-inputs, k) + torch.pow(inputs, k))
        output = torch.mean(output)
        return output
    
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _collate_fn(self, G):
        return [copy.deepcopy(g) for g in G]

    def forward(self, G):
        # GNN computation to get node embeddings
        num_nodes_batch = G.x.shape[0]
        num_layers_batch = max(G.bi_layer[0][0]).item() + 1

        G.h = torch.zeros(num_nodes_batch, self.vhs)
        
        # forward
        for round_idx in range(self.nrounds):
            for l_idx in range(num_layers_batch):
                layer = G.bi_layer_index[0][0] == l_idx
                layer = G.bi_layer_index[0][1][layer]   # the vertices ID for this batch layer
                
                if round_idx == 0:
                    inp = G.x[layer]    # input node feature vector
                else:
                    inp = self.projector(G.h)
                
                if l_idx > 0:   # no predecessors at first layer
                    le_idx = []
                    for n in layer:
                        ne_idx = G.edge_index[1] == n
                        le_idx += [ne_idx.nonzero().squeeze(-1)]    # the index of edge edge in edg_index
                    le_idx = torch.cat(le_idx, dim=-1)
                    lp_edge_index = G.edge_index[:, le_idx] # the subset of edge_idx which contains the target vertices ID
                
                if l_idx == 0:
                    ps_h = None
                else:
                    hs1 = G.h
                    node_agg = self.node_aggr_forward()
                    ps_h = node_agg(hs1, lp_edge_index, edge_attr=None, **kwargs)[layer]
                
                inp = cell(inp, ps_h)
                G.h[layer] = inp
            
            # backword
            for l_idx in range(num_layers_batch):
                layer = G.bi_layer_index[1][0] == l_idx
                layer = G.bi_layer_index[1][1][layer]   # the vertices ID for this batch layer

                inp = G.x[layer]    # input node feature vector
                
                if l_idx > 0:   # no predecessors at first layer
                    le_idx = []
                    for n in layer:
                        ne_idx = G.edge_index[1] == n
                        le_idx += [ne_idx.nonzero().squeeze(-1)]    # the index of edge edge in edg_index
                    le_idx = torch.cat(le_idx, dim=-1)
                    lp_edge_index = G.edge_index[:, le_idx] # the subset of edge_idx which contains the target vertices ID
                
                if l_idx == 0:
                    ps_h = None
                else:
                    hs1 = G.h[0]
                    node_agg = self.node_aggr_backward(hs1, lp_edge_index, edge_attr=None)
                    ps_h = node_agg(hs1, lp_edge_index, edge_attr=None, **kwargs)[layer]
                
                inp = cell(inp, ps_h)
                G.h[layer] = inp
        return G.h


    def solver(self, G):
        if type(G) != list:
            G = [G]
        # encode graphs G into latent vectors
        b = Batch.from_data_list(G)
        b.to(self.get_device())
        Hn = self(b)
        # get the soft-assigment (spase, only has values for variable nodes)
        num_nodes_batch = G.x.shape[0]
        b.softassign = torch.zeros(num_nodes_batch, 1)
        first_layer = G.bi_layer_index[0][0] == 0
        first_layer = G.bi_layer_index[0][1][first_layer]   # the vertices ID for this batch layer
        HLiteral = Hn[first_layer]
        softassign = self.literal_classifier(HLiteral)
        b.softassign[first_layer] = softassign
        return b
    
    def evaluator(self, G):
        num_nodes_batch = G.x.shape[0]
        num_layers_batch = max(G.bi_layer[0][0]).item() + 1

        for l_idx in range(1, num_layers_batch):
            layer = G.bi_layer_index[0][0] == l_idx
            layer = G.bi_layer_index[0][1][layer]   # the vertics ID for this batch layer
            
            inp = G.softassign[layer]    # input soft assignment

            le_idx = []
            for n in layer:
                ne_idx = G.edge_index[1] == n
                le_idx += [ne_idx.nonzero().squeeze(-1)]    # theindex of edge edge in edg_index
            le_idx = torch.cat(le_idx, dim=-1)
            lp_edge_index = G.edge_index[:, le_idx] # the subsetof edge_idx which contains the target vertices ID
        
            assignment = G.softassign
            update_assigment = self.soft_evaluator(assignment,lp_edge_index, node_attr=G.x)[layer]
            G.softassign[layer] = update_assigment

        last_layer = G.bi_layer_index[1][0] == 0
        last_layer = G.bi_layer_index[0][1][last_layer]
        satisﬁability = G.softassign[last_layer]
        return satisﬁability

    def solve_and_evaluate(self, G):
        G = self.solver(G)
        satisﬁability = self.evaluator(G)
        return satisﬁability
    

    
        

class GatedSumConv(MessagePassing):  # dvae needs outdim parameter
    '''
    Some parameter definitions:
        num_relations (integer): the edge types. If 1, then no information from edge attribute.
                        if not zero, then it reprensent the number of edge types.
        wea (bool): with edge attributes. If num_relations > 1, then the graph is with edge attributes.
        edge_encoder: cast the one-hot edge feature vector into emb_dim size.
    It is not the exactly Deep-Set. Should implement DeepSet later.
    Consider change `aggr` from 'add' to 'mean'. It makes sense when there are AND gates and NOT gates.
    '''
    def __init__(self, emb_dim, num_relations=1, reverse=False, mapper=None, gate=None):
        super(GatedSumConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)
        else:
            self.wea = False
        self.mapper = nn.Linear(emb_dim, emb_dim) if mapper is None else mapper
        self.gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid()) if gate is None else gate

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # HACK assume x contains only message sources
        if edge_index is None:
            h = self.gate(x) * self.mapper(x)
            return torch.sum(h, dim=1)

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        h_j = x_j + edge_attr if self.wea else x_j
        return self.gate(h_j) * self.mapper(h_j)

    def update(self, aggr_out):
        return aggr_out

class SoftEvaluator(MessagePassing):
    '''
    AND node => Soft Min;
    Not node => 1 - z;
    '''
    def __init__(self, temperature=5.0):
        super(GatedSumConv, self).__init__(aggr='add', flow='source_to_target')

        self.temperature = 5.0
        self.softmin = nn.Softmin(dim=0)


    def forward(self, x, edge_index, node_attr=None):

        return self.propagate(edge_index, x=x, node_attr=node_attr)

    def message(self, x_j, node_attr_i):
        # x_j has shape [E, out_channels], where out_channel is jut one-dimentional value in range of (0, 1)
        if node_attr_i[1] == 1.0: 
            tmp = self.softmin(x_j / self.temperature)
            return x_j * temp
        elif node_attr_i[2] == 1.0:
            return 1 - x_j
        else:
            raise ValueError

    def update(self, aggr_out):
        return aggr_out

class HardEvaluator(MessagePassing):
    '''
    AND node => Hard Min;
    Not node => 1 - z;
    '''
    def __init__(self, temperature=5.0):
        super(GatedSumConv, self).__init__(aggr='add', flow='source_to_target')

        self.softmin = nn.Softmin(dim=0)


    def forward(self, x, edge_index, node_attr=None):

        return self.propagate(edge_index, x=x, node_attr=node_attr)

    def message(self, x_j, node_attr_i):
        # x_j has shape [E, out_channels], where out_channel is jut one-dimentional value in range of (0, 1)
        if node_attr_i[0][1] == 1.0: 
            return torch.min(x_j, keepdim=True)
        elif node_attr_i[0][2] == 1.0:
            return 1 - x_j
        else:
            raise ValueError

    def update(self, aggr_out):
        return aggr_out

