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

'''
    DAG Variational Autoencoder (D-VAE).
    https://github.com/muhanzhang/D-VAE/blob/master/models.py
    I adopt the encoder part, and add two-layer MLP as the classifier.
    For now, the edge information is encoded using one-hot vectors.
    This version is customized for Pytorch Geometric Data.
'''
class DVAEncoder_PYG(nn.Module):
    '''
    The implemnetation of DVAEncoder with Pytorch Geometric.
    Attributes:
        max_n (integer) - The maximum number of vertices of graph dataset
        nvt (integer, default: 4) - # vertex types.
        net (integer, default: 3) - # node types.
        hs (integer, default: 100) - the size of hidden state of nodes.
        gs (integer, equals to hs) - the size of hidden state of graph.
        bidirectional (bool, default: True)
        (Removed for now) num_rounds (integer, default: 26) - # GRU iterations. TODO: check whether this part is nessary or not.
        (Removed for now) vid (bool, default: True) - The way to integrate the edge information. Cat the hidden state of predecessor
                                and the corresponding edge vector right now. TODO: check whether we need this flag.
        (Removed for now) num_layers (integer, default: 1) - # layers of GRU
    '''
    def __init__(self, max_n, nvt=4, net=3, hs=100, bidirectional=True):
        super(DVAEncoder_PYG, self).__init__()
        self.max_n = max_n # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.net = net  # number of edge types
        self.hs = hs  # hidden state size of each vertex
        self.gs = hs  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.device = None

        self.vs = hs + net


        # 0. encoding-related
        self.grue_forward = nn.GRUCell(nvt, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(nvt, hs)  # backward encoder GRU

        # 1. classifer-related
        self.graph_classifier = nn.Sequential(
                nn.Linear(self.gs, self.gs * 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.gs * 2),
                nn.Linear(self.gs * 2, 1)
            )
        self.literal_classifier = nn.Sequential(
                nn.Linear(self.hs, self.hs * 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hs * 2),
                nn.Linear(self.hs * 2, 1)
            )
        

        # 2. gate-related
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False), 
                )


        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.hs * 2, self.gs), 
                    nn.BatchNorm1d(self.gs)
                )

        # 4. LLoss functions
        self.sat_loss = nn.BCEWithLogitsLoss()
        self.literal_loss = nn.BCEWithLogitsLoss()

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
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

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [copy.deepcopy(g) for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g.to(self.get_device()) for g in G if g.x.shape[0] > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.x.shape[0] > v]
            H = H[idx]
        X = torch.stack([g.x[v] for g in G], dim=0)
        
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = []
            E_pred = []
            for g in G:
                # np: node parents
                np_idx = g.edge_index[0] == v
                np_idx = g.edge_index[1][np_idx]
                H_pred += [[g.vs[x][H_name] for x in np_idx]]
                E_pred += [g.edge_attr[np_idx]]
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'
            H_pred = []
            E_pred = []
            for g in G:
                # np: node parents
                np_idx = g.edge_index[1] == v
                np_idx = g.edge_index[0][np_idx]
                H_pred += [[g.vs[x][H_name] for x in np_idx]]
                E_pred += [g.edge_attr[np_idx]]
            gate, mapper = self.gate_forward, self.mapper_forward

        H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, E_pred)]
        # print(H_pred[0].size())
        # exit()

        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) 
                            for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        # prop_order = G.top_order.tolist()
        # if reserse:
        #      prop_order.reverse()
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator, reverse=reverse)  # ML: For the with predecessors, do not consider the previous states of this nodes.
        return Hv

    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.x.shape[0]:
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, bidir=False):
        # get the graph states
        Hg = []
        for g in G:
            print(g.vs[g.x.shape[0] - 1])
            hg = g.vs[g.x.shape[0] - 1]['H_forward']
            exit()
            if bidir:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if bidir:
            Hg = self.hg_unify(Hg)
        return Hg

    
    def _get_literal_state(self, G):
        # get the literal state
        HLiteral = []
        solutions = []
        for g in G:
            if g.solution is not None:
                for idx_literal in range(g.num_literals):
                    HLiteral.append(g.vs[idx_literal+1]['H_backward'])
                    solutions.append(g.solution[idx_literal])
        HLiteral = torch.cat(HLiteral, 0)
        solutions = torch.FloatTensor(solutions).unsqueeze(1).to(self.get_device())

        return HLiteral, solutions



    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            H_vb = self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_graph_state(G), reverse=True)

        Hg = self._get_graph_state(G, self.bidir)
        return Hg
        


    def graph_loss(self, binary_logit, y):        
        return self.sat_loss(binary_logit, y)
    
    def solution_loss(self, predicted, solutions):
        return self.literal_loss(predicted, solutions)


    def forward(self, G):
        Hg = self.encode(G)
        binary_logit = self.graph_classifier(Hg)
        HLiteral, solutions = self._get_literal_state(G)
        predicted_solutions  = self.literal_classifier(HLiteral)

        return binary_logit, predicted_solutions, solutions
