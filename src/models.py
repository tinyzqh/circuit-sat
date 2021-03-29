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
    The model used in this paper named Deep-Gated DAG Recursive Neural Networks (DG-DARGNN)
'''
class DGDAGRNN(nn.Module):
    '''
    The implemnetation of DVAEncoder with Pytorch Geometric.
    Attributes:
        max_n (integer) - The maximum number of vertices of graph dataset
        nvt (integer, default: 3) - # vertex types.
        hs (integer, default: 100) - the size of hidden state of nodes.
        num_rounds (integer, default: 10) - # GRU iterations. 
    '''
    def __init__(self, max_n, nvt=3, vhs=100, chs=30, temperature=5, kstep=10, nrounds=10):
        super(DGDAGRNN, self).__init__()
        self.max_n = max_n # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.vhs = vhs  # hidden state size of each vertex
        self.chs = chs  # hidden state size of classifier
        self.nrounds = nrounds
        self.temperature = temperature
        self.kstep = kstep
        self.dirs = [0, 1]
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
        

        # 2. gate-related
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

        # 4. Loss functions
        self.sat_loss = self.smooth_step()

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
                if len(H_pred[-1]) != 0 and H_name in g.vs[v]:
                    H_pred[-1] += [g.vs[v][H_name]]
                    E_pred[-1] = torch.cat((E_pred[-1], self._get_zeros(1, self.net)), dim=0)
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
                if len(H_pred[-1]) != 0 and H_name in g.vs[v]:
                    H_pred[-1] += [g.vs[v][H_name]]
                    E_pred[-1] = torch.cat((E_pred[-1], self._get_zeros(1, self.net)), dim=0)
            gate, mapper = self.gate_forward, self.mapper_forward
        
        H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, E_pred)]


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
            hg = g.vs[g.x.shape[0] - 1]['H_forward']
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

    def solve(self, predictions_digit):
        pass



    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        H_vf = self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            H_vb = self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        for _ in range(self.nrounds - 1):
            H_vf = self._propagate_from(G, 0, self.grue_forward, H0=H_vf,
                             reverse=False)
            if self.bidir:
                H_vb = self._propagate_from(G, self.max_n-1, self.grue_backward, H0=H_vb,
                                 reverse=True)

        Hg = self._get_graph_state(G, self.bidir)
        return Hg
        


    def graph_loss(self, binary_logit, y):        
        return self.sat_loss(binary_logit, y)
    
    def solution_loss(self, predicted, solutions):
        return self.literal_loss(predicted, solutions)


    # def forward(self, G):
    #     Hg = self.encode(G)
    #     binary_logit = self.graph_classifier(Hg)
    #     HLiteral, solutions = self._get_literal_state(G)
    #     predicted_solutions  = self.literal_classifier(HLiteral)

    #     return binary_logit, predicted_solutions, solutions

    def forward(self, G):
        device = self.get_device()
        G = G.to(device)
        
        num_nodes_batch = G.x.shape[0]
        num_layrers_batch = max(G.bi_layer[0][0]).item() + 1

        G.h = [torch.zeros(num_nodes_batch, self.vhs)]

        for l_idx in range(num_layers_batch):
            layer = G.bi_layer_index[d][0] == l_idx
            layer = G.bi_layer_index[d][1][layer]

            inp = G.x[layer]
            
            


