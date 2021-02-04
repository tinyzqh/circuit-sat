import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import igraph
import pdb

'''
    DAG Variational Autoencoder (D-VAE).
    https://github.com/muhanzhang/D-VAE/blob/master/models.py
    I adopt the encoder part, and add two-layer MLP as the classifier.
    For now, the edge information is encoded using one-hot vectors.
'''
class DVAEncoder(nn.Module):
    def __init__(self, max_n, nvt, net, hs=100, gs=100, n_rounds=26, bidirectional=True, vid=True):
        super(DVAEncoder, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.net = net  # number of edge types
        self.hs = hs  # hidden state size of each vertex
        self.gs = gs  # size of graph state
        self.n_rounds = n_rounds
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.vid = vid  # ML: Change to flag of including one-hot edge vector
        self.device = None

        # Use one-hot vector to encode edge types
        if self.vid:
            self.vs = hs + net
        else:
            self.vs = hs


        # 0. encoding-related
        self.grue_forward = nn.GRUCell(nvt, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(nvt, hs)  # backward encoder GRU

        # 1. classifer. ML
        self.classifier = nn.Sequential(
                nn.Linear(self.gs, self.gs * 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.gs * 2),
                nn.Linear(self.gs * 2, 1)
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
                    nn.Linear(self.gs * 2, self.gs), 
                    nn.BatchNorm1d(self.gs)
                    )

        # 4. other
        self.sat_loss = nn.BCEWithLogitsLoss()

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
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['v_type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.es[[g.get_eid(v, i) for i in g.successors(v)]]['e_type'], self.net) for g in G]
            else:
                H_pred = [[-g.vs[x][H_name] if g.get_eid(v, x) else g.vs[x][H_name] for x in g.successors(v)] for g in G]
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.es[[g.get_eid(i, v) for i in g.predecessors(v)]]['e_type'], self.net) for g in G]
            else:
                H_pred = [[-g.vs[x][H_name] if g.get_eid(x, v) else g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            gate, mapper = self.gate_forward, self.mapper_forward
        if self.vid:
            H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + 
                            [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) 
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
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator, reverse=reverse)
        return Hv

    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        H_vf = self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            H_vb = self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        for _ in range(self.n_rounds):
            H_vf = self._propagate_from(G, 0, self.grue_forward, H0=H_vf,
                             reverse=False)
            if self.bidir:
                H_vb = self._propagate_from(G, self.max_n-1, self.grue_backward,  H0=H_vb,
                                 reverse=True)

        Hg = self._get_graph_state(G)
        return Hg


    def loss(self, binary_logit, y):        
        return self.sat_loss(binary_logit, y)
        


    def forward(self, G):
        Hg = self.encode(G)
        binary_logit = self.classifier(Hg)
        return binary_logit


class DVAEdgeEncoder(nn.Module):
    def __init__(self, max_n, nvt, net, hs=100, gs=100, n_rounds=26, bidirectional=True, vid=True):
        super(DVAEdgeEncoder, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.net = net  # number of edge types
        self.hs = hs  # hidden state size of each vertex
        self.gs = gs  # size of graph state
        self.n_rounds = n_rounds
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.vid = vid  # ML: Change to flag of including one-hot edge vector
        self.device = None

        if self.vid:
            self.vs = hs + net
        else:
            self.vs = hs


        # 0. encoding-related
        self.grue_forward_v = nn.GRUCell(nvt, hs)  # encoder node GRU
        self.grue_backward_v = nn.GRUCell(nvt, hs)  # backward encoder node GRU

        self.grue_forward_e = nn.GRUCell(net, hs)  # encoder edge GRU
        self.grue_backward_e = nn.GRUCell(net, hs)  # backward encoder edge GRU

        # 1. classifer. ML
        self.classifier = nn.Sequential(
                nn.Linear(self.gs, self.gs * 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.gs * 2),
                nn.Linear(self.gs * 2, 1)
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
                    nn.Linear(self.gs * 2, self.gs), 
                    nn.BatchNorm1d(self.gs)
                    )

        # 4. other
        self.sat_loss = nn.BCEWithLogitsLoss()

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
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator_v, propagator_e, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['v_type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred_v = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
            inputs_e = [self._one_hot(g.es[[g.get_eid(v, i) for i in g.successors(v)]]['e_type'], self.net) for g in G]
            H_pred_v = torch.cat(H_pred_v, 0)
            inputs_e = torch.cat(inputs_e, 0)
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred_v = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            inputs_e = [self._one_hot(g.es[[g.get_eid(i, v) for i in g.predecessors(v)]]['e_type'], self.net) for g in G]
            H_pred_v = torch.cat(H_pred_v, 0)
            inputs_e = torch.cat(inputs_e, 0)
            gate, mapper = self.gate_forward, self.mapper_forward
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            He = propagator_e(inputs_e, H_pred_v)
            max_n_pred = max([len(x) for x in He])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + 
                            [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) 
                            for h_pred in He]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator_v(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        n_e = 0
        for i, g in enumerate(G):
            for x in g.successors(v):
                g.es[g.get_eid(v, x)][H_name] = He[n_e]
                n_e += 1
        return Hv

    def _propagate_from(self, G, v, propagator_v, propagator_e, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator_v, propagator_e, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator_v, propagator_e, reverse=reverse)
        return Hv

    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        H_vf = self._propagate_from(G, 0, self.grue_forward_v, self.grue_forward_e, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            H_vb = self._propagate_from(G, self.max_n-1, self.grue_backward_v, self.grue_backward_e, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        for _ in range(self.n_rounds):
            H_vf = self._propagate_from(G, 0, self.grue_forward_v, self.grue_forward_e, H0=H_vf,
                             reverse=False)
            if self.bidir:
                H_vb = self._propagate_from(G, self.max_n-1, self.grue_backward_v, self.grue_backward_e, H0=H_vb,
                                 reverse=True)

        Hg = self._get_graph_state(G)
        return Hg


    def loss(self, binary_logit, y):        
        return self.sat_loss(binary_logit, y)
        


    def forward(self, G):
        Hg = self.encode(G)
        binary_logit = self.classifier(Hg)
        return binary_logit
