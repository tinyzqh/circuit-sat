'''
Parse AIG graph (*.aag) into igraph format (https://github.com/igraph/python-igraph)
or Parse AIG graph (*.aag) into PytorchGeometric Data format(https://github.com/rusty1s/pytorch_geometric)
Circuit-sat: no virtual input node; Represent NOT gate as Node/Virtex.
'''
from os import listdir
from os.path import join, splitext
import pickle
import numpy as np
import igraph
import argparse
import networkx as nx
import torch
from torch_geometric.data import Data
from utils_dag import add_order_info
from util import one_hot

# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()


def aig2graph(folder_name, solution_folder_name, n_vtypes=3, print_interval=100):
    '''
    load AIG graph (*.aag) to igraph.
    n_vtypes: # of vertex types; 3 in DGDAGRNN case;
    '''
    g_list = []
    max_n = 0 # maximum number of nodes

    for (i, filename) in enumerate([filename for filename in listdir(folder_name) if filename.endswith('.aag')]):
        if i % print_interval == 0: print("Processing # [%d] instances..." % i)
        aag_path = join(folder_name, filename)
        solution = None
        if solution_folder_name:
            solution_path = join(solution_folder_name, splitext(filename)[0][:-4] + '.solution')
            if "_sat=1" in solution_path:
                with open(solution_path, 'r') as f:
                    solution = [int(x_value) for x_value in f.read().strip().split(' ')]
        with open(aag_path, 'r') as f:
            lines = f.readlines()

        g, n = decode_aag_to_pyg(lines, solution, n_vtypes)
        if g == None:
            continue
        y = 1 if 'sat=1' in filename else 0

        max_n = max(max_n, n)
        g_list.append((g, y))

    graph_args.num_vertex_type = n_vtypes
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.literal_label = True if solution_folder_name else False
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    print('# graph: %d' % ng)
    return g_list, graph_args



def decode_aag_to_pyg(lines, solution, n_vtypes): 
    '''
    For AIG, the nodes can be categorized as the Literal node, internal AND nodes, internal NOT node. The type values for each kind of nodes are as follows:
        * Literal input node: 0;
        * Internal AND nodes: 1;
        * Internal NOT nodes: 2;
    '''
    header = lines[0].strip().split(" ")
    assert header[0] == 'aag', 'The header of AIG file is wrong.'
    # “M”, “I”, “L”, “O”, “A” separated by spaces.
    n_variables = eval(header[1])
    n_inputs = eval(header[2])
    n_outputs = eval(header[4])
    n_and = eval(header[5])
    assert n_outputs == 1, 'The AIG has multiple outputs.'
    assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    if n_variables == n_inputs:
        return None, None
    # Construct AIG graph
    # adj = np.zeors(n_variables+2, n_variables+2)
    x = []
    edge_index = []
    node_types2 = []

    # Add Literal node
    for i in range(n_inputs):
        x += [one_hot(0, n_vtypes)]

    # Add AND node
    for i in range(n_inputs+1, n_inputs+1+n_and):
        x += [one_hot(1, n_vtypes)]
        node_types2 += [1]


    # sanity-check
    for (i, line) in enumerate(lines[1:1+n_inputs]):
        literal = line.strip().split(" ")
        assert len(literal) == 1, 'The literal of input should be single.'
        assert int(literal[0]) == 2 * (i + 1), 'The value of a input literal should be the index of variables mutiplying by two.'

    literal = lines[1+n_inputs].strip().split(" ")[0]
    assert int(literal) == (n_variables * 2) or int(literal) == (n_variables * 2) + 1, 'The value of the output literal shoud be (n_variables * 2)'
    sign_final = int(literal) % 2
    index_final_and = int(literal) // 2 - 1

    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        literals = line.strip().split(" ")
        assert len(literals) == 3, 'invalidate the definition of two-input AND gate.'
        assert int(literals[0]) == 2 * (i + 1 + n_inputs)
    var_def = lines[2+n_variables].strip().split(" ")[0]

    assert var_def == 'i0', 'The definition of variables is wrong.'
    # finish sanity-check

    # Add edge
    for (i, line) in enumerate(lines[n_inputs+2: n_inputs+2+n_and]):
        line = line.strip().split(" ")
        # assert len(line) == 3, 'The length of AND lines should be 3.'
        output_idx = int(line[0]) // 2 - 1
        # assert (int(line[0]) % 2) == 0, 'There is inverter sign in output literal.'

        # 1. First edge
        input1_idx = int(line[1]) // 2 - 1
        sign1_idx = int(line[1]) % 2
        # If there's a NOT node
        if sign1_idx == 1:
            x += [one_hot(2, n_vtypes)]
            node_types2 += [2]
            not_idx = len(x) - 1
            edge_index += [[input1_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input1_idx, output_idx]]

        # 2. Second edge
        input2_idx = int(line[2]) // 2 - 1
        sign2_idx = int(line[2]) % 2
        # If there's a NOT node
        if sign2_idx == 1:
            x += [one_hot(2, n_vtypes)]
            node_types2 += [2]
            not_idx = len(x) - 1
            edge_index += [[input2_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input2_idx, output_idx]]
    
    if sign_final == 1:
        x += [one_hot(2, n_vtypes)]
        node_types2 += [2]
        not_idx = len(x) - 1
        edge_index += [[index_final_and, not_idx]]
    

    x = torch.cat(x, dim=0).float()
    edge_index = torch.tensor(edge_index).t().contiguous()

    g = Data(x=x, edge_index=edge_index)
    g.num_literals = n_inputs

    add_order_info(g) # What's the purpose of this info?

    # to be able to use igraph methods in DVAE models
    g.vs = [{'type': t} for t in node_types2]

    # Add Literal labels
    g.solution = None
    if solution:
        g.solution = torch.tensor(solution, dtype=torch.long)
        
    return g, n_variables


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('abcaig_dir', action='store', type=str)
    parser.add_argument('graph_dir', action='store', type=str)
    parser.add_argument('dataset_name', action='store', type=str)
    parser.add_argument('--aig_solution_dir', type=str, default=None)
    parser.add_argument('--gformat', type=str, default='pyg', choices=['pyg', 'igraph'])
    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

    opts = parser.parse_args()

    g_list, graph_args = aig2graph(opts.abcaig_dir, opts.aig_solution_dir)
    print('AIG graph data size: %d' % len(g_list))


    pkl_name = join(opts.graph_dir, opts.dataset_name + '.pkl')
    print('Saving Graph dataset to %s' % pkl_name)
    with open(pkl_name, 'wb') as f:
        pickle.dump((g_list, graph_args), f)
    

    
