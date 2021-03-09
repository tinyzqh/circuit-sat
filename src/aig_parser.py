'''
Parse AIG graph (*.aag) into igraph format (https://github.com/igraph/python-igraph)
or Parse AIG graph (*.aag) into PytorchGeometric Data format(https://github.com/rusty1s/pytorch_geometric)
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


def aig2graph(folder_name, solution_folder_name, format='pyg', n_vtypes=4, n_etypes=3, print_interval=100):
    '''
    load AIG graph (*.aag) to igraph.
    n_vtypes: # of vertex types; 4 in AIG case;
    n_etypes: # of edge types; 3 in AIG case;
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
                    solution = f.read().strip().split(' ')
        with open(aag_path, 'r') as f:
            lines = f.readlines()
        if format == 'igraph':
            # The solution hasn't been considered.
            g, n = decode_aag_to_igraph(lines, solution)
        elif format == 'pyg':
            g, n = decode_aag_to_pyg(lines, solution, n_vtypes, n_etypes)
        if g == None:
            continue
        y = 1 if 'sat=1' in filename else 0

        max_n = max(max_n, n)
        g_list.append((g, y))

    graph_args.num_vertex_type = n_vtypes
    graph_args.num_edge_type = n_etypes
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.literal_label = True if solution_folder_name else False
    # graph_args.START_TYPE = 0  # predefined start vertex type
    # graph_args.END_TYPE = 3 # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    print('# AIG graph: %d' % ng)
    return g_list, graph_args

def decode_aag_to_igraph(lines, solution=None):
    '''
    For AIG, the nodes can be categorized as the input node, internal AND nodes, the output node. The type values for each kind of nodes are as follows:
        * Input node: 0 (one single virtual starting node, be compatible with D-VAE);
        * Literal input node: 1 (input nodes, have a common predecessor Input node);
        * Internal AND nodes: 2;
        * Output node: 3 (connect to the last-level AND gate, but the edge might be Inverter).

        The type values for non-inverter and inverter edge:
        * Non-inverter: 0;
        * Inverter: 1;
        * Edges from the virtual starting node to literal input nodes: 2.
    '''
    g = igraph.Graph(directed=True)

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
        # The graph is UNSAT.
        return None, None
    # Construct AIG graph
    g.add_vertices(n_variables + 2)
    g.vs[0]['v_type'] = 0   # Input node (a virtual starting node connecting to all of input literals)
    for i in range(n_inputs):
        g.vs[i + 1]['v_type'] = 1   # Literal input node
        g.add_edge(0, i+1, e_type=2) # Always connect the virtual input node to literal input node.
    for i in range(n_inputs + 1, n_inputs + 1 + n_and):
        g.vs[i]['v_type'] = 2  # AND node
    g.vs[n_variables+1]['v_type'] = 3   # Output node    

    # Sanity-check the variables 
    for (i, line) in enumerate(lines[1:1+n_inputs]):
        literal = line.strip().split(" ")
        assert len(literal) == 1, 'The literal of input should be single.'
        assert int(literal[0]) == 2 * (i + 1), 'The value of a input literal should be the index of variables mutiplying by two.'

    literal = lines[1+n_inputs].strip().split(" ")[0]
    assert int(literal) == (n_variables * 2) or int(literal) == (n_variables * 2) + 1, 'The value of the output literal shoud be (n_variables * 2)'
    sign_final = int(literal) % 2
    index_final_and = int(literal) // 2

    for (i, line) in enumerate(lines[n_inputs+2: n_inputs+2+n_and]):
        literals = line.strip().split(" ")
        assert len(literals) == 3, 'Invalidate the definition of two-input AND gate.'
        assert int(literals[0]) == 2 * (i + 1 + n_inputs)
    var_def = lines[2+n_variables].strip().split(" ")[0]

    assert var_def == 'i0', 'The definition of variables is wrong.'

    # Add edge
    for (i, line) in enumerate(lines[n_inputs+2: n_inputs+2+n_and]):
        line = line.strip().split(" ")
        # assert len(line) == 3, 'The length of AND lines should be 3.'
        output_idx = int(line[0]) // 2
        assert (int(line[0]) % 2) == 0, 'There is inverter sign in output literal.'
        # First edge
        input1_idx = int(line[1]) // 2
        sign1_idx = int(line[1]) % 2
        g.add_edge(input1_idx, output_idx, e_type=sign1_idx)
        # Second edge
        input2_idx = int(line[2]) // 2
        sign2_idx = int(line[2]) % 2
        g.add_edge(input2_idx, output_idx, e_type=sign2_idx)
    
    g.add_edge(index_final_and, n_variables+1, e_type=sign_final)
    # In D-VAE, the nodes 0, 1, ..., n+1 are in a topological order
    # Is that true in AIG graph?


    return g, n_variables+2



def decode_aag_to_pyg(lines, solution, n_vtypes, n_etypes): 
    '''
    For AIG, the nodes can be categorized as the input node, internal AND nodes, the output node. The type values for each kind of nodes are as follows:
        * Input node: 0 (one single virtual starting node, be compatible with D-VAE);
        * Literal input node: 1 (input nodes, have a common predecessor Input node);
        * Internal AND nodes: 2;
        * Output node: 3 (connect to the last-level AND gate, but the edge might be Inverter).

        The type values for non-inverter and inverter edge:
        * Non-inverter: 0;
        * Inverter: 1;
        * Edges from the virtual starting node to literal input nodes: 2.
    '''
    print(lines)
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
    edge_attr = []
    node_types2 = []

    # Add Input node
    x += [one_hot(0, n_vtypes)]
    node_types2 += [0]

    for i in range(n_inputs):
        # Add Literal node
        x += [one_hot(1, n_vtypes)]
        node_types2 += [1]
        edge_index += [0, i+1]
        edge_attr += [one_hot(2, n_etypes)]
    for i in range(n_inputs+1, n_inputs + 1 + n_and):
        # Add AND node
        x += [one_hot(2, n_vtypes)]
        node_types2 += [2]
    # Add Output
    x += [one_hot(3, n_vtypes)]
    node_types2 += [3]

    # sanity-check the variables 
    for (i, line) in enumerate(lines[1:1+n_inputs]):
        literal = line.strip().split(" ")
        assert len(literal) == 1, 'The literal of input should be single.'
        assert int(literal[0]) == 2 * (i + 1), 'The value of a input literal should be the index of variables mutiplying by two.'

    literal = lines[1+n_inputs].strip().split(" ")[0]
    assert int(literal) == (n_variables * 2) or int(literal) == (n_variables * 2) + 1, 'The value of the output literal shoud be (n_variables * 2)'
    sign_final = int(literal) % 2
    index_final_and = int(literal) // 2

    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        literals = line.strip().split(" ")
        assert len(literals) == 3, 'invalidate the definition of two-input AND gate.'
        assert int(literals[0]) == 2 * (i + 1 + n_inputs)
    var_def = lines[2+n_variables].strip().split(" ")[0]

    assert var_def == 'i0', 'The definition of variables is wrong.'

    # Add edge
    for (i, line) in enumerate(lines[n_inputs+2: n_inputs+2+n_and]):
        line = line.strip().split(" ")
        # assert len(line) == 3, 'The length of AND lines should be 3.'
        output_idx = int(line[0]) // 2
        assert (int(line[0]) % 2) == 0, 'There is inverter sign in output literal.'
        # First edge
        input1_idx = int(line[1]) // 2
        sign1_idx = int(line[1]) % 2
        
        edge_index += [[input1_idx, output_idx]]
        edge_attr += [one_hot(sign1_idx, n_etypes)]

        # Second edge
        input2_idx = int(line[2]) // 2
        sign2_idx = int(line[2]) % 2
        edge_index += [[input2_idx, output_idx]]
        edge_attr += [one_hot(sign2_idx, n_etypes)]
    
    edge_index += [[index_final_and, n_variables+1]]
    edge_attr += [one_hot(sign_final, n_etypes)]

    x = torch.cat(x, dim=0).float()
    edge_index = torch.tensor(edge_index).t().contiguous()
    print(edge_index)
    edge_attr = torch.cat(edge_attr, dim=0).float()

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print(g)
    print(g.keys)
    print(g['x'])
    for key, item in g:
        print("{} found in graph".format(key))
    print(g.num_nodes)
    print(g.num_edges)
    print(g.num_node_features)
    print(g.is_directed())
    print(g.contains_isolated_nodes())
    print(g.contains_self_loops())
    exit()
    # add_order_info(g) # What's the purpose of this info?

    # to be able to use igraph methods in DVAE models
    g.vs = [{'type': t} for t in node_types2]

    # Add Literal labels
    if solution:
        g.y = torch.tensor(solution).float()

    return g, n_variables+2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('abcaig_dir', action='store', type=str)
    parser.add_argument('graph_dir', action='store', type=str)
    parser.add_argument('dataset_name', action='store', type=str)
    parser.add_argument('--aig_solution_dir', type=str, default=None)
    parser.add_argument('--gformat', type=str, default='pyg', choices=['pyg', 'igraph'])
    # parser.add_argument('gen_log', action='store', type=str)
    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

    opts = parser.parse_args()

    g_list, graph_args = aig2graph(opts.abcaig_dir, opts.aig_solution_dir, opts.gformat)
    print('AIG graph data size: %d' % len(g_list))


    pkl_name = join(opts.igraph_dir, opts.dataset_name + '.pkl')
    print('Saving Graph dataset to %s' % pkl_name)
    with open(pkl_name, 'wb') as f:
        pickle.dump((g_list, graph_args), f)
    

    