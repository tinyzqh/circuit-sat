'''
Parse AIG graph (*.aag) into igraph format (https://github.com/igraph/python-igraph)
'''
from os import listdir
from os.path import join
import pickle
import igraph
import argparse

# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()


def aig2igraph(folder_name, n_vtypes=4, n_etypes=3, print_interval=100):
    '''
    load AIG graph (*.aag) to igraph.
    n_vtypes: # of vertex types; 4 in AIG case;
    n_etypes: # of edge types; 3 in AIG case;
    '''
    g_list = []
    max_n = 0 # maximum number of nodes

    for (i, filename) in enumerate([filename for filename in listdir(folder_name) if filename.endswith('.aag')]):
        aag_path = join(folder_name, filename)
        with open(aag_path, 'r') as f:
            lines = f.readlines()
        g, n = decode_aag_to_igraph(lines)
        y = 1 if 'sat=1' in filename else 0

        max_n = max(max_n, n)
        g_list.append((g, y))

    graph_args.num_vertex_type = n_vtypes
    graph_args.num_edge_type = n_etypes
    graph_args.max_n = max_n  # maximum number of nodes
    # graph_args.START_TYPE = 0  # predefined start vertex type
    # graph_args.END_TYPE = 3 # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    print('# AIG graph: %d' % ng)
    return g_list, graph_args

def decode_aag_to_igraph(lines):

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
    # Construct AIG graph
    g.add_vertices(n_variables + 2)
    g.vs[0]['v_type'] = 0   # input node (a virtual starting node connecting to all of input literals)
    for i in range(n_inputs):
        g.vs[i + 1]['v_type'] = 1   # literal input node
        g.add_edge(0, i+1, e_type=2) # always connect the virtual input node to literal input node.
    for i in range(n_inputs + 1, n_inputs + 1 + n_and):
        g.vs[i]['v_type'] = 2
        # not sure whether it is ok to add edge here.
    g.vs[n_variables+1]['v_type'] = 3           

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
        g.add_edge(input1_idx, output_idx, e_type=sign1_idx)
        # Second edge
        input2_idx = int(line[2]) // 2
        sign2_idx = int(line[2]) % 2
        g.add_edge(input2_idx, output_idx, e_type=sign2_idx)
    
    g.add_edge(index_final_and, n_variables+1, e_type=sign_final)
    # In D-VAE, the nodes 0, 1, ..., n+1 are in a topological order
    # Is that true in AIG graph?

    print(g)
    print(g.es[[g.get_eid(i, 4) for i in g.predecessors(4)]]['e_type'])
    return g, n_variables+2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('abcaig_dir', action='store', type=str)
    parser.add_argument('igraph_dir', action='store', type=str)
    parser.add_argument('dataset_name', action='store', type=str)
    # parser.add_argument('gen_log', action='store', type=str)
    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

    opts = parser.parse_args()

    g_list, graph_args = aig2igraph(opts.abcaig_dir)

    pkl_name = join(opts.igraph_dir, opts.dataset_name + '.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump((g_list, graph_args), f)
    

    