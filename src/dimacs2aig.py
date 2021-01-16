from os import listdir
from os.path import join, splitext
import subprocess
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dimacs_dir', action='store', type=str)
    parser.add_argument('out_dir', action='store', type=str)
    parser.add_argument('gen_log', action='store', type=str)
    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

    opts = parser.parse_args()

    f = open(opts.gen_log, 'w')


    for (i, filename) in enumerate(listdir(opts.dimacs_dir)):
        if i % opts.print_interval == 0: print("Processing # [%d] instances..." % i, file=f)
        dimacs_name = join(opts.dimacs_dir, filename)
        aig_name = join(opts.out_dir, splitext(filename)[0] + '.aig')
        subprocess.call(["./src/aiger/cnf2aig/cnf2aig", dimacs_name, aig_name])



        



