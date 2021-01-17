from os import listdir
from os.path import join, splitext
import subprocess
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('aig_dir', action='store', type=str)
    parser.add_argument('abcaig_dir', action='store', type=str)
    parser.add_argument('gen_log', action='store', type=str)
    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

    opts = parser.parse_args()

    f = open(opts.gen_log, 'w')


    for (i, filename) in enumerate(listdir(opts.aig_dir)):
        if i % opts.print_interval == 0: print("Processing # [%d] instances..." % i, file=f)
        aig_name = join(opts.aig_dir, filename)
        aig_abc_name = join(opts.abcaig_dir, splitext(filename)[0] + '_abc.aig')
        aig_abc_name_ascii = join(opts.abcaig_dir, splitext(filename)[0] + '_abc.aag')
        print("\"r %s; b; ps; b; rw -l; rw -lz; b; rw -lz; b; ps; cec; w %s\"" % (aig_name, aig_abc_name))

        subprocess.call(["./src/abc/abc", "-c", "r %s; b; ps; b; rw -l; rw -lz; b; rw -lz; b; ps; cec; w %s" % (aig_name, aig_abc_name)])
        subprocess.call(["./src/aiger/aiger/aigtoaig", aig_abc_name, aig_abc_name_ascii])



        



