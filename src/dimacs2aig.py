import os
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


