import argparse


parser = argparse.ArgumentParser(description='Train GNN (The encoder part from D-VAE) for AIGs')
# general settings
parser.add_argument('--data-type', default='AIG', choices=['AIG'],
                    help='AIG format')
parser.add_argument('--data-name', default='sr10', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=4, help='number of different node types, \
                    4 for AIG setting')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, only training.')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='DVAEncoder', choices=['DVAEncoder'],help='model to use: DVAE. No other options for now.')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=100, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')
parser.add_argument('--n_rounds', type=int, default=10, metavar='N',
                    help='The number of rounds for information propagation.')
# parser.add_argument('--predictor', action='store_true', default=False,
#                     help='whether to train a performance predictor from latent\
#                     encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# parser.add_argument('--all-gpus', action='store_true', default=False,
#                     help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


parser.add_argument('--log-dir', type=str, default='log/', help='log folder dir')
parser.add_argument('--model-dir', type=str, default='model/', help='model folder dir')
parser.add_argument('--data-dir', type=str, default='data/', help='data folder dir')

