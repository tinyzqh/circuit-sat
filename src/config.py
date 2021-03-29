import argparse


parser = argparse.ArgumentParser(description='Circuit-SAT: Learning to Solve Circuit-SAT, Network: DG-DAGRNN')
# general settings
parser.add_argument('--data-type', default='AIG', choices=['AIG'],
                    help='The format to represent circuits, AIG format')
parser.add_argument('--train-data', default='sr3to4', help='graph dataset name')
parser.add_argument('--test-data', default='sr5', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=3, help='number of different node types, \
                    3 for DG setting')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--check-inteval', default=None, type=int, help='The intevral to check model weights.')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='DGDAGRNN', choices=['DGDAGRNN'],help='model to use: DGDAGRNN. No other options for now.')
parser.add_argument('--continue-from', type=str, default=None, 
                    help="checkpoint file name to continue training")
parser.add_argument('--vhs', type=int, default=100, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--chs', type=int, default=30, metavar='N',
                    help='hidden size of Classifiers')
parser.add_argument('--temperature', type=int, default=5, metavar='N',
                    help='initial value for temperature')
parser.add_argument('--k-step', type=int, default=10, metavar='N',
                    help='the value for step funtion parameter k.')
parser.add_argument('--num-rounds', type=int, default=10, metavar='N',
                    help='The number of rounds for information propagation.')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--weight-decay', type=float, default=1e-10, 
                    help='weight decay (default: 1e-10)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


parser.add_argument('--log-dir', type=str, default='log/', help='log folder dir')
parser.add_argument('--model-dir', type=str, default='model/', help='model folder dir')
parser.add_argument('--fig-dir', type=str, default='figs/', help='figure folder dir')
parser.add_argument('--data-dir', type=str, default='data/', help='data folder dir')

