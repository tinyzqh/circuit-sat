from __future__ import print_function
import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models import DVAEncoder
# from bayesian_optimization.evaluate_BN import Eval_BN


parser = argparse.ArgumentParser(description='Train GNN (The encoder part from D-VAE) for AIGs')
# general settings
parser.add_argument('--data-type', default='AIG', choices=['AIG'],
                    help='AIG format')
parser.add_argument('--igraph-dir', default='data', type=str)
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


'''Prepare data'''
args.res_dir = 'results/{}{}'.format(args.data_name, args.save_appendix)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

# Only use SR10 validation for now.
train_pkl = os.path.join(args.igraph_dir, args.data_name + '_validation.pkl')
validation_pkl = os.path.join(args.igraph_dir, args.data_name + '_validation.pkl')


# Load pre-stored pickle data
# Need to consider testing-only case
if os.path.isfile(train_pkl):
    with open(train_pkl, 'rb') as f:
        train_data, graph_train_args = pickle.load(f)
else:
    raise('Training data no found...')

if os.path.isfile(validation_pkl):
    with open(validation_pkl, 'rb') as f:
        test_data, _ = pickle.load(f)
else:
    raise('Validation data no found')


# construct train data
# if args.no_test:
#     train_data = train_data + test_data

if args.small_train:
    train_data = train_data[:100]


'''Prepare the model'''
# model
model = eval(args.model)(
        graph_train_args.max_n, 
        graph_train_args.num_vertex_type, 
        graph_train_args.num_edge_type,
        hs=args.hs, 
        nz=args.nz, 
        bidirectional=args.bidirectional
        )
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)


if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'))
else:
    if args.continue_from is not None:
        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 
                                              'model_checkpoint{}.pth'.format(epoch)))
        load_module_state(optimizer, os.path.join(args.res_dir, 
                                                  'optimizer_checkpoint{}.pth'.format(epoch)))
        load_module_state(scheduler, os.path.join(args.res_dir, 
                                                  'scheduler_checkpoint{}.pth'.format(epoch)))

# plot sample train/test graphs
if not os.path.exists(os.path.join(args.res_dir, 'train_graph_id0.pdf')) or args.reprocess:
    for data in ['train_data', 'test_data']:
        G = [g for g, y in eval(data)[:10]]
        for i, g in enumerate(G):
            name = '{}_graph_id{}'.format(data[:-5], i)
            plot_DAG(g, args.res_dir, name, data_type=args.data_type)


'''
Define train/test functions.
For these two functions, 
'''
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    # recon_loss = 0
    # kld_loss = 0
    # pred_loss = 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(pbar):
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            g_embedding = model.encode(g_batch)
            binary_logit = model.classifier(g_embedding)
            y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
            loss= model.loss(binary_logit,y_batch)
            
            loss.backward()
            optimizer.step()

            predicted = (binary_logit > 0).to(float)
            correct += y_batch.eq(predicted).sum().item()  
            total += len(g_batch)
            train_loss += float(loss)

            pbar.set_description('Epoch: %d, loss: %0.4f, Acc: %.3f%% (%d/%d)' % (
                             epoch, loss.item()/len(g_batch), 100.*correct/total, correct, total))
            g_batch = []
            y_batch = []

    train_loss /= len(train_data)
    acc = correct / len(train_data)

    print('====> Epoch: {} Average loss: {:.4f}, Accuracy: {:.4f}'.format(
          epoch, train_loss, acc))

    return train_loss, acc



def test():
    # test recon accuracy
    model.eval()
    encode_times = 10
    test_loss = 0
    correct = 0
    total = 0
    print('Testing begins...')
    pbar = tqdm(test_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(pbar):
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
            g = model._collate_fn(g_batch)
            g_embedding = model.encode(g_batch)
            binary_logit = model.classifier(g_embedding)
            predicted = (binary_logit > 0).to(float)
            loss = model.loss(binary_logit, y_batch)
            correct += y_batch.eq(predicted).sum().item()
            total += len(g_batch)
            test_loss += loss.item()

            pbar.set_description('loss: {:.4f}, Acc: %.3f%% (%d/%d)'.format(pred_loss.item()/len(g_batch), 100.*correct/total, correct, total))

            g_batch = []
            y_batch = []
    test_loss /= len(test_data)
    test_acc = correct / len(test_data)
    print('Test average loss: {0}, Accuracy: {1:.4f}'.format(
        pred_loss, correct))
    return test_loss, test_acc
    


'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

# if args.only_test:
#     epoch = args.continue_from
#     #sampled = model.generate_sample(args.sample_number)
#     #save_latent_representations(epoch)
#     visualize_recon(300)
#     #interpolation_exp2(epoch)
#     #interpolation_exp3(epoch)
#     #prior_validity(True)
#     #test()
#     #smoothness_exp(epoch, 0.1)
#     #smoothness_exp(epoch, 0.05)
#     #interpolation_exp(epoch)
#     pdb.set_trace()

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
  
    train_loss, train_acc = train(epoch)
    pred_loss = 0.0
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.2f} {:.2f} \n".format(
            train_loss, 
            train_acc
            ))
    scheduler.step(train_loss)
    if epoch % args.save_interval == 0:
        print("save current model...")
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)
        
        losses = np.loadtxt(loss_name)
        if losses.ndim == 1:
            continue
        fig = plt.figure()
        num_points = losses.shape[0]
        plt.plot(range(1, num_points+1), losses[:, 0], label='Train_Loss')
        plt.plot(range(1, num_points+1), losses[:, 1], label='Train_Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Training')
        plt.legend()
        plt.savefig(loss_plot_name)
    
    if not args.no_test:
        test_loss, test_acc = test()
        with open(test_results_name, 'a') as result_file:
            result_file.write("Epoch {} Test recon loss: {} | Acc: {:.4f}".format(
            epoch, test_loss, test_acc))




pdb.set_trace()
