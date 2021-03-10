from __future__ import print_function
import os
import sys
import math
import pickle
import argparse
import random
from loguru import logger
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
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
from models import *
from models_pyg import *

from config import parser


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

args.exp_name = '{}_{}_{}_hs{:d}_gs{:d}_nr{:d}_lr{:.2e}_b{:d}_bi{:d}_in{:d}'.format(args.train_data, args.test_data, args.model, args.hs, args.gs, 
                            args.n_rounds, args.lr, args.batch_size, int(args.bidirectional), int(args.no_invert))
log_dir = os.path.join(args.log_dir, args.exp_name + '.log')
logger.add(log_dir)
logger.info(args)
writer = SummaryWriter(logdir='runs/lstm')

logger.info('Using device: {}'.format(device))

args.res_dir = os.path.join(args.model_dir, args.exp_name)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
args.fig_dir = os.path.join(args.fig_dir, args.train_data)
if not os.path.exists(args.fig_dir):
    os.makedirs(args.fig_dir)

'''Prepare data'''
logger.info('Preparing data...')

train_pkl = os.path.join(args.data_dir, args.train_data + '_train.pkl')
validation_pkl = os.path.join(args.data_dir, args.test_data + '_validation.pkl')
if not args.only_test:
    if os.path.isfile(train_pkl):
        with open(train_pkl, 'rb') as f:
            train_data, graph_train_args = pickle.load(f)
    else:
        raise KeyError('Training data not found..')

if os.path.isfile(validation_pkl):
    with open(validation_pkl, 'rb') as f:
        test_data, _ = pickle.load(f)
else:
    raise KeyError('Validation data no found..')

logger.info('Data statistics:')
logger.info('# of training samples: {:d}'.format(len(train_data)))
n_SAT, total = cal_postive_percentage(train_data)
logger.info('SAT percentage {:.2f} / ({:.2f}, {:.2f}) in training data.'.format(n_SAT/total, n_SAT, total))

logger.info('# of validation samples: {:d}'.format(len(test_data)))#, file=log_file, flush=True)
n_SAT, total = cal_postive_percentage(test_data)
logger.info('SAT percentage {:.2f} / ({:.2f}, {:.2f}) in test data.'.format(n_SAT/total, n_SAT, total))

if args.small_train:
    train_data = train_data[:100]
    logger.info('# of training samples shrink: {:d}'.format(len(train_data)))


'''Prepare the model'''
# model
# model = eval(args.model)(
#         graph_train_args.max_n, 
#         graph_train_args.num_vertex_type, 
#         graph_train_args.num_edge_type,
#         hs=args.hs, 
#         gs=args.gs,
#         n_rounds=args.n_rounds,
#         bidirectional=args.bidirectional,
#         vid=args.no_invert
#         )
model = DVAEncoder_PYG(
    graph_train_args.max_n, 
    graph_train_args.num_vertex_type, 
    graph_train_args.num_edge_type,
    hs=args.hs, bidirectional=True
)
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)
logger.info(model)

best_acc = 0.0
start_epoch = 0

if args.continue_from is not None:
    logger.info('Continue training from {}...'.format(args.continue_from))
    ckpt = torch.load(os.path.join(args.res_dir, args.continue_from))
    start_epoch = ckpt['epoch']
    load_module_state(model, ckpt['state_dict'])
    load_module_state(optimizer, ckpt['optimizer'])
    load_module_state(scheduler, ckpt['scheduler'])

# plot sample train/test graphs
if not os.path.exists(os.path.join(args.fig_dir, 'train_graph_id0.png')):
    logger.info('Plotting sample graphs...')
    for data in ['train_data', 'test_data']:
        G = [g for g, y in eval(data)[:5]]
        for i, g in enumerate(G):
            name = '{}_graph_id{}'.format(data[:-5], i)
            plot_DAG(g, args.fig_dir, name, data_type=args.data_type)

'''
Define train/test functions.
'''
def train(epoch):
    model.train()
    train_loss = 0
    TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()

    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    y_batch = []
    batch_idx = 0
    for i, (g, y) in enumerate(pbar):
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            batch_idx += 1
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            # binary_logit = model(g_batch)
            binary_logit, predicted_solutions, solutions = model(g_batch)
            y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
            # loss= model.loss(binary_logit, y_batch)
            loss = model.graph_loss(binary_logit, y_batch) + model.solution_loss(predicted_solutions, solutions)

            
            loss.backward()
            optimizer.step()
            if args.check_inteval and (batch_idx % args.check_inteval) == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), batch_idx)

            predicted = (binary_logit > 0).to(float)
            TP += (predicted.eq(1) & y_batch.eq(1)).sum().item()
            TN += (predicted.eq(0) & y_batch.eq(0)).sum().item()
            FN += (predicted.eq(0) & y_batch.eq(1)).sum().item()
            FP += (predicted.eq(1) & y_batch.eq(0)).sum().item()
            TOT = TP + TN + FN + FP

            train_loss += loss.item()
            # The calculation of True positive, etc seems wrong...
            pbar.set_description('Epoch: %d, loss: %0.4f, Acc: %.3f%%, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
                             epoch, loss.item()/len(g_batch), 100.*(TP + TN) * 1.0 / TOT, TP * 1.0 / TOT, TN * 1.0 / TOT, FN * 1.0 / TOT, FP * 1.0 / TOT))
            g_batch = []
            y_batch = []

    train_loss /= len(train_data)
    acc = ((TP + TN) * 1.0 / TOT).item()

    logger.info('====> Epoch Train: {:d} Average loss: {:.4f}, Accuracy: {:.4f}'.format(
          epoch, train_loss, acc))

    return train_loss, acc


def test(epoch):
    model.eval()
    test_loss = 0
    TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()

    pbar = tqdm(test_data)
    g_batch = []
    y_batch = []
    with torch.no_grad():
        for i, (g, y) in enumerate(pbar):
            g_batch.append(g)
            y_batch.append(y)
            if len(g_batch) == args.batch_size or i == len(train_data) - 1:
                y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                g = model._collate_fn(g_batch)
                g_embedding = model.encode(g_batch)
                binary_logit = model.classifier(g_embedding)
                predicted = (binary_logit > 0).to(float)
                loss = model.loss(binary_logit, y_batch)

                predicted = (binary_logit > 0).to(float)
                TP += (predicted.eq(1) & y_batch.eq(1)).sum().item()
                TN += (predicted.eq(0) & y_batch.eq(0)).sum().item()
                FN += (predicted.eq(0) & y_batch.eq(1)).sum().item()
                FP += (predicted.eq(1) & y_batch.eq(0)).sum().item()
                TOT = TP + TN + FN + FP

                test_loss += loss.item()
                # The calculation of True positive, etc seems wrong...
                pbar.set_description('Test Epoch: %d, loss: %0.4f, Acc: %.3f%%, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
                                epoch, loss.item()/len(g_batch), 100.*(TP + TN) * 1.0 / TOT, TP * 1.0 / TOT, TN * 1.0 / TOT, FN * 1.0 / TOT, FP * 1.0 / TOT))
                g_batch = []
                y_batch = []

    test_loss /= len(train_data)
    acc = ((TP + TN) * 1.0 / TOT).item()

    logger.info('====> Epoch Test: {:d} Average loss: {:.4f}, Accuracy: {:.4f}'.format(
          epoch, test_loss, acc))

    return test_loss, acc



'''Training begins here'''
min_loss = math.inf
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')

if os.path.exists(loss_name):
    os.remove(loss_name)

if args.only_test:
    test_loss, test_acc = train(epoch)
    

for epoch in range(start_epoch + 1, args.epochs + 1):
  
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.2f} {:.2f} \n".format(
            train_loss, 
            train_acc
            ))
    scheduler.step(train_loss)
    if epoch % args.save_interval == 0:
        logger.info("Save current model...")
        ckpt = {'epoch': epoch+1, 'acc': best_acc, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        ckpt_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        torch.save(ckpt, ckpt_name)
        
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
    
    if test_acc >= best_acc:
        best_acc = test_acc
        ckpt = {'epoch': epoch+1, 'acc': best_acc, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        ckpt_name = os.path.join(args.res_dir, 'model_best.pth'.format(epoch))
        torch.save(ckpt, ckpt_name)

ckpt = {'epoch': epoch+1, 'acc': best_acc, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
ckpt_name = os.path.join(args.res_dir, 'model_last.pth'.format(epoch))
torch.save(ckpt, ckpt_name)
