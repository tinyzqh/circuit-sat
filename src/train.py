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
from models import DVAEncoder

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

args.exp_name = '{}_{}_hs{:d}_nz{:d}_nr{:d}_lr{:.2e}_b{:d}_bi{:d}_in{:d}'.format(args.data_name, args.model, args.hs, args.nz, 
                            args.n_rounds, args.lr, args.batch_size, int(args.bidirectional), int(args.invert_hidden))
log_dir = os.path.join(args.log_dir, args.exp_name + '.log')
logger.add(log_dir)
logger.info(args)
writer = SummaryWriter(logdir='runs/lstm')

logger.info('Using device: {}'.format(device))

args.res_dir = os.path.join(args.model_dir, args.exp_name)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
args.fig_dir = os.path.join(args.fig_dir, args.data_name)
if not os.path:
    os.makedirs(args.fig_dir)

'''Prepare data'''
logger.info('Preparing data...')

train_pkl = os.path.join(args.data_dir, args.data_name + '_train.pkl')
validation_pkl = os.path.join(args.data_dir, args.data_name + '_validation.pkl')
# Load pre-stored pickle data
# Need to consider testing-only case
if not args.only_test:
    if os.path.isfile(train_pkl):
        with open(train_pkl, 'rb') as f:
            train_data, graph_train_args = pickle.load(f)
    else:
        raise BaseException('Training data not found..')

if os.path.isfile(validation_pkl):
    with open(validation_pkl, 'rb') as f:
        test_data, _ = pickle.load(f)
else:
    raise BaseException('Validation data no found..')

logger.info('# of training samples: {:d}'.format(len(train_data)))

SAT=0
total = 0
for (graph, y) in train_data:
    if y == 1: SAT += 1
    total += 1
logger.info('SAT percentage {:.2f} / {:.2f} {:.2f} in training data.' % (SAT/total, SAT, total))

logger.info('# of validation samples: {:d}'.format(len(test_data)))#, file=log_file, flush=True)
SAT=0
total = 0
for (graph, y) in test_data:
    if y == 1: SAT += 1
    total += 1
logger.info('SAT percentage {:.2f} / {:.2f} {:.2f} in test data.' % (SAT/total, SAT, total))

exit()

if args.small_train:
    train_data = train_data[:100]
    print('# of training samples shrink: ', len(train_data))


'''Prepare the model'''
# model
model = eval(args.model)(
        graph_train_args.max_n, 
        graph_train_args.num_vertex_type, 
        graph_train_args.num_edge_type,
        hs=args.hs, 
        nz=args.nz,
        n_rounds=args.n_rounds,
        bidirectional=args.bidirectional,
        vid=args.invert_hidden
        )
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)
print(model)


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
            binary_logit = model(g_batch)
            y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
            loss= model.loss(binary_logit, y_batch)
            
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

            train_loss += float(loss.item())
            # The calculation of True positive, etc seems wrong...
            pbar.set_description('Epoch: %d, loss: %0.4f, Acc: %.3f%%, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
                             epoch, loss.item()/len(g_batch), (TP + TN) * 1.0 / TOT, TP * 1.0 / TOT, TN * 1.0 / TOT, FN * 1.0 / TOT, FP * 1.0 / TOT))
            g_batch = []
            y_batch = []

    train_loss /= len(train_data)
    acc = ((TP + TN) * 1.0 / TOT).item()

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

            pbar.set_description('loss: {:.4f}, Acc: %.3f%% (%d/%d)'.format(test_loss/len(g_batch), 100.*correct/total, correct, total))

            g_batch = []
            y_batch = []
    test_loss /= len(test_data)
    test_acc = correct / len(test_data)
    print('Test average loss: {0}, Accuracy: {1:.4f}'.format(
        test_loss, correct))
    return test_loss, test_acc
    


'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')

if os.path.exists(loss_name):
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

