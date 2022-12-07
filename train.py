from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from utils import load_data2, accuracy
from models import GCN


torch.autograd.set_detect_anomaly(True)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, pairwise, unnormal_features = load_data2()
print('data loaded')

# Model and optimizer
model = GCN(nfeat=features[0].shape[1],
            nhid=args.hidden,
            nclass=args.hidden,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def loss(output, pairwise):
    loss = 0
    for i in range(len(pairwise)):
        for j in range(i+1, len(pairwise)):
            #print(output[i], output[j], torch.sqrt(torch.sum((output[i]-output[j])**2)+10**-5), pairwise[i][j])
            tmp =  (torch.sqrt(torch.sum((output[i]-output[j])**2)+10**-5) - pairwise[i][j])**2
            loss+=tmp
    #print(loss)
    return loss


def train(epoch):
    t = time.time()
    model.train()
    avg_loss = 0
    for i in range(16):
        idx = random.sample(range(len(features)),1)[0]
        optimizer.zero_grad()
        output = model(features[idx], adj[idx])
        loss_train = loss(output, pairwise[idx])
        
        # #l2 norm
        # l2_lambda = 0.001
        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss_train = loss_train + l2_lambda * l2_norm
        avg_loss+=loss_train
        loss_train.backward()
        #print(loss_train)
        #print(model.gc2.weight)
        
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(avg_loss.item()/16),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(100):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

torch.save(model.state_dict(), "6lsaved100.model")