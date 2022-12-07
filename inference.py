from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data2, accuracy, load_data3
from models import GCN

import open3d as o3d

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = GCN(nfeat=3,
            nhid=args.hidden,
            nclass=args.hidden,
            dropout=args.dropout)


model.load_state_dict(torch.load("saved10works.model"))
adj, features, _ , unnormal_features= load_data3()

if args.cuda:
    model.cuda()

model.eval()
xyz = []
embeds = []
for i in range(len(adj)):
    xyz+=list(unnormal_features[i])
    embeds+= list(model(features[i], adj[i]).detach().numpy())



kmeans = KMeans(n_clusters=3, random_state=0).fit(embeds)
labels = kmeans.labels_
colors = np.zeros((len(xyz), 3))
for i in range(3):
    colors[np.argwhere(labels==i)[:,0],i] = 1

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])