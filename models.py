import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 64)
        self.gc2 = GraphConvolution(64, 128)
        self.gc3 = GraphConvolution(128, 256)
        self.gc4 = GraphConvolution(256, 256)
        self.gc5 = GraphConvolution(256, 256)
        self.gc6 = GraphConvolution(256, 256)

        self.dropout = dropout

    def forward(self, x, adj,n_conv=5):
        l1 = F.relu(self.gc1(x, adj))
        l1 = F.dropout(l1, self.dropout, training=self.training)
        
        l2 = F.relu(self.gc2(l1, adj))
        l2 = F.dropout(l2, self.dropout, training=self.training)

        l3 = F.relu(self.gc3(l2, adj))
        l3 = F.dropout(l3, self.dropout, training=self.training)

        l4 = F.relu(self.gc4(l3, adj))
        l4 = F.dropout(l4, self.dropout, training=self.training)

        l5 = F.relu(self.gc5(l4+l3, adj))
        l5 = F.dropout(l5, self.dropout, training=self.training)

        l6 = F.relu(self.gc6(l5+l4+l3, adj))
        l6 = F.dropout(l6, self.dropout, training=self.training)

        return l6

