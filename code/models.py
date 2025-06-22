import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv

class GAT_Encoder(torch.nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_head=12):
        super(GAT_Encoder, self).__init__()
        self.n_feat = nfeat
        self.hid = nhid
        self.n_head = n_head
        self.dropout = dropout
        self.conv1 = GATConv(self.n_feat, self.hid, heads=self.n_head, edge_dim=self.hid, concat=True, dropout=self.dropout)
        self.conv2 = GATConv(self.hid * self.n_head, self.hid, heads=self.n_head, concat=False, dropout=self.dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x, F.log_softmax(x, dim=1)



class BaseClassifier(nn.Module):
    def __init__(self, nhid, nclass):
        super(BaseClassifier, self).__init__()
        self.hidden_state = nhid
        self.fc = nn.Linear(nhid, nclass, bias=False)

    def forward(self, spt_embedding_i, loss_type, scale=0.1):
        if loss_type == 'crossentropy':
            wf = F.linear(F.normalize(spt_embedding_i, p=2, dim=1), F.normalize(self.fc.weight, p=2, dim=1))/scale
        else:
            wf = F.linear(F.normalize(spt_embedding_i, p=2, dim=1), F.normalize(self.fc.weight, p=2, dim=1))
        return wf
