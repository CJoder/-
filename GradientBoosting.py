import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import time
import argparse
from torchvision import transforms, utils
import pandas as pd
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda:0')

class GraphConvolutionSparse(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionSparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(Module):
    def __init__(self, dropout = 0., act = F.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, inputs):
        x = torch.transpose(inputs,0,1)
        x = torch.matmul(inputs, x)
        # x = torch.reshape(x, [-1])
        outputs = torch.matrix_power(x, 2)
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.dropout) + ' -> ' \
               + str(self.act) + ')'

class GCNVAE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNVAE, self).__init__()

        self.gc1 = GraphConvolution(nhid, nclass)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolutionSparse(nfeat, nhid)
        self.inn = InnerProductDecoder()
        self.dropout = dropout

    def encoder(self, x, adj):
        h1 = F.relu(self.gc3(x, adj))
        z_mean = torch.matrix_power(self.gc1(h1, adj), 2)
        z_std = torch.matrix_power(self.gc2(h1, adj), 2)
        return z_mean, z_std

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def decoder(self, z):
        return F.relu(self.inn(z))

    def forward(self, x, adj):
        mu, val = self.encoder(x, adj)
        z = self.reparameterize(mu, val)
        y = self.decoder(z)
        return mu, val, y

def get_new_data(DEVICE, batch_size, shuffle=True):
    '''
    Trace数据集
    输出包括Trace邻接矩阵，Trace的特征矩阵，对应的label
    '''
    feature_name = "C:\\Users\\14831\\Desktop\\data\\x_train_1.npy"
    adj_name = "C:\\Users\\14831\\Desktop\\data\\adj_x_1.npy"
    label_name = "C:\\Users\\14831\\Desktop\\data\\train_y_1.npy"
    feature = np.load(feature_name)
    adj = np.load(adj_name)
    label = np.load(label_name)

    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    for i in range(adj.shape[0]):
        adj[i] = adj[i] + np.multiply(adj[i].T, adj[i].T > adj[i]) - np.multiply(adj[i], adj[i].T > adj[i])
    # adj = adj + np.multiply(adj.T, adj.T > adj) - np.multiply(adj, adj.T > adj)

    for i in range(adj.shape[0]):
        adj[i] = normalize(adj[i] + sp.eye(adj[i].shape[0]))

    feature = feature[:-1]
    train_x_tensor = torch.from_numpy(feature).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(adj).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_fuction(x, m, l, y, q):
    L = (1 + (q - 1) * y)
    MLD = torch.mm(1 - y, x) + torch.mm(L, (torch.log(1 + torch.exp(-abs(x))) + F.relu(-x)))
    KLD = -0.5 * torch.sum(1 + l - m.pow(2) - l.exp())
    return torch.mean(MLD + KLD)


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



# Model and optimizer
model = GCNVAE(nfeat=5,
            nhid=2,
            nclass=101,
            dropout=0)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
model.cuda()


def train(epoch, q):
    print("========================TRAIN MODE============================")
    model.train()
    train_loss = 0
    train_loader = get_new_data(DEVICE, 16, True)
    for batch_index, batch_data in enumerate(train_loader):
        features, adj = batch_data
        if args.cuda:
            features = features.cuda()
            adj = adj.cuda()
        optimizer.zero_grad()
        mu, val, y = model(features[batch_index], adj[batch_index])
        train_loss = loss_fuction(adj[batch_index], mu, val, y, q)
        train_loss.backward()
        optimizer.step()
        print(batch_index, train_loss)


def test(epoch, q):
    print("========================TEST MODE============================")
    model.eval()
    test_loss = 0
    train_loader = get_new_data(DEVICE, 16, True)
    print(train_loader)
    for batch_index, batch_data in enumerate(train_loader):
        features, adj = batch_data
        if args.cuda:
            features = features.cuda()
            adj = adj.cuda()
        mu, val, y = model(features, adj)
        test_loss = loss_fuction(adj, mu, val, y, q)
        print(epoch, test_loss)


if __name__ == "__main__":
    adj = np.array(pd.read_csv("C:\\Users\\14831\\Desktop\\data\\adj_1_1700.csv"))
    q = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    for epoch in range(200):
        train(epoch, q)
        # test(epoch, q)