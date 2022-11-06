from __future__ import print_function, division
import argparse
import random
import numpy as np
from numpy import mean
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from GAT import GraphAttentionLayer
from evaluation import eva
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# torch.cuda.set_device(1)
acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
z_tilde_result = []
kmeans_result = []


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class CDFG(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1, alpha=0.2):
        super(CDFG, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GraphAttentionLayer(n_input, n_enc_1, alpha)
        self.gnn_2 = GraphAttentionLayer(n_enc_1, n_enc_2, alpha)
        self.gnn_3 = GraphAttentionLayer(n_enc_2, n_enc_3, alpha)
        self.gnn_4 = GraphAttentionLayer(n_enc_3, n_z, alpha)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj, M):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.1

        # GCN Module
        h_1 = self.gnn_1(x, adj, M)

        h_2 = self.gnn_2((1 - sigma) * h_1 + sigma * tra1, adj, M)  # 对自动编码器得到的隐层表示与图自动编码器融合

        h_3 = self.gnn_3((1 - sigma) * h_2 + sigma * tra2, adj, M)

        h_4 = self.gnn_4((1 - sigma) * h_3 + sigma * tra3, adj, M)

        h = self.gnn_5((1 - sigma) * h_4 + sigma * z, adj, active=False )

        # s = torch.spmm(adj, h_1)
        s = torch.mm(h_1, h_1.t())
        s = F.softmax(s, dim=1)
        # print(s.size())
        z_3 = torch.mm(s, h)
        # print(z_3.size())
        # z_3 = torch.cat((s, h), dim=1)
        # print(z_3.size())

        predict = F.softmax(z_3, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, h_4


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_cdfg(dataset):
    model = CDFG(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    adj = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cpu()
    adj = adj_dense.cpu()
    # KNN Graph

    # adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)

    y_pred = kmeans.fit_predict(z.data.cpu().numpy())

    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(500):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, _, z_tilde = model(data, adj, M)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')
            acc, nmi, ari, f1 = eva(y, res2, epoch)
            acc_reuslt.append(acc)
            nmi_result.append(nmi)
            ari_result.append(ari)
            f1_result.append(f1)
            z_tilde_result.append(z_tilde.data.cpu().numpy())
            kmeans_result.append(pred.data.cpu().numpy().argmax(1))

        x_bar, q, pred, _, _ = model(data, adj, M)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(pred.data.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='hhar')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.lr = 1e-3
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.k = 1
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.lr = 1e-4
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703
    '''
    if args.name == 'cora':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 7
        args.n_input = 1433

    if args.name == 'wiki':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 17
        args.n_input = 4973

    if args.name == 'pubmed':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 3
        args.n_input = 500
    '''
    print(args)
    train_cdfg(dataset)
    print("ACC: {:.4f}".format(max(acc_reuslt)))
    print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])
