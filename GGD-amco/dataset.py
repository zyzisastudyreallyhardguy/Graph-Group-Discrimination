from dgl.data import CoraGraphDataset, CitationGraphDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr, gdc
import scipy.sparse as sp
import networkx as nx
import numpy as np
from load_npz import load_pitfall_dataset
import os


def download(dataset):
    if dataset == 'cora':
        return CoraGraphDataset()
    elif dataset == 'citeseer':
        return CitationGraphDataset(name=dataset)
    elif dataset == 'pubmed':
        return CitationGraphDataset(name=dataset)
    elif dataset == 'computer':
        return AmazonCoBuyComputerDataset()
    elif dataset == 'photo':
        return AmazonCoBuyPhotoDataset()
    else:
        return None


def load(dataset):
    datadir = os.path.join('data', dataset)

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(dataset)

        adj = nx.to_numpy_array(ds.graph)
        # diff = compute_ppr(ds.graph, 0.2)
        feat = ds.features[:]
        labels = ds.labels[:]

        idx_train = np.argwhere(ds.train_mask == 1).reshape(-1)
        idx_val = np.argwhere(ds.val_mask == 1).reshape(-1)
        idx_test = np.argwhere(ds.test_mask == 1).reshape(-1)

        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
        np.save(f'{datadir}/idx_train.npy', idx_train)
        np.save(f'{datadir}/idx_val.npy', idx_val)
        np.save(f'{datadir}/idx_test.npy', idx_test)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')
        idx_train = np.load(f'{datadir}/idx_train.npy')
        idx_val = np.load(f'{datadir}/idx_val.npy')
        idx_test = np.load(f'{datadir}/idx_test.npy')

    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        # epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        # avg_degree = np.sum(adj) / adj.shape[0]
        # epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
        #                               for e in epsilons])]

        # diff[diff < epsilon] = 0.0
        # scaler = MinMaxScaler()
        # scaler.fit(diff)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, feat, labels, idx_train, idx_val, idx_test

def load_coauthor(dataset):
    adj, features, labels, train_mask, val_mask, test_mask = load_pitfall_dataset(dataset)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    features, _ = preprocess_features(features)

    return adj, features, labels, train_mask, val_mask, test_mask
