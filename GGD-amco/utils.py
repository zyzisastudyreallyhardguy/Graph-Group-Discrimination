import numpy as np
import networkx as nx
import torch
from cytoolz import curry
from scipy.linalg import fractional_matrix_power, inv
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import multiprocessing as mp
import os

#borrowed from subg-con https://github.com/yzjiao/Subg-Con
class PPR:
    # Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=20, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.node_num = adj_mat.shape[0]
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()

    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)

        # idx = scores.argsort()[::-1][:self.maxsize]
        # neighbor = np.array(x.indices[idx])

        # seed_idx = np.where(neighbor == seed)[0]
        # if seed_idx.size == 0:
        #     neighbor = np.append(np.array([seed]), neighbor)
        # else:
        #     seed_idx = seed_idx[0]
        #     neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
        #
        # assert np.where(neighbor == seed)[0].size == 1
        # assert np.where(neighbor == seed)[0][0] == 0

        return scores

    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            print('Processing node {}.'.format(seed))
            scores = self.search(seed)
            torch.save(scores, ppr_path)
        else:
            print('File of node {} exists.'.format(seed))

    def search_all(self, node_num, path):
        neighbor = {}
        if os.path.isfile(path + '_scores') and os.stat(path + '_scores').st_size != 0:
            print("Exists scores file")
            neighbor = torch.load(path + '_scores')
        else:
            print("Extracting top k neighbors for nodes")
            os.system('mkdir {}'.format(path))
            with mp.Pool() as pool:
                list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))

            print("Finish Extracting")
            for i in range(node_num):
                neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
            torch.save(neighbor, path + '_scores')
            os.system('rm -r {}'.format(path))
            print("Finish Writing")
        return neighbor

def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    # Self-loops
    A_loop = sp.eye(N) + A

    # Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec

    return T_S

def compute_heat(graph: nx.Graph, t=5, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)