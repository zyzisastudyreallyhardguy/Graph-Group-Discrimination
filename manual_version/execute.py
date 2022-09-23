import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import time
from layers import GCN

from models import LogReg
from utils import process
import os
import copy
import random
import argparse
import sys


class GGD(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GGD, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.lin = nn.Linear(n_h, n_h)

    def forward(self, seq1, seq2, adj, sparse):
        h_1 = self.gcn(seq1, adj, sparse)
        h_2 = self.gcn(seq2, adj, sparse)
        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)
        return logits

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)
        h_2 = h_1.clone().squeeze(0)
        for i in range(5):
            h_2 = adj @ h_2

        h_2 = h_2.unsqueeze(0)

        return h_1.detach(), h_2.detach()

def aug_random_edge(input_adj, drop_percent=0.1):
    drop_percent = drop_percent
    b = np.where(input_adj > 0,
                 np.random.choice(2, (input_adj.shape[0], input_adj.shape[0]), p=[drop_percent, 1 - drop_percent]),
                 input_adj)
    drop_num = len(input_adj.nonzero()[0]) - len(b.nonzero()[0])
    mask_p = drop_num / (input_adj.shape[0] * input_adj.shape[0] - len(b.nonzero()[0]))
    c = np.where(b == 0, np.random.choice(2, (input_adj.shape[0], input_adj.shape[0]), p=[1 - mask_p, mask_p]), b)

    return b

def aug_feature_dropout(input_feat, drop_percent=0.2):
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

if __name__ == '__main__':
    acc_results = []
    import warnings

    warnings.filterwarnings("ignore")

    #setting arguments
    parser = argparse.ArgumentParser('GGD')
    parser.add_argument('--classifier_epochs', type=int, default=100, help='classifier epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--np_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=500, help='Patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Patience')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--drop_prob', type=float, default=0.0, help='Tau value')
    parser.add_argument('--hid_units', type=int, default=512, help='Top-K value')
    parser.add_argument('--sparse', action='store_true', help='Whether to use sparse tensors')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name: cora, citeseer, pubmed, cs, phy')
    parser.add_argument('--num_hop', type=int, default=0, help='graph power')
    parser.add_argument('--n_trials', type=int, default=5, help='number of trails')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    n_trails = args.n_trials
    acc_res = []
    for i in range(n_trails):
        #find the GPU with most available space
        free_gpu_id = get_free_gpu()
        torch.cuda.set_device(int(free_gpu_id))

        dataset = args.dataset

        # training params
        batch_size = args.batch_size
        nb_epochs = args.np_epochs
        patience = args.patience
        classifier_epochs = args.classifier_epochs
        lr = args.lr
        l2_coef = args.l2_coef
        drop_prob = args.drop_prob
        hid_units = args.hid_units
        num_hop = args.num_hop
        sparse = True
        nonlinearity = 'prelu'  # special name to separate parameters

        #load dataset
        if dataset in ['cora','citeseer','pubmed']:
            adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
        else:
            adj = torch.load('data/' + dataset + '_adj.pt')
            adj = sp.csr_matrix(adj.cpu().numpy())
            features = np.load('data/' + dataset + '.npy')
            features = sp.lil_matrix(features)
            labels = np.load('data/' + dataset + '_labels.npy')

            n_values = np.max(labels) + 1
            labels = np.eye(n_values)[labels]

            test_ratio = 0.9

            idx_test = random.sample(list(np.arange(features.shape[0])), int(test_ratio * features.shape[0]))
            remain_num = len(idx_test)
            idx_val = idx_test
            idx_train = list(set(np.arange(features.shape[0])) - set(idx_test))

            train_mask = torch.zeros(features.shape[0]).long()
            train_mask[idx_train] = 1
            test_mask = torch.zeros(features.shape[0]).long()
            test_mask[idx_test] = 1
            val_mask = torch.zeros(features.shape[0]).long()
            val_mask[idx_val] = 1

            train_mask = train_mask.bool()
            test_mask = test_mask.bool()
            val_mask = val_mask.bool()

        #preprocessing and initialisation
        features, _ = process.preprocess_features(features)

        nb_nodes = features.shape[0]
        nb_classes = labels.shape[1]

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)

        ft_size = features.shape[1]

        features = torch.FloatTensor(features)
        original_features = features.unsqueeze(0).cuda()

        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
        labels = torch.FloatTensor(labels[np.newaxis])
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        ggd = GGD(ft_size, hid_units, nonlinearity)
        optimiser_disc = torch.optim.Adam(ggd.parameters(), lr=lr, weight_decay=l2_coef)
        if torch.cuda.is_available():
            ggd.cuda()
            features = features.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        features = features.unsqueeze(0)

        #generate a random number --> later use as a tag for saved model
        tag = str(int(np.random.random() * 10000000000))

        nb_feats = features.shape[2]

        avg_time = 0
        counts = 0

        for epoch in range(nb_epochs):
            ggd.train()
            optimiser_disc.zero_grad()

            aug_fts = aug_feature_dropout(features.squeeze(0)).unsqueeze(0) #augmentation on features
            idx = np.random.permutation(nb_nodes)
            shuf_fts = aug_fts[:, idx, :]  # shuffled embeddings / corruption

            aug_adj = sp.csr_matrix(adj)[idx, :]

            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                aug_fts = aug_fts.cuda()
                lbl = lbl.cuda()

            logits_1 = ggd(aug_fts,  shuf_fts, sp_adj, sparse=True)
            loss_disc = b_xent(logits_1, lbl)

            # print('Loss:', loss_disc)

            if loss_disc < best:
                best = loss_disc
                best_t = epoch
                cnt_wait = 0
                torch.save(ggd.state_dict(), 'pkl/best_dgi' + tag + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break
            loss_disc.backward()
            optimiser_disc.step()

        ggd.load_state_dict(torch.load('pkl/best_dgi' + tag + '.pkl'))

        or_embeds, pr_embeds = ggd.embed(original_features, sp_adj if sparse else adj, sparse)

        embeds = or_embeds + pr_embeds

        train_embs = embeds[0, idx_train]
        val_embs = embeds[0, idx_val]
        test_embs = embeds[0, idx_test]

        train_lbls = torch.argmax(labels[0, idx_train], dim=1)
        val_lbls = torch.argmax(labels[0, idx_val], dim=1)
        test_lbls = torch.argmax(labels[0, idx_test], dim=1)

        tot = torch.zeros(1)
        tot = tot.cuda()

        accs = []

        for _ in range(50):
            log = LogReg(train_embs.shape[1], nb_classes)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.cuda()

            pat_steps = 0
            best_acc = torch.zeros(1)
            best_acc = best_acc.cuda()
            for _ in range(args.classifier_epochs):
                log.train()
                opt.zero_grad()

                logits = log(train_embs)

                loss = xent(logits, train_lbls)

                loss.backward()
                opt.step()

            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc * 100)
            tot += acc

        # print('Average accuracy:', tot / 50)

        accs = torch.stack(accs)
        print(accs.mean())
        acc_results.append(accs.mean().cpu().numpy())

    print(np.mean(acc_results))

    with open('gslog_{}.txt'.format(args.dataset), 'a') as f:
        f.write(str(args))
        f.write('\n' + str(np.mean(acc_results)) + '\n')
        f.write(str(np.std(acc_results)) + '\n')
