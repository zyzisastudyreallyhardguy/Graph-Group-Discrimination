import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import load, load_coauthor
from torch_geometric.seed import seed_everything
import dgl
from dgl.data import register_data_args, load_data
import random
import copy
from ggd import GGD, Classifier
import os

def aug_feature_dropout(input_feat, drop_percent=0.2):
    # aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels.argmax(1))
        return correct.item() * 1.0 / len(labels.argmax(1))

def main(args):
    # load and preprocess dataset
    # data = load_data(args)
    cuda = True
    free_gpu_id = int(get_free_gpu())
    torch.cuda.set_device(free_gpu_id)
    adj, features, labels, idx_train, idx_val, idx_test = load_coauthor(args.dataset_name)

    train_val_ratio = 0.2

    idx_train_val = random.sample(list(np.arange(features.shape[0])), int(train_val_ratio * features.shape[0]))
    remain_num = len(idx_train_val)
    idx_train = idx_train_val[remain_num//2:]
    idx_val = idx_train_val[:remain_num//2]
    idx_test = list(set(np.arange(features.shape[0])) - set(idx_train_val))

    src, dst = np.nonzero(adj)
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.FloatTensor(features)
    g.ndata['label'] = torch.LongTensor(labels)

    mask = ['train_mask', 'test_mask', 'val_mask']
    for i, idx in enumerate([idx_train, idx_test, idx_val]):
        temp_mask = torch.zeros(g.num_nodes())
        temp_mask[idx] = 1
        g.ndata[mask[i]] = temp_mask.bool()

    g, labels, train_idx, val_idx, test_idx, features = map(
        lambda x: x.to(free_gpu_id), (g, g.ndata['label'], g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'], g.ndata['feat'])
    )

    in_feats = g.ndata['feat'].shape[1]
    n_classes = labels.shape[1]
    n_edges = g.num_edges()

    g = g.to(free_gpu_id)
    # create GGD model
    ggd = GGD(g,
              in_feats,
              args.n_hidden,
              args.n_layers,
              nn.PReLU(args.n_hidden),
              args.dropout,
              args.proj_layers,
              args.gnn_encoder,
              args.num_hop)

    if cuda:
        ggd.cuda()

    ggd_optimizer = torch.optim.Adam(ggd.parameters(),
                                     lr=args.ggd_lr,
                                     weight_decay=args.weight_decay)

    b_xent = nn.BCEWithLogitsLoss()

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    counts = 0
    avg_time = 0
    dur = []

    tag = str(int(np.random.random() * 10000000000))

    for epoch in range(args.n_ggd_epochs):
        t0 = time.time()
        ggd.train()
        if epoch >= 3:
            t0 = time.time()

        ggd_optimizer.zero_grad()

        lbl_1 = torch.ones(1, g.num_nodes())
        lbl_2 = torch.zeros(1, g.num_nodes())
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        # import pdb; pdb.set_trace()
        aug_feat = aug_feature_dropout(features, 0.2)
        loss = ggd(aug_feat.cuda(), lbl, b_xent)
        loss.backward()
        ggd_optimizer.step()

        comp_time = time.time() - t0
        # print('{} seconds'.format(comp_time))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(ggd.state_dict(), 'pkl/best_ggd' + tag + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            n_edges / np.mean(dur) / 1000))

        avg_time += comp_time
        counts += 1

    # create classifier model
    classifier = Classifier(args.n_hidden, n_classes)
    if cuda:
        classifier.cuda()

    classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)

    # train classifier
    print('Loading {}th epoch'.format(best_t))
    ggd.load_state_dict(torch.load('pkl/best_ggd' + tag + '.pkl'))

    l_embeds, g_embeds= ggd.embed(features, g)

    embeds = (l_embeds + g_embeds).squeeze(0)

    dur = []
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[g.ndata['train_mask']], labels[g.ndata['train_mask']].argmax(1))
        loss.backward()
        classifier_optimizer.step()
        
        if epoch >= 3:
            dur.append(time.time() - t0)

    acc = evaluate(classifier, embeds, labels, g.ndata['test_mask'])
    print("Test Accuracy {:.4f}".format(acc))

    return acc

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='GGD')
    register_data_args(parser)
    parser.add_argument("--seed", type=int, default=0.,
                        help="seed")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--ggd-lr", type=float, default=0.001,
                        help="ggd learning rate")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    parser.add_argument("--classifier-lr", type=float, default=0.05,
                        help="classifier learning rate")
    parser.add_argument("--n-ggd-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--proj_layers", type=int, default=1,
                        help="number of project linear layers")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=500,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--n_trails", type=int, default=5,
                        help="number of trails")
    parser.add_argument("--gnn_encoder", type=str, default='gcn',
                        help="choice of gnn encoder")
    parser.add_argument("--num_hop", type=int, default=10,
                        help="number of k for sgc")
    parser.add_argument('--data_root_dir', type=str, default='default',
                           help="dir_path for saving graph data. Note that this model use DGL loader so do not mix up with the dir_path for the Pyg one. Use 'default' to save datasets at current folder.")
    parser.add_argument("--pretrain_path", type=str, default='None',
                        help="path for pretrained node features")
    parser.add_argument('--dataset_name', type=str, default='cora',
                        help='Dataset name: cora, citeseer, pubmed, cs, phy, computer, photo')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    seed = False

    if seed:
        seed = args.seed
        seed_everything(seed)
        print('seed_number:' + str(seed))

    accs = []
    for i in range(args.n_trails):
        accs.append(main(args))
    print('mean accuracy:' + str(np.array(accs).mean()))

    with open('gslog_{}.txt'.format(args.dataset_name), 'a') as f:
        f.write(str(args))
        f.write('\n' + str(np.mean(accs)) + '\n')
        f.write(str(np.std(accs)) + '\n')
