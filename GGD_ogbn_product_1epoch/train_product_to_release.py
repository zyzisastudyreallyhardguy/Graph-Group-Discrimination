import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from typing import List
import dgl
from dgl import DGLGraph
import dgl.function as fn
from dgl.data import register_data_args, load_data
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from torch.utils.data.dataloader import DataLoader
from dgl.nn import EdgeWeightNorm
from tqdm import tqdm
import random
import copy
from ggd_sp import GGD, Classifier
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import os
from sklearn import preprocessing as sk_prep

class NodeSet(Dataset):
    def __init__(self, node_list: List[int], labels):
        super(NodeSet, self).__init__()
        self.node_list = node_list
        self.labels = labels
        assert len(self.node_list) == len(self.labels)

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        return self.node_list[idx], self.labels[idx]

class NbrSampleCollater(object):
    def __init__(self, graph: dgl.DGLHeteroGraph,
                 block_sampler: dgl.dataloading.BlockSampler):
        self.graph = graph
        self.block_sampler = block_sampler

    def collate(self, batch):
        batch = torch.tensor(batch)
        nodes = batch[:, 0]
        labels = batch[:, 1]
        blocks = self.block_sampler.sample_blocks(self.graph, nodes)
        return blocks, labels

def aug_feature_dropout(input_feat, drop_percent=0.2):
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
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def load_data_ogb(dataset, args):
    global n_node_feats, n_classes

    if args.data_root_dir == 'default':
        data = DglNodePropPredDataset(name=dataset)
    else:
        data = DglNodePropPredDataset(name=dataset, root=args.data_root_dir)

    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]


    # Replace node features here
    if args.pretrain_path != 'None':
        graph.ndata["feat"] = torch.tensor(np.load(args.pretrain_path)).float()
        print("Pretrained node feature loaded! Path: {}".format(args.pretrain_path))

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator

def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph

def main(args):
    cuda = True
    free_gpu_id = int(get_free_gpu())
    torch.cuda.set_device(free_gpu_id)
    # load and preprocess dataset
    if 'ogbn' not in args.dataset_name:
        data = load_data(args)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        if hasattr(torch, 'BoolTensor'):
            train_mask = torch.BoolTensor(data.train_mask)
            val_mask = torch.BoolTensor(data.val_mask)
            test_mask = torch.BoolTensor(data.test_mask)
        else:
            train_mask = torch.ByteTensor(data.train_mask)
            val_mask = torch.ByteTensor(data.val_mask)
            test_mask = torch.ByteTensor(data.test_mask)
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
        g = data.graph
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        if args.self_loop:
            g.remove_edges_from(nx.selfloop_edges(g))
            g.add_edges_from(zip(g.nodes(), g.nodes()))
        g = DGLGraph(g)
    else:
        g, all_labels, train_mask, val_mask, test_mask, evaluator = load_data_ogb(args.dataset_name, args)
        g = preprocess(g)

        features = g.ndata['feat']
        all_labels = all_labels.T.squeeze(0)

        all_labels, train_idx, val_idx, test_idx, features = map(
            lambda x: x.to(free_gpu_id), (all_labels, train_mask, val_mask, test_mask, features)
        )

        in_feats = g.ndata['feat'].shape[1]
        n_classes = all_labels.T.max().item() + 1
        n_edges = g.num_edges()

    fanouts_train = [12,12,12]
    fanouts_test = [12,12,12]

    train_collater = NbrSampleCollater(
        g, MultiLayerNeighborSampler(fanouts=fanouts_train, replace=False))
    train_node_set = NodeSet(torch.LongTensor(np.arange(g.num_nodes())).tolist(), all_labels.tolist())
    train_node_loader = DataLoader(dataset=train_node_set, batch_size=2048,
                                        shuffle=True, num_workers=0, pin_memory=True,
                                        collate_fn=train_collater.collate, drop_last=False)

    # create DGI model
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

    ggd_optimizer = torch.optim.AdamW(ggd.parameters(),
                                     lr=args.ggd_lr,
                                     weight_decay=args.weight_decay)

    b_xent = nn.BCEWithLogitsLoss()

    # train graph group discrimination
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []

    tag = str(int(np.random.random() * 10000000000)) #generate a unique tag

    for epoch in range(args.n_ggd_epochs):
        t0 = time.time()
        ggd.train()
        if epoch >= 3:
            t0 = time.time()

        loss = 0
        for n_iter, (blocks, labels) in enumerate(tqdm(train_node_loader, desc=f'train epoch {epoch}')):
            blocks = [block.to(free_gpu_id) for block in blocks]
            labels = labels.to(free_gpu_id)
            loss = ggd(blocks, labels, b_xent)

            ggd_optimizer.zero_grad()
            loss.backward()
            ggd_optimizer.step()

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

    # # train classifier
    print('Loading {}th epoch'.format(best_t))
    ggd.load_state_dict(torch.load('pkl/best_ggd' + tag + '.pkl'))

    # get all embeddings for evaluation
    ggd.eval()
    embeds = []

    test_collater = NbrSampleCollater(
        g, MultiLayerNeighborSampler(fanouts=fanouts_test, replace=False))
    test_node_set = NodeSet(torch.LongTensor(np.arange(g.num_nodes())).tolist(), all_labels.tolist())
    test_node_loader = DataLoader(dataset=test_node_set, batch_size=4196,
                                        shuffle=False, num_workers=0, pin_memory=True,
                                        collate_fn=test_collater.collate, drop_last=False)

    for n_iter, (blocks, labels) in enumerate(tqdm(test_node_loader, desc=f'loading embedding for evaluation')):
        blocks = [block.to(free_gpu_id) for block in blocks]
        labels = labels.to(free_gpu_id)
        embed = ggd.embed(blocks)
        embeds.append(embed.cpu())

    l_embeds = torch.cat(embeds, dim = 0)

    torch.cuda.empty_cache()

    '''obtain embedding for downstream classifier training'''

    print('Start Testing. Please wait...')
    g_embeds = graph_power(l_embeds, g)
    embeds = l_embeds + g_embeds
    embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")
    embeds = torch.FloatTensor(embeds).cuda()

    # create classifier model
    classifier = Classifier(args.n_hidden, n_classes)
    if cuda:
        classifier.cuda()

    classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)

    all_labels = all_labels.cuda()
    dur = []
    best_acc = 0
    patience = 100
    wait = 0
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[train_mask], all_labels[train_mask])
        loss.backward()
        classifier_optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        acc = evaluate(classifier, embeds, all_labels, val_mask)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                     acc, n_edges / np.mean(dur) / 1000))
        if acc > best_acc:
            best_acc = acc
            wait = 0
        wait += 1
        if wait > patience:
            break

    acc = evaluate(classifier, embeds, all_labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))

    return acc

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def graph_power(embed, g):
    feat = embed.squeeze(0)

    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5)
    norm = norm.to(feat.device).unsqueeze(1)
    for _ in range(10):
        feat = feat * norm
        g.ndata['h2'] = feat
        g.update_all(fn.copy_u('h2', 'm'),
                     fn.sum('m', 'h2'))
        feat = g.ndata.pop('h2')
        feat = feat * norm

    return feat


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='DGI')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--ggd-lr", type=float, default=0.001,
                        help="ggd learning rate")
    parser.add_argument("--drop_feat", type=float, default=0.2,
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
                        help='Dataset name: cora, citeseer, pubmed, cs, phy')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    accs = []
    for i in range(args.n_trails):
        accs.append(main(args))
    mean_acc = str(np.array(accs).mean())
    print('mean accuracy:' + mean_acc)

    file_name = str(args.dataset_name)
    f = open('result/' + 'result_' + file_name + '.txt', 'a')
    f.write(str(args) + '\n')
    f.write(mean_acc + '\n')

