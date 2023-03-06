# Graph-Group-Discrimination

Code for NeurIPS 2022 paper **"Rethinking and Scaling Up Graph Contrastive Learning: An Extremely Efficient Approach with Group Discrimination"** https://arxiv.org/abs/2206.01535

![image](https://user-images.githubusercontent.com/75228223/191444300-b15ab48b-11c4-477d-b9bd-1a6b4cb931b8.png)

# Overview
Our implementation for Graph Group Discrimination (GGD) is based on PyTorch.

**Requirement**
```
dgl                     0.7.1
networkx                2.6.2
numpy                   1.22.3
ogb                     1.3.2
scikit-learn            0.24.2
torch                   1.9.0
torch-cluster           1.5.9
torch-geometric         2.0.4
torch-scatter           2.0.8
torch-sparse            0.6.12
torch-spline-conv       1.2.1
torchaudio              0.9.0a0+33b2469
torchmetrics            0.5.1
torchvision             0.10.0
```

**Please run the following command to play the demo in the folder "GGD_ogbn_arxiv_1epoch":**

#hidden 256

```
python3 train_arxiv_ready.py --dataset_name 'ogbn-arxiv' --dataset=ogbn-arxiv --ggd-lr 0.0001 --n-hidden 256 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-ggd-epochs 1
```

#hidden 1500

```
python3 train_arxiv_ready.py --dataset_name 'ogbn-arxiv' --dataset=ogbn-arxiv --ggd-lr 0.0001 --n-hidden 1500 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-ggd-epochs 1
```

**Please run the following command to play the demo in the folder "GGD_ogbn_product_1epoch":**

```
python3 train_product_to_release.py --dataset_name 'ogbn-products' --dataset=ogbn-products --n-classifier-epochs 3000 --self-loop --ggd-lr 0.0001 --n-hidden 1024 --n-layers 4 --proj_layers 4 --gnn_encoder 'gcn' --n-ggd-epochs 1
```

**Please run the following command to run GGD for Cora dataset in the folder "GGD-citation":**
```
python execute.py
```

**Please run the following command to run GGD for Amazon/Coauthor datasets in the folder "GGD-amco":**
To download these datasets, please use this link "https://github.com/shchur/gnn-benchmark/tree/master/data/npz". The downloaded files should be put under the "GGD-amco/data" folder.

**For Amazon Photo**
```
python train_coauthor.py --n-classifier-epochs 2000 --n-hidden 512 --n-ggd-epochs 2000 --ggd-lr 0.0005 --proj_layers 1 --dataset_name 'photo'
```
**For Amazon Computer**
```
-n-classifier-epochs 3500 --n-hidden 1024 --n-ggd-epochs 1500 --ggd-lr 0.0001 --proj_layers 1 --dataset_name 'computer'
```

# Reference

```
@inproceeding{zheng2022rethinking,
  title={Rethinking and Scaling Up Graph Contrastive Learning: An Extremely Efficient Approach with Group Discrimination},
  author={Zheng, Yizhen and Pan, Shirui and Lee, Vincent Cs and Zheng, Yu and Yu, Philip S},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

#License

```
MIT
```
