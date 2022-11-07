# Graph-Group-Discrimination

Code for NeurIPS 2022 paper **"Rethinking and Scaling Up Graph Contrastive Learning: An Extremely Efficient Approach with Group Discrimination"** https://arxiv.org/abs/2206.01535

![image](https://user-images.githubusercontent.com/75228223/191444300-b15ab48b-11c4-477d-b9bd-1a6b4cb931b8.png)

# Overview
Our implementation for Graph Group Discrimination (GGD) is based on PyTorch. There are two versions of our implementation including the manual version and the DGL-based version.

**Please run the following command to play the manual version of GGD for Cora dataset in the folder "Manual_version":**
```
python execute.py
```

**Please run the following command to play the demo in the folder "dgl_ogbn_arxiv_demo":**

#hidden 256

```
python3 train_arxiv_ready.py --dataset_name 'ogbn-arxiv' --dataset=ogbn-arxiv --dgi-lr 0.0001 --n-hidden 256 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-dgi-epochs 1
```

#hidden 1500

```
python3 train_arxiv_ready.py --dataset_name 'ogbn-arxiv' --dataset=ogbn-arxiv --dgi-lr 0.0001 --n-hidden 1500 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-dgi-epochs 1
```

**Please run the following command to play the demo in the folder "GGD_ogbn_product_1epoch":**

```
python3 train_product_to_release.py --dataset_name 'ogbn-products' --dataset=ogbn-products --n-classifier-epochs 3000 --self-loop --ggd-lr 0.0001 --n-hidden 1024 --n-layers 4 --proj_layers 4 --gnn_encoder 'gcn' --n-ggd-epochs 1
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
