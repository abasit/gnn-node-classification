# Graph Neural Networks for Node Classification
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

PyTorch Geometric implementations of GraphSAGE and GAT (Graph Attention Networks) for node classification on citation networks.


## Overview

This repository contains implementations of two popular Graph Neural Network architectures:
- **GraphSAGE** ([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)): Inductive representation learning on large graphs  
- **GAT** ([Veličković et al., 2018](https://arxiv.org/abs/1710.10903)): Graph Attention Networks with a multi-head attention mechanism

Both models are tested on the Cora citation network dataset, where the task is to classify academic papers into different categories based on their content and citation relationships.

## Results

Performance on Cora dataset (7 classes, 2708 nodes):

| Model     | Test Accuracy | Parameters                   | Training Time |
|-----------|---------------|------------------------------|---------------|
| GraphSAGE | 80.90%        | 2 layers, 64 hidden          | < 1 min       |
| GAT       | 82.80%        | 2 layers, 64 hidden, 8 heads | ~ 5 min       |

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/gnn-node-classification.git
cd gnn-node-classification

# Create a virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Below are example commands to train and evaluate the GNN models. Adjust paths or hyperparameter files as needed.

### 1. Train GraphSAGE on Cora

```bash
python src/train.py --config configs/graphsage_cora.yaml
```

### 2. Train GAT on Cora

```bash
python src/train.py --config configs/gat_cora.yaml
```

## Acknowledgments

Parts of this code were adapted from the Stanford CS224W “Machine Learning with Graphs” course materials:
> Leskovec, Jure (Instructor). *CS224W: Machine Learning with Graphs*. Stanford University.  
> URL: https://web.stanford.edu/class/cs224w/ (accessed June 2, 2025)
