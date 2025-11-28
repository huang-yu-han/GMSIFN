# GMSIFN: Graph-based Multi-Sensor Information Fusion Network

Official PyTorch implementation of **"Dynamic Graph Meta-Learning with Multi-Sensor Spatial Dependencies for Cross-Category Small-Sample Fault Diagnosis in ZDJ9-RTAs"**.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.2+](https://img.shields.io/badge/pytorch-1.2+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
GMSIFN adopts a three-stage processing architecture that effectively integrates multi-sensor fault information through the synergistic operation of: 
(1) unsupervised graph construction, (2) node feature aggregation, and (3) 
graph classiﬁcation. In the unsupervised graph construction stage, the GMSIFN network constructs sensor topology graphs in an unsupervised manner, relying solely on similarity computation between nodes without requiring any manual labeling.

### Key Features

- **Graph Generation Layer (GGL)**: Dynamically constructs graph structures based on learned node similarity
- **GAT with GRU**: Aggregates neighbor information through attention mechanism and GRU-based feature fusion
- **Graph-level Attention**: Captures global context through iterative attention mechanism
- **Cross-Category Meta-Learning**: Trains on base categories and generalizes to novel categories with small-sample learning
- **MAML Framework**: Enables rapid adaptation through gradient-based meta-learning

---

## Architecture

```
Input [B, 9, 5120]
    ↓
┌─────────────────────────────────────┐
│ 1. Input Projection Layer           │
│    [5120 → 240 dimensions]          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Graph Generation Layer (GGL)     │
│    - Compute node similarity        │
│    - Select top-k neighbors (k=5)   │
│    - Generate edge weights           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. GAT Aggregation                  │
│    ┌──────────────────────────┐    │
│    │  Iteration 1:            │    │
│    │  - Attention mechanism   │    │
│    │  - Neighbor aggregation  │    │
│    │  - GRU feature update    │    │
│    └──────────────────────────┘    │
│    ┌──────────────────────────┐    │
│    │  Iteration 2:            │    │
│    │  - Repeat with updated   │    │
│    │    node features         │    │
│    └──────────────────────────┘    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Graph-level Attention            │
│    (2 iterations)                   │
│    ┌──────────────────────────┐    │
│    │  - Attention mechanism   │    │
│    │  - GRU-based aggregation │    │
│    │  - Iterative refinement  │    │
│    └──────────────────────────┘    │
└─────────────────────────────────────┘
    ↓
Output [B, 16 classes]
```

---

## Installation

### Requirements

- Python >= 3.7
- PyTorch >= 1.2.0
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/GMSIFN.git
cd GMSIFN

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare Your Data

Place your data files in the `RPM/` directory:
```
RPM/
├── data.npy    # Shape: [num_samples, feature_dim, num_channels]
└── labels.npy  # Shape: [num_samples]
```

**Data Format**:
- `data.npy`: NumPy array with shape `[N, 5120, 9]` where N is the number of samples
- `labels.npy`: NumPy array with shape `[N]` containing class labels (0-15 for 16 classes)

### 2. Configure Training

Edit `config.py` to customize hyperparameters:

```python
# Model configuration
MODEL_CONFIG = {
    'radius': 2,                # Number of GAT iterations
    'T': 2,                     # Graph-level attention iterations
    'fingerprint_dim': 240,     # Hidden dimension
    'top_k': 5                  # Number of neighbors in GGL
}

# Meta-learning configuration
META_CONFIG = {
    'k_support': 5,             # Support set size per class
    'k_query': 10,              # Query set size per class
    'finetune_samples': 10,     # Samples for fine-tuning on novel classes
    'finetune_epochs': 10       # Fine-tuning epochs
}

# Data configuration
DATA_CONFIG = {
    'test_classes': [0, 1, 3, 5, 13],  # Novel classes for testing
    'seed': 99010                       # Random seed
}
```

### 3. Train the Model

```bash
python train.py
```

**Training Pipeline**:
1. **Meta-Training**: Learn initial parameters on base classes (11 classes)
2. **Fine-Tuning**: Adapt to novel classes (5 classes) with limited samples
3. **Evaluation**: Test on novel class query sets



---

## Code Structure

```
GMSIFN/
├── models/
│   ├── __init__.py
│   ├── gmsifn.py          # Main GMSIFN model
│   ├── ggl.py             # Graph Generation Layer
│   └── meta_learner.py    # MAML meta-learner
├── data/
│   ├── __init__.py
│   └── meta_dataset.py    # Meta-learning dataset & data loaders
├── utils/
│   └── __init__.py
├── checkpoints/           # Saved model checkpoints
├── config.py              # Configuration file
├── train.py               # Training script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Detailed Usage

### Model Components

#### 1. GMSIFN Model

```python
from models import GMSIFN

model = GMSIFN(
    radius=2,                # Number of GAT iterations
    T=2,                     # Graph-level attention iterations
    input_feature_dim=240,   # Node feature dimension
    input_bond_dim=1,        # Edge feature dimension (from GGL)
    fingerprint_dim=240,     # Hidden dimension
    output_units_num=16,     # Number of classes
    p_dropout=0.2,           # Dropout probability
    top_k=5                  # Number of neighbors in GGL
)

# Forward pass
output = model(input_data, params=list(model.parameters()))
```

#### 2. Graph Generation Layer (GGL)

```python
from models import GGL

ggl = GGL(top_k=5, dim=240)

# Generate graph structure
node_neighbor, bond_neighbor = ggl(node_features)
# node_neighbor: [B, N, K, feat_dim] - Neighbor node features
# bond_neighbor: [B, N, K, 1] - Edge weights
```

#### 3. Meta-Learner

```python
from models import MetaLearner

meta_learner = MetaLearner(model)

# Meta-training
meta_learner.meta_train(train_loader, epochs=1)

# Fine-tuning
meta_learner.finetune(finetune_loader, epochs=10)

# Evaluation
acc, f1, auc = meta_learner.evaluate(eval_loader)
```

### Data Loading

```python
from data import get_meta_loaders

train_loader, finetune_loader, eval_loader = get_meta_loaders(
    data_path='RPM/data.npy',
    label_path='RPM/labels.npy',
    k_support=5,
    k_query=10,
    train_episodes=60,
    test_episodes=10,
    finetune_samples=10,
    seed=42
)
```

---

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{gmsifn2026,
  title={Dynamic Graph Meta-Learning with Multi-Sensor Spatial Dependencies for Cross-Category Small-Sample Fault Diagnosis in ZDJ9-RTAs},
  author={Huang Y, Hu X, Chen F,et al},
  journal={ADVANCED ENGINEERING INFORMATICS},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work builds upon:
- **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
- **GAT**: Veličković et al., "Graph Attention Networks", ICLR 2018

---

## Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/yourusername/GMSIFN/issues)
- Contact: your.email@example.com

---

## Changelog

### Version 1.0.0 (2024)
- Initial release
- GMSIFN model implementation
- MAML-based meta-learning framework
- Cross-category small-sample learning support
- Comprehensive documentation and examples
