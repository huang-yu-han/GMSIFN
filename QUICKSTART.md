# GMSIFN Quick Start Guide

This guide will help you get started with GMSIFN in 5 minutes.

## Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

## Installation

```bash
# 1. Navigate to the GMSIFN directory
cd GMSIFN

# 2. Install dependencies
pip install -r requirements.txt
```

## Running Your First Experiment

### Step 1: Verify Data

Ensure your data files are in place:
```
RPM/
├── data.npy    # [num_samples, 5120, 9]
└── labels.npy  # [num_samples] with values 0-15
```

### Step 2: Quick Configuration Check

Open `config.py` and verify the data paths:
```python
DATA_CONFIG = {
    'data_path': r'your data',     # ← Check this path
    'label_path': r'your data',  # ← Check this path
    ...
}
```

### Step 3: Run Training

```bash
python train.py
```

That's it! The model will:
1. Meta-train on 11 base classes
2. Fine-tune on 5 novel classes
3. Evaluate and report Accuracy, F1-score, and AUC

## Expected Output

```
=============================================================
GMSIFN: Cross-Category Meta-Learning
=============================================================

Device: cuda:0
Random seed: 99010

-------------------------------------------------------------
Loading Data...
-------------------------------------------------------------
Base training classes (11 classes): [2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15]
Novel test classes (5 classes): [0, 1, 3, 5, 13]

[Stage 1/3] Meta-Training on Base Classes...
[Stage 2/3] Fine-Tuning on Novel Classes...
[Stage 3/3] Evaluating on Novel Classes...

🏆 Final Best Test Results:
  Accuracy: 0.8756
  F1-score: 0.8691
  AUC: 0.9234
```

## Customization Tips

### Change Novel Test Classes

In `config.py`:
```python
DATA_CONFIG = {
    'test_classes': [0, 1, 3, 5, 13],  # ← Change these class IDs
    ...
}
```

### Adjust Small-Sample Settings

```python
META_CONFIG = {
    'k_support': 5,          # ← Support set size (1-10)
    'k_query': 10,           # ← Query set size (5-20)
    'finetune_samples': 10,  # ← Samples for fine-tuning (5-20)
    ...
}
```

### Modify Model Architecture

```python
MODEL_CONFIG = {
    'radius': 2,             # ← GAT hops (1-3)
    'T': 2,                  # ← Attention iterations (1-3)
    'fingerprint_dim': 240,  # ← Hidden dimension (120-600)
    'top_k': 5,              # ← Neighbors in GGL (2-8)
    ...
}
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or model dimension:
```python
MODEL_CONFIG = {
    'fingerprint_dim': 120,  # Reduce from 240
    ...
}
```

### Poor Performance

Try increasing fine-tuning epochs:
```python
META_CONFIG = {
    'finetune_epochs': 20,  # Increase from 10
    ...
}
```

### Data Loading Errors

Check data shapes:
```python
import numpy as np
data = np.load('your data')
labels = np.load('your data')
print(f"Data shape: {data.shape}")      # Should be [N, 5120, 9]
print(f"Labels shape: {labels.shape}")  # Should be [N]
print(f"Classes: {np.unique(labels)}")  # Should be 0-15
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the model architecture in `models/gmsifn.py`
- Customize the meta-learning pipeline in `models/meta_learner.py`
- Modify the data loader in `data/meta_dataset.py`

## Getting Help
- Review the code comments for implementation details
- Contact: 17771470797@163.com

