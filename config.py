"""
Configuration file for GMSIFN training

Modify these parameters to customize training behavior.
"""

# ===== Model Configuration =====
MODEL_CONFIG = {
    'radius': 2,  # Number of iterations for GAT aggregation
    'T': 2,  # Number of graph-level attention iterations
    'input_feature_dim': 240,  # Node feature dimension
    'input_bond_dim': 1,  # Edge feature dimension (from GGL)
    'fingerprint_dim': 240,  # Hidden dimension
    'output_units_num': 16,  # Number of output classes
    'p_dropout': 0.2,  # Dropout probability
    'top_k': 5  # Number of neighbors in GGL
}

# ===== Meta-Learning Configuration =====
META_CONFIG = {
    'k_support': 5,  # Support set size per class
    'k_query': 10,  # Query set size per class
    'train_episodes': 60,  # Number of meta-training episodes
    'test_episodes': 10,  # Number of meta-testing episodes
    'finetune_samples': 10,  # Samples per class for fine-tuning on novel classes
    'meta_epochs': 1,  # Epochs per meta-training iteration
    'finetune_epochs': 10,  # Fine-tuning epochs on novel classes
    'total_epochs': 1  # Total number of meta-training epochs
}

# ===== Data Configuration =====
DATA_CONFIG = {
    'data_path': r'your data',  # Path to data file
    'label_path': r'your data',  # Path to label file
    'test_classes': [0, 1, 3, 5, 13],  # Novel classes for meta-testing
    'seed': 99010  # Random seed for reproducibility
}

# ===== Training Configuration =====
TRAIN_CONFIG = {
    'device': 'cuda:0',  # Training device
    'save_dir': 'checkpoints',  # Directory to save checkpoints
    'log_interval': 5  # Print interval for training logs
}
