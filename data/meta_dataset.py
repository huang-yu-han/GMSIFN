"""
Meta-Learning Dataset for Cross-Category Small-Sample Learning

This module implements the data handling strategy for cross-category meta-learning:
1. CategorySplitter: Splits classes into base (training) and novel (testing) categories
2. MetaDataset: Generates small-sample episodes (N-way K-shot tasks)
3. get_meta_loaders: Convenience function for creating train/test data loaders

Reference:
    Paper: Dynamic Graph Meta-Learning with Multi-Sensor Spatial Dependencies for
           Cross-Category Small-Sample Fault Diagnosis in ZDJ9-RTAs
    Original file: duibishiyan_dataset.py
"""

import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from copy import deepcopy


class CategorySplitter:
    """
    Cross-Category Task Splitter

    Ensures strict separation between base (meta-training) and novel (meta-testing)
    categories to evaluate generalization to unseen classes.

    Args:
        all_classes: List of all class labels
        test_classes: List of classes reserved for testing
        seed: Random seed for reproducibility

    Example:
        >>> all_classes = list(range(16))  # 16 total classes
        >>> test_classes = [0, 1, 3, 5, 13]  # 5 test classes
        >>> splitter = CategorySplitter(all_classes, test_classes)
        >>> print(splitter.get_base_train_classes())  # 11 training classes
        [2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15]
    """

    def __init__(self, all_classes, test_classes, seed=42):
        self.all_classes = all_classes
        self.test_classes = test_classes
        self.seed = seed
        random.seed(seed)

        # Base training classes (excluding test classes)
        self.base_train_classes = [c for c in all_classes if c not in test_classes]

    def get_base_train_classes(self):
        """Get classes for meta-training"""
        return self.base_train_classes

    def get_test_classes(self):
        """Get classes for meta-testing"""
        return self.test_classes


class MetaDataset(Dataset):
    """
    Meta-Learning Dataset for Small-Sample Episodes

    Generates N-way K-shot episodes where each episode contains:
    - Support set: N classes × K samples per class (for adaptation)
    - Query set: N classes × Q samples per class (for evaluation)

    Args:
        full_data: Complete dataset [num_samples, num_channels, feature_dim]
        full_labels: Class labels [num_samples]
        base_classes: List of classes to sample from
        mode: Dataset mode ('train', 'test_finetune', 'test_eval')
        n_way: Number of classes per episode
        k_support: Number of support samples per class
        k_query: Number of query samples per class
        total_episodes: Total number of pre-generated episodes
        seed: Random seed
        use_all_remaining: Use all remaining samples as query set

    Example:
        >>> # 5-way 5-shot meta-learning
        >>> dataset = MetaDataset(
        ...     full_data, full_labels, base_classes=[0,1,2,3,4],
        ...     n_way=5, k_support=5, k_query=10
        ... )
        >>> support_x, support_y, query_x, query_y = dataset[0]
        >>> print(support_x.shape)  # [25, channels, features] (5 classes × 5 shots)
    """

    def __init__(self, full_data, full_labels, base_classes, mode='train',
                 n_way=4, k_support=3, k_query=3, total_episodes=5,
                 seed=42, use_all_remaining=False):

        self.mode = mode
        self.n_way = n_way
        self.k_support = k_support
        self.k_query = k_query
        self.total_episodes = total_episodes
        self.seed = seed
        self.base_classes = base_classes
        self.use_all_remaining = use_all_remaining

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Store data
        self.full_data = full_data
        self.full_labels = full_labels

        # Initialize sample management
        self._init_sample_pools()
        self.batchs_data = []
        self._pre_generate_batches()

    def _init_sample_pools(self):
        """Organize samples by class and initialize availability pools"""
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.full_labels):
            if label in self.base_classes:
                self.class_indices[label].append(idx)

        # Initialize available sample pools for each class
        self.available_samples = {
            cls: deepcopy(indices)
            for cls, indices in self.class_indices.items()
        }

    def _pre_generate_batches(self):
        """Pre-generate all episodes to ensure consistency across epochs"""
        for _ in range(self.total_episodes):
            support_x, support_y = [], []
            query_x, query_y = [], []

            # Sample N classes for this episode
            selected_classes = random.sample(self.base_classes, self.n_way)

            for cls in selected_classes:
                # Get available samples for this class
                available = deepcopy(self.available_samples[cls])
                random.shuffle(available)

                # Refill pool if insufficient samples
                if len(available) < self.k_support:
                    self.available_samples[cls] = deepcopy(self.class_indices[cls])
                    available = deepcopy(self.available_samples[cls])
                    random.shuffle(available)

                # Sample support set
                spt_samples = available[:self.k_support]

                # Sample query set
                if self.use_all_remaining:
                    # Use all remaining samples
                    qry_samples = available[self.k_support:]
                else:
                    # Use fixed number of query samples
                    total_needed = self.k_support + self.k_query
                    if len(available) < total_needed:
                        self.available_samples[cls] = deepcopy(self.class_indices[cls])
                        available = deepcopy(self.available_samples[cls])
                        random.shuffle(available)
                    qry_samples = available[self.k_support:self.k_support + self.k_query]

                # Remove used samples from pool (prevent reuse within epoch)
                for sample in spt_samples + qry_samples:
                    if sample in self.available_samples[cls]:
                        self.available_samples[cls].remove(sample)

                # Collect samples
                support_x.append(self.full_data[spt_samples])
                support_y.extend([cls] * len(spt_samples))
                query_x.append(self.full_data[qry_samples])
                query_y.extend([cls] * len(qry_samples))

            # Stack into tensors
            support_x = np.concatenate(support_x, axis=0) if support_x else np.array([])
            query_x = np.concatenate(query_x, axis=0) if query_x else np.array([])

            self.batchs_data.append((
                torch.FloatTensor(support_x),
                torch.LongTensor(np.array(support_y)),
                torch.FloatTensor(query_x),
                torch.LongTensor(np.array(query_y))
            ))

    def __getitem__(self, index):
        """Get episode at index"""
        return self.batchs_data[index]

    def __len__(self):
        """Number of episodes"""
        return len(self.batchs_data)


def get_meta_loaders(data_path, label_path, k_support=5, k_query=10,
                     train_episodes=100, test_episodes=20, finetune_samples=5,
                     seed=42):
    """
    Create meta-learning data loaders with cross-category splitting

    This function handles the complete data pipeline:
    1. Load and normalize data
    2. Split classes into base (training) and novel (testing)
    3. Create meta-training episodes from base classes
    4. Create fine-tuning and evaluation sets from novel classes

    Args:
        data_path: Path to data.npy file [num_samples, feature_dim, num_channels]
        label_path: Path to labels.npy file [num_samples]
        k_support: Support set size per class (default: 5)
        k_query: Query set size per class (default: 10)
        train_episodes: Number of meta-training episodes (default: 100)
        test_episodes: Number of meta-testing episodes (default: 20)
        finetune_samples: Samples per class for fine-tuning on novel classes (default: 5)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        train_loader: DataLoader for meta-training (base classes)
        test_finetune_loader: DataLoader for fine-tuning (novel classes, support set)
        test_eval_loader: DataLoader for evaluation (novel classes, query set)

    Example:
        >>> train_loader, finetune_loader, eval_loader = get_meta_loaders(
        ...     'data/data.npy', 'data/labels.npy',
        ...     k_support=5, k_query=10
        ... )
        >>> # Meta-training
        >>> for spt_x, spt_y, qry_x, qry_y in train_loader:
        ...     # Train on base classes
        ...     pass
        >>> # Fine-tuning and evaluation on novel classes
        >>> for spt_x, spt_y, _, _ in finetune_loader:
        ...     # Fine-tune on novel classes
        ...     pass
    """

    # === Stage 1: Load Data ===
    full_data = np.load(data_path).astype(np.float32)
    full_labels = np.load(label_path).astype(np.int64)

    # Transpose to [num_samples, num_channels, feature_dim]
    full_data = np.transpose(full_data, (0, 2, 1))

    # === Stage 2: Category Splitting ===
    all_classes = np.unique(full_labels)
    assert len(all_classes) == 16, "Dataset should contain 16 classes"

    # Fixed test classes (5 novel categories)
    test_classes = [0, 1, 3, 5, 13]
    splitter = CategorySplitter(all_classes, test_classes, seed=seed)

    print(f"Base training classes (11 classes): {sorted(splitter.base_train_classes)}")
    print(f"Novel test classes (5 classes): {sorted(test_classes)}")

    # === Stage 3: Data Normalization ===
    # Compute statistics from base training classes only
    base_train_classes = splitter.get_base_train_classes()
    train_mask = np.isin(full_labels, base_train_classes)
    train_data_for_norm = full_data[train_mask]

    mean = np.mean(train_data_for_norm, axis=(0, 1), keepdims=True)
    std = np.std(train_data_for_norm, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    # Apply normalization to all data
    full_data = (full_data - mean) / std

    # === Stage 4: Create Meta-Training Dataset ===
    train_dataset = MetaDataset(
        full_data,
        full_labels,
        base_classes=base_train_classes,
        mode='train',
        n_way=5,
        k_support=k_support,
        k_query=k_query,
        total_episodes=train_episodes,
        seed=seed,
        use_all_remaining=False
    )

    # === Stage 5: Split Test Data ===
    test_class_indices = defaultdict(list)
    for idx, label in enumerate(full_labels):
        if label in test_classes:
            test_class_indices[label].append(idx)

    # Separate fine-tuning and evaluation samples
    test_finetune_indices = []
    test_eval_indices = []

    for cls in test_classes:
        indices = test_class_indices[cls]
        random.shuffle(indices)

        assert len(indices) > finetune_samples, \
            f"Class {cls} has insufficient samples ({len(indices)} < {finetune_samples})"

        test_finetune_indices.extend(indices[:finetune_samples])
        test_eval_indices.extend(indices[finetune_samples:])

    # Extract subsets
    test_finetune_data = full_data[test_finetune_indices]
    test_finetune_labels = full_labels[test_finetune_indices]
    test_eval_data = full_data[test_eval_indices]
    test_eval_labels = full_labels[test_eval_indices]

    # === Stage 6: Create Test Datasets ===
    # Fine-tuning dataset (support set for novel classes)
    test_finetune_dataset = MetaDataset(
        test_finetune_data,
        test_finetune_labels,
        base_classes=test_classes,
        mode='test_finetune',
        n_way=5,
        k_support=finetune_samples,
        k_query=0,
        total_episodes=test_episodes,
        seed=seed + 2,
        use_all_remaining=False
    )

    # Evaluation dataset (query set for novel classes)
    test_eval_dataset = MetaDataset(
        test_eval_data,
        test_eval_labels,
        base_classes=test_classes,
        mode='test_eval',
        n_way=5,
        k_support=0,
        k_query=0,
        total_episodes=test_episodes,
        seed=seed + 3,
        use_all_remaining=True
    )

    # === Stage 7: Create DataLoaders ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    test_finetune_loader = DataLoader(
        test_finetune_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    test_eval_loader = DataLoader(
        test_eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_finetune_loader, test_eval_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_finetune_loader, test_eval_loader = get_meta_loaders(
        data_path='your data',
        label_path='your data',
        k_support=5,
        k_query=10,
        train_episodes=60,
        test_episodes=10,
        finetune_samples=10,
        seed=42
    )

    print(f"\nTrain episodes: {len(train_loader)}")
    print(f"Test finetune episodes: {len(test_finetune_loader)}")
    print(f"Test eval episodes: {len(test_eval_loader)}")

    # Check data shapes
    spt_x, spt_y, qry_x, qry_y = next(iter(train_loader))
    print(f"\nTrain episode sample:")
    print(f"Support set: {spt_x.shape}, labels: {np.unique(spt_y[0].numpy())}")
    print(f"Query set: {qry_x.shape}, labels: {np.unique(qry_y[0].numpy())}")
