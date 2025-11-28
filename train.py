"""
Training script for GMSIFN

This script implements the complete cross-category meta-learning pipeline:
1. Meta-training on base classes (11 classes)
2. Fine-tuning on novel classes (5 classes)
3. Evaluation on novel class query sets

Usage:
    python train.py

To modify hyperparameters, edit config.py
"""

import os
import torch
import random
import numpy as np
from config import MODEL_CONFIG, META_CONFIG, DATA_CONFIG, TRAIN_CONFIG
from models import GMSIFN, MetaLearner
from data import get_meta_loaders


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # === Setup ===
    print("=" * 60)
    print("GMSIFN: Cross-Category Meta-Learning")
    print("=" * 60)

    # Set device
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"\nDevice: {device}")

    # Set seed
    set_seed(DATA_CONFIG['seed'])
    print(f"Random seed: {DATA_CONFIG['seed']}")

    # Create checkpoint directory
    os.makedirs(TRAIN_CONFIG['save_dir'], exist_ok=True)

    # === Load Data ===
    print("\n" + "-" * 60)
    print("Loading Data...")
    print("-" * 60)

    train_loader, test_finetune_loader, test_eval_loader = get_meta_loaders(
        data_path=DATA_CONFIG['data_path'],
        label_path=DATA_CONFIG['label_path'],
        k_support=META_CONFIG['k_support'],
        k_query=META_CONFIG['k_query'],
        train_episodes=META_CONFIG['train_episodes'],
        test_episodes=META_CONFIG['test_episodes'],
        finetune_samples=META_CONFIG['finetune_samples'],
        seed=DATA_CONFIG['seed']
    )

    print(f"\nMeta-training episodes: {len(train_loader)}")
    print(f"Fine-tuning episodes: {len(test_finetune_loader)}")
    print(f"Evaluation episodes: {len(test_eval_loader)}")

    # Check data shapes
    finetune_support, finetune_labels, _, _ = next(iter(test_finetune_loader))
    print(f"\nFine-tuning support set shape: {finetune_support.shape}")
    print(f"Label distribution: {np.unique(finetune_labels[0].numpy(), return_counts=True)}")

    _, _, eval_query, eval_labels = next(iter(test_eval_loader))
    print(f"\nEvaluation query set shape: {eval_query.shape}")
    print(f"Label distribution: {np.unique(eval_labels[0].numpy(), return_counts=True)}")

    # === Initialize Model ===
    print("\n" + "-" * 60)
    print("Initializing Model...")
    print("-" * 60)

    model = GMSIFN(**MODEL_CONFIG).to(device)
    meta = MetaLearner(model).cuda()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # === Training Loop ===
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    best_test_results = {
        'epoch': -1,
        'acc': 0.0,
        'f1': 0.0,
        'auc': 0.0
    }

    for epoch in range(META_CONFIG['total_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{META_CONFIG['total_epochs']}")
        print(f"{'='*60}")

        # === Stage 1: Meta-Training ===
        print("\n[Stage 1/3] Meta-Training on Base Classes...")
        meta.meta_train(train_loader, epochs=META_CONFIG['meta_epochs'])

        # === Stage 2: Fine-Tuning ===
        print("\n[Stage 2/3] Fine-Tuning on Novel Classes...")
        meta.finetune(test_finetune_loader, epochs=META_CONFIG['finetune_epochs'])

        # === Stage 3: Evaluation ===
        print("\n[Stage 3/3] Evaluating on Novel Classes...")
        task_acc, task_f1, task_auc = meta.evaluate(test_eval_loader)

        # Track best results
        if task_acc > best_test_results['acc']:
            best_test_results = {
                'epoch': epoch,
                'acc': task_acc,
                'f1': task_f1,
                'auc': task_auc
            }
            print(f"\n{'*'*60}")
            print(f"🌟 New Best Test Results!")
            print(f"Accuracy: {task_acc:.4f}")
            print(f"F1-score: {task_f1:.4f}")
            print(f"AUC: {task_auc:.4f}")
            print(f"{'*'*60}")

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': meta.net.state_dict(),
                'meta_optim_state_dict': meta.meta_optim.state_dict(),
                'acc': task_acc,
                'f1': task_f1,
                'auc': task_auc
            }, os.path.join(TRAIN_CONFIG['save_dir'], 'best_model.pth'))
        else:
            print(f"\nCurrent Results:")
            print(f"Accuracy: {task_acc:.4f}")
            print(f"F1-score: {task_f1:.4f}")
            print(f"AUC: {task_auc:.4f}")

        # Print historical best
        print(f"\nHistorical Best (Epoch {best_test_results['epoch']}):")
        print(f"Accuracy: {best_test_results['acc']:.4f}")
        print(f"F1-score: {best_test_results['f1']:.4f}")
        print(f"AUC: {best_test_results['auc']:.4f}")

    # === Final Results ===
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\n🏆 Final Best Test Results (Epoch {best_test_results['epoch']}):")
    print(f"  Accuracy: {best_test_results['acc']:.4f}")
    print(f"  F1-score: {best_test_results['f1']:.4f}")
    print(f"  AUC: {best_test_results['auc']:.4f}")
    print(f"\nBest model saved to: {os.path.join(TRAIN_CONFIG['save_dir'], 'best_model.pth')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
