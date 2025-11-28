"""
MAML-based Meta-Learner

Implements Model-Agnostic Meta-Learning (MAML) for rapid adaptation
to new tasks with limited data.

Reference:
    MAML: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
    GMSIFN: Dynamic Graph Meta-Learning with Multi-Sensor Spatial Dependencies for
            Cross-Category Small-Sample Fault Diagnosis in ZDJ9-RTAs
    Original file: MetaLearner.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


class MetaLearner(nn.Module):
    """
    MAML Meta-Learner for Small-Sample Learning

    Implements two-loop optimization:
    - Inner loop: Task-specific adaptation (5 gradient steps on support set)
    - Outer loop: Meta-parameter optimization (gradient descent on query set)

    Args:
        model: Base model to meta-learn (GMSIFN)

    Attributes:
        update_step (int): Number of inner loop gradient steps (default: 5)
        meta_lr (float): Meta-learning rate for outer loop (default: 0.001)
        base_lr (float): Base learning rate for inner loop (default: 0.001)
        lr_decay_rate (float): Learning rate decay (default: 0.95)
        lr_decay_interval (int): Decay interval in steps (default: 20)
    """

    def __init__(self, model):
        super(MetaLearner, self).__init__()

        self.update_step = 5  # Inner loop steps
        self.update_step_test = 5  # Test-time adaptation steps
        self.net = model

        # Learning rates
        self.meta_lr = 0.001  # Outer loop
        self.base_lr = 0.001  # Inner loop

        # Optimizer
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)

        # Learning rate decay
        self.global_step = 0
        self.lr_decay_interval = 20
        self.lr_decay_rate = 0.95

        # Training history
        self.history = {
            'train_acc': [],
            'finetune_acc': [],
            'test_acc': 0,
            'test_f1': 0,
            'test_auc': 0
        }

        # Store best meta-trained parameters
        self.best_meta_weights = None

    def get_current_lr(self):
        """Compute current learning rate with exponential decay"""
        decay_steps = self.global_step // self.lr_decay_interval
        return self.base_lr * (self.lr_decay_rate ** decay_steps)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        Meta-training forward pass

        Args:
            x_spt: Support set inputs
            y_spt: Support set labels
            x_qry: Query set inputs
            y_qry: Query set labels

        Returns:
            accs: Accuracy after each inner loop step
            loss: Loss after each inner loop step
            weights: Final adapted weights
        """
        self.global_step += 1
        task_num = x_spt.size(0)
        query_size = len(x_qry[0])

        # Initialize tracking
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]

        # === Inner Loop: Task Adaptation ===

        # Step 0: Compute initial gradient on support set
        y_hat = self.net(x_spt, params=list(self.net.parameters()))
        loss = F.cross_entropy(y_hat, y_spt.squeeze())
        grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True)

        # One-step gradient update
        fast_weights = list(map(
            lambda p: p[1] - self.base_lr * p[0],
            zip(grad, self.net.parameters())
        ))

        # Evaluate on query set (before adaptation)
        with torch.no_grad():
            y_hat = self.net(x_qry, list(self.net.parameters()))
            loss_qry = F.cross_entropy(y_hat, y_qry.squeeze())
            loss_list_qry[0] += loss_qry.item()
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
            correct_list[0] += torch.eq(pred_qry, y_qry).sum().item()

        # Evaluate on query set (after 1 step)
        with torch.no_grad():
            y_hat = self.net(x_qry, fast_weights)
            loss_qry = F.cross_entropy(y_hat, y_qry.squeeze())
            loss_list_qry[1] += loss_qry.item()
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
            correct_list[1] += torch.eq(pred_qry, y_qry).sum().item()

        # Continue inner loop updates (steps 2-5)
        for k in range(1, self.update_step):
            y_hat = self.net(x_spt, params=fast_weights)
            loss = F.cross_entropy(y_hat, y_spt.squeeze())
            grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)

            # Handle None gradients
            tuples = [(g if g is not None else 0, w) for g, w in zip(grad, fast_weights)]
            fast_weights = list(map(
                lambda p: p[1] - self.base_lr * p[0],
                tuples
            ))

            # Evaluate after each step
            if k < self.update_step - 1:
                with torch.no_grad():
                    y_hat = self.net(x_qry, params=fast_weights)
                    loss_qry = F.cross_entropy(y_hat, y_qry.squeeze())
                    loss_list_qry[k + 1] += loss_qry.item()
            else:
                # Last step: compute gradient for meta-update
                y_hat = self.net(x_qry, params=fast_weights)
                loss_qry = F.cross_entropy(y_hat, y_qry.squeeze())
                loss_list_qry[k + 1] += loss_qry

            with torch.no_grad():
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                correct_list[k + 1] += torch.eq(pred_qry, y_qry).sum().item()

        # === Outer Loop: Meta-Optimization ===
        weights = fast_weights
        loss_qry = loss_list_qry[-1] / task_num

        self.meta_optim.zero_grad()
        loss_qry.backward()
        self.meta_optim.step()

        # Compute accuracy across all steps
        accs = np.array(correct_list) / (query_size * task_num)

        # Save best meta-weights
        if self.best_meta_weights is None or accs[-1] > max(self.history['train_acc'], default=0):
            self.best_meta_weights = deepcopy(self.net.state_dict())

        # Convert loss to numpy
        loss = np.array([
            loss_item.cpu().item() if torch.is_tensor(loss_item) else loss_item
            for loss_item in loss_list_qry
        ]) / task_num

        return accs, loss, weights

    def meta_train(self, train_loader, epochs):
        """
        Complete meta-training loop

        Args:
            train_loader: DataLoader providing meta-training episodes
            epochs: Number of epochs to train
        """
        print("\n=== Starting Meta-Training ===")
        for epoch in range(epochs):
            accs_list = []
            for x_spt, y_spt, x_qry, y_qry in train_loader:
                x_spt, y_spt = x_spt.cuda(), y_spt.cuda()
                x_qry, y_qry = x_qry.cuda(), y_qry.cuda()

                accs, _, _ = self.forward(x_spt, y_spt, x_qry, y_qry)
                accs_list.append(accs[-1])

            avg_acc = np.mean(accs_list)
            self.history['train_acc'].append(avg_acc)
            print(f"Meta-Train Epoch {epoch + 1}/{epochs} | Acc: {avg_acc:.4f}")

    def finetune(self, finetune_loader, epochs=10):
        """
        Fine-tune on test support set

        Args:
            finetune_loader: DataLoader providing test support set
            epochs: Number of fine-tuning epochs (default: 10)
        """
        print("\n=== Starting Finetuning ===")

        # Load best meta-trained weights
        if self.best_meta_weights is not None:
            self.net.load_state_dict(self.best_meta_weights)

        finetune_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.base_lr)
        best_acc = 0.0

        for epoch in range(epochs):
            self.net.train()
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0.0

            for x_spt, y_spt, _, _ in finetune_loader:
                x_spt, y_spt = x_spt.cuda(), y_spt.cuda()

                finetune_optimizer.zero_grad()
                outputs = self.net(x_spt, params=list(self.net.parameters()))
                loss = F.cross_entropy(outputs, y_spt.squeeze())
                loss.backward()
                finetune_optimizer.step()

                # Track metrics
                with torch.no_grad():
                    _, preds = torch.max(outputs, 1)
                    correct = (preds == y_spt.squeeze()).sum().item()

                    epoch_correct += correct
                    epoch_total += y_spt.size(1)
                    epoch_loss += loss.item() * y_spt.size(0)

            # Epoch metrics
            epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            epoch_loss = epoch_loss / epoch_total if epoch_total > 0 else 0.0

            self.history['finetune_acc'].append(epoch_acc)

            # Save best fine-tuned model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.net.state_dict(), 'finetuned_model.pth')

            print(f"Finetune Epoch {epoch + 1}/{epochs} | "
                  f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    def evaluate(self, eval_loader):
        """
        Evaluate on test query set

        Args:
            eval_loader: DataLoader providing test query set

        Returns:
            test_acc: Test accuracy
            test_f1: Test F1-score
            test_auc: Test AUC-ROC
        """
        print("\n=== Final Evaluation ===")
        self.net.load_state_dict(torch.load('finetuned_model.pth'))
        self.net.eval()

        all_features, all_preds, all_labels, all_probs = [], [], [], []

        with torch.no_grad():
            for _, _, x_qry, y_qry in eval_loader:
                x_qry, y_qry = x_qry.cuda(), y_qry.cuda()

                # Get features and predictions
                features = self.net.get_features(x_qry, params=list(self.net.parameters()))
                outputs = self.net(x_qry, params=list(self.net.parameters()))

                all_features.append(features.cpu().numpy())
                all_preds.append(torch.argmax(outputs, 1).cpu())
                all_labels.append(y_qry.squeeze().cpu())
                all_probs.append(F.softmax(outputs, 1).cpu())

        # Aggregate results
        all_features = np.concatenate(all_features, axis=0)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # Compute metrics
        self.history['test_acc'] = (all_preds == all_labels).mean()
        all_labels = all_labels.squeeze()
        self.history['test_f1'] = f1_score(all_labels, all_preds, average='weighted')

        # Compute AUC
        unique_classes = np.unique(all_labels)
        n_classes = len(unique_classes)

        if n_classes > 1:
            try:
                if n_classes == 2:
                    label_mapping = {cls: i for i, cls in enumerate(unique_classes)}
                    remapped_labels = np.array([label_mapping[cls] for cls in all_labels])
                    pos_prob = all_probs[:, 1] if unique_classes[0] < unique_classes[1] else all_probs[:, 0]
                    self.history['test_auc'] = roc_auc_score(remapped_labels, pos_prob, average='macro')
                else:
                    label_mapping = {cls: i for i, cls in enumerate(unique_classes)}
                    remapped_labels = np.array([label_mapping[cls] for cls in all_labels])
                    proba_preds = np.zeros((len(all_labels), n_classes))
                    for i, cls in enumerate(unique_classes):
                        proba_preds[:, i] = all_probs[:, cls]
                    self.history['test_auc'] = roc_auc_score(
                        label_binarize(remapped_labels, classes=range(n_classes)),
                        proba_preds,
                        multi_class='ovr',
                        average='macro'
                    )
            except ValueError as e:
                print(f"Warning: AUC calculation failed - {str(e)}")
                self.history['test_auc'] = 0.0
        else:
            print("Warning: Only one class present, cannot compute AUC")
            self.history['test_auc'] = 0.0

        # Print results
        print(f"\nTest Accuracy: {self.history['test_acc']:.4f}")
        print(f"Test F1-score: {self.history['test_f1']:.4f}")
        print(f"Test AUC: {self.history['test_auc']:.4f}")

        return self.history['test_acc'], self.history['test_f1'], self.history['test_auc']

    def run_full_pipeline(self, train_loader, finetune_loader, eval_loader,
                          meta_epochs=10, finetune_epochs=5):
        """
        Complete meta-learning pipeline

        Args:
            train_loader: Meta-training episodes
            finetune_loader: Test support set for fine-tuning
            eval_loader: Test query set for evaluation
            meta_epochs: Number of meta-training epochs (default: 10)
            finetune_epochs: Number of fine-tuning epochs (default: 5)

        Returns:
            Evaluation metrics (acc, f1, auc)
        """
        # Stage 1: Meta-training
        self.meta_train(train_loader, meta_epochs)

        # Stage 2: Fine-tuning on novel classes
        self.finetune(finetune_loader, finetune_epochs)

        # Stage 3: Evaluation
        return self.evaluate(eval_loader)
