"""Evaluation and visualization for Driver Activity Recognition."""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)


def mean_per_class_accuracy(y_true, y_pred, num_classes):
    """Compute mean per-class accuracy (primary metric).

    For each class, computes the fraction of correctly classified samples,
    then averages across all classes (giving equal weight to each class).
    """
    per_class_acc = []
    for c in range(num_classes):
        mask = np.array(y_true) == c
        if mask.sum() > 0:
            acc = (np.array(y_pred)[mask] == c).mean()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    return np.mean(per_class_acc), per_class_acc


def compute_all_metrics(y_true, y_pred, idx_to_label):
    """Compute comprehensive evaluation metrics.

    Returns:
        dict with overall_acc, mean_per_class_acc, per_class_acc,
        macro_f1, weighted_f1, classification_report_str
    """
    num_classes = len(idx_to_label)
    overall_acc = accuracy_score(y_true, y_pred)
    mpca, per_class_acc = mean_per_class_accuracy(y_true, y_pred, num_classes)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    label_names = [idx_to_label[i] for i in range(num_classes)]
    report = classification_report(
        y_true, y_pred, labels=range(num_classes),
        target_names=label_names, zero_division=0
    )

    return {
        "overall_acc": overall_acc,
        "mean_per_class_acc": mpca,
        "per_class_acc": per_class_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }


def plot_confusion_matrix(y_true, y_pred, idx_to_label, save_path=None):
    """Plot normalized confusion matrix as a heatmap."""
    num_classes = len(idx_to_label)
    label_names = [idx_to_label[i] for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    # Normalize by row (true label)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(
        cm_norm, annot=False, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_path=None):
    """Plot training vs validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_per_class_metrics(per_class_acc, idx_to_label, save_path=None):
    """Plot per-class accuracy as a horizontal bar chart."""
    num_classes = len(idx_to_label)
    label_names = [idx_to_label[i] for i in range(num_classes)]

    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_names = [label_names[i] for i in sorted_indices]
    sorted_acc = [per_class_acc[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, max(8, num_classes * 0.3)))
    colors = plt.cm.RdYlGn(np.array(sorted_acc))
    ax.barh(range(num_classes), sorted_acc, color=colors)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_xlim(0, 1)
    ax.axvline(x=np.mean(per_class_acc), color="red", linestyle="--",
               label=f"Mean: {np.mean(per_class_acc):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def evaluate_model(model, test_loader, device, idx_to_label, config, logger=None):
    """Full evaluation pipeline: predictions -> metrics -> plots -> save.

    Args:
        model: trained ActivityLSTM model
        test_loader: DataLoader for test set
        device: torch device
        idx_to_label: dict mapping index -> activity name
        config: loaded config dict
        logger: optional logger instance
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Compute metrics
    metrics = compute_all_metrics(all_labels, all_preds, idx_to_label)

    log_fn = logger.info if logger else print
    log_fn(f"Overall Accuracy: {metrics['overall_acc']:.4f}")
    log_fn(f"Mean Per-Class Accuracy: {metrics['mean_per_class_acc']:.4f}")
    log_fn(f"Macro F1: {metrics['macro_f1']:.4f}")
    log_fn(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    log_fn(f"\nClassification Report:\n{metrics['classification_report']}")

    # Generate plots
    figure_dir = config["output"]["figure_dir"]
    os.makedirs(figure_dir, exist_ok=True)

    plot_confusion_matrix(
        all_labels, all_preds, idx_to_label,
        save_path=os.path.join(figure_dir, "confusion_matrix.png")
    )
    plot_per_class_metrics(
        metrics["per_class_acc"], idx_to_label,
        save_path=os.path.join(figure_dir, "per_class_accuracy.png")
    )

    log_fn(f"Plots saved to {figure_dir}")
    return metrics
