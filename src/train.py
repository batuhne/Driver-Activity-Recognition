"""Training script for ActivityLSTM on pre-extracted features.

Usage: python src/train.py --config configs/config.yaml
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import load_config, set_seed, setup_logging, compute_effective_number_weights
from src.dataset import parse_annotations, get_dataloaders, DriveActFeatureDataset, mixup_batch
from src.models import ActivityLSTM
from src.evaluate import evaluate_model, compute_all_metrics


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    FL = -(1 - p_t)^gamma * log(p_t)
    Supports both hard labels and soft targets (from mixup).
    """

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes) raw predictions
            targets: (batch,) integer labels OR (batch, num_classes) soft targets
        """
        if targets.dim() == 2:
            # Soft targets (e.g. from mixup) — focal weighting is not
            # semantically meaningful, fall back to plain soft CE
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(targets * log_probs).sum(dim=1).mean()
            return loss

        # Hard labels — apply focal loss with optional label smoothing
        num_classes = logits.size(1)
        one_hot = F.one_hot(targets, num_classes).float()
        if self.label_smoothing > 0:
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Focal weight: (1 - p_t)^gamma for each class
        focal_weight = (1 - probs) ** self.gamma

        # Weighted cross-entropy
        loss = -(focal_weight * one_hot * log_probs).sum(dim=1).mean()
        return loss


def train(config):
    """Full training pipeline: train LSTM, validate, test, save best model."""
    logger = setup_logging(config["output"]["log_dir"], "train")
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Parse annotations to get num_classes
    splits, label_to_idx, idx_to_label = parse_annotations(config)
    num_classes = len(label_to_idx)
    logger.info(f"Number of classes: {num_classes}")

    # Create dataloaders from pre-extracted features
    loaders = get_dataloaders(config, feature_based=True, num_classes=num_classes)
    logger.info(f"Train: {len(loaders['train'].dataset)} samples")
    logger.info(f"Val: {len(loaders['val'].dataset)} samples")
    logger.info(f"Test: {len(loaders['test'].dataset)} samples")

    # Class weights for loss (when not using weighted sampler)
    train_labels = loaders.pop("train_labels", None)
    train_cfg = config["training"]
    use_sampler = train_cfg.get("use_weighted_sampler", True)
    class_weights = None
    if not use_sampler and train_labels is not None:
        beta = train_cfg.get("en_beta", 0.999)
        class_weights = compute_effective_number_weights(train_labels, num_classes, beta=beta).to(device)
        raw_ratio = class_weights.max() / class_weights.min()
        # Sqrt dampening: reduces extreme ratios (e.g. 834x → ~29x)
        # Preserves relative ordering but prevents common classes from being ignored
        class_weights = torch.sqrt(class_weights)
        class_weights = class_weights / class_weights.mean()  # Re-normalize to mean=1
        logger.info(f"Class-weighted loss: EN beta={beta}, sqrt dampened, "
                    f"weight range=[{class_weights.min():.4f}, {class_weights.max():.4f}], "
                    f"ratio={class_weights.max()/class_weights.min():.1f}x (raw: {raw_ratio:.1f}x)")
    elif use_sampler:
        logger.info("Using WeightedRandomSampler for class balancing")

    # Model — use .get() for backward compatibility with old configs
    model_cfg = config["model"]
    model = ActivityLSTM(
        input_dim=model_cfg["feature_dim"],
        hidden_dim=model_cfg["lstm_hidden"],
        num_layers=model_cfg["lstm_layers"],
        num_classes=num_classes,
        lstm_dropout=model_cfg["lstm_dropout"],
        fc_dropout=model_cfg["fc_dropout"],
        use_layernorm=model_cfg.get("use_layernorm", False),
        bidirectional=model_cfg.get("bidirectional", False),
        pooling=model_cfg.get("pooling", "last"),
        noise_std=config["training"].get("noise_std", 0.0),
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"BiLSTM: {model_cfg.get('bidirectional', False)}, "
                f"Pooling: {model_cfg.get('pooling', 'last')}, "
                f"LayerNorm: {model_cfg.get('use_layernorm', False)}")

    # Loss selection
    loss_type = train_cfg.get("loss_type", "ce")
    if loss_type == "focal":
        criterion = FocalLoss(
            gamma=train_cfg.get("focal_gamma", 2.0),
            label_smoothing=train_cfg["label_smoothing"],
        )
        logger.info(f"Using Focal Loss (gamma={train_cfg.get('focal_gamma', 2.0)})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=train_cfg["label_smoothing"],
        )
        logger.info(f"Using CrossEntropyLoss (class_weights={'yes' if class_weights is not None else 'no'})")

    # Mixup config
    mixup_alpha = train_cfg.get("mixup_alpha", 0.0)
    if mixup_alpha > 0:
        logger.info(f"Mixup enabled (alpha={mixup_alpha})")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    scheduler_type = train_cfg.get("scheduler_type", "plateau")
    if scheduler_type == "cosine_warm":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_cfg.get("cosine_T0", 10),
            T_mult=train_cfg.get("cosine_T_mult", 2),
            eta_min=train_cfg.get("cosine_eta_min", 1e-6),
        )
        logger.info(f"Scheduler: CosineAnnealingWarmRestarts "
                    f"(T_0={train_cfg.get('cosine_T0', 10)}, "
                    f"T_mult={train_cfg.get('cosine_T_mult', 2)})")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=train_cfg["scheduler_factor"],
            patience=train_cfg["scheduler_patience"],
        )
        logger.info("Scheduler: ReduceLROnPlateau")

    # TensorBoard
    writer = SummaryWriter(log_dir=config["output"]["log_dir"])

    # Checkpointing
    checkpoint_dir = config["output"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    epochs = train_cfg["epochs"]
    grad_clip = train_cfg["gradient_clip"]

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in tqdm(loaders["train"], desc=f"Epoch {epoch}/{epochs}",
                                     leave=False):
            features = features.to(device)
            labels = labels.to(device)

            # Apply mixup if enabled
            if mixup_alpha > 0:
                features, soft_targets = mixup_batch(features, labels, mixup_alpha, num_classes)

            optimizer.zero_grad()
            logits = model(features)

            if mixup_alpha > 0:
                loss = criterion(logits, soft_targets)
            else:
                loss = criterion(logits, labels)

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1)
            if mixup_alpha > 0:
                # Compare against dominant class in soft targets
                train_correct += (preds == soft_targets.argmax(dim=1)).sum().item()
            else:
                train_correct += (preds == labels).sum().item()
            train_total += features.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in loaders["val"]:
                features = features.to(device)
                labels = labels.to(device)

                logits = model(features)
                loss = criterion(logits, labels)

                val_loss += loss.item() * features.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += features.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        elapsed = time.time() - start_time

        # Logging
        logger.info(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Scheduler step
        if scheduler_type == "cosine_warm":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # Early stopping & checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "config": config,
            }, best_model_path)
            logger.info(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stop_patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    writer.close()

    # --- Test evaluation ---
    logger.info("Loading best model for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(model, loaders["test"], device, idx_to_label, config, logger)

    logger.info("Training complete.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ActivityLSTM")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
