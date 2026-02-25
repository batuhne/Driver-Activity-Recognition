"""Training script for ActivityLSTM (feature-based) and CNN+LSTM (end-to-end).

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
from src.models import ActivityLSTM, CNNLSTMModel
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
    """Full training pipeline. Supports feature-based (LSTM only) and end-to-end (CNN+LSTM) modes."""
    logger = setup_logging(config["output"]["log_dir"], "train")
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_cfg = config["training"]
    model_cfg = config["model"]
    mode = train_cfg.get("mode", "feature_based")
    logger.info(f"Training mode: {mode}")

    # Parse annotations to get num_classes
    splits, label_to_idx, idx_to_label = parse_annotations(config)
    num_classes = len(label_to_idx)
    logger.info(f"Number of classes: {num_classes}")

    # Create dataloaders
    if mode == "end_to_end":
        loaders = get_dataloaders(config, splits=splits, feature_based=False, num_classes=num_classes)
    else:
        loaders = get_dataloaders(config, feature_based=True, num_classes=num_classes)

    logger.info(f"Train: {len(loaders['train'].dataset)} samples")
    logger.info(f"Val: {len(loaders['val'].dataset)} samples")
    logger.info(f"Test: {len(loaders['test'].dataset)} samples")

    # Class weights for loss (when not using weighted sampler)
    train_labels = loaders.pop("train_labels", None)
    use_sampler = train_cfg.get("use_weighted_sampler", True)
    class_weights = None
    if not use_sampler and train_labels is not None:
        beta = train_cfg.get("en_beta", 0.999)
        class_weights = compute_effective_number_weights(train_labels, num_classes, beta=beta).to(device)
        raw_ratio = class_weights.max() / class_weights.min()
        class_weights = torch.sqrt(class_weights)
        class_weights = class_weights / class_weights.mean()
        logger.info(f"Class-weighted loss: EN beta={beta}, sqrt dampened, "
                    f"weight range=[{class_weights.min():.4f}, {class_weights.max():.4f}], "
                    f"ratio={class_weights.max()/class_weights.min():.1f}x (raw: {raw_ratio:.1f}x)")
    elif use_sampler:
        logger.info("Using WeightedRandomSampler for class balancing")

    # Model
    if mode == "end_to_end":
        freeze_mode = model_cfg.get("freeze_mode", "all")
        model = CNNLSTMModel(
            num_classes=num_classes,
            hidden_dim=model_cfg["lstm_hidden"],
            num_layers=model_cfg["lstm_layers"],
            lstm_dropout=model_cfg["lstm_dropout"],
            fc_dropout=model_cfg["fc_dropout"],
            use_layernorm=model_cfg.get("use_layernorm", False),
            bidirectional=model_cfg.get("bidirectional", False),
            pooling=model_cfg.get("pooling", "last"),
            noise_std=train_cfg.get("noise_std", 0.0),
            freeze_mode=freeze_mode,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"CNNLSTMModel — total: {total_params:,}, trainable: {trainable_params:,}")
        logger.info(f"CNN freeze_mode: {freeze_mode}")

        # Log CNN layer status
        for idx, (name, child) in enumerate(model.cnn.features.named_children()):
            child_params = sum(p.numel() for p in child.parameters())
            child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            if child_params > 0:
                status = "TRAINABLE" if child_trainable > 0 else "frozen"
                logger.info(f"  CNN [{idx}] {name}: {child_params:,} params ({status})")
    else:
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
            noise_std=train_cfg.get("noise_std", 0.0),
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

    # Mixup config (disabled for end-to-end)
    mixup_alpha = train_cfg.get("mixup_alpha", 0.0)
    if mode == "end_to_end":
        mixup_alpha = 0.0
    if mixup_alpha > 0:
        logger.info(f"Mixup enabled (alpha={mixup_alpha})")

    # Optimizer — differential LR for end-to-end mode
    if mode == "end_to_end":
        cnn_lr = train_cfg.get("cnn_lr", 1e-5)
        lstm_lr = train_cfg["lr"]
        # Only include params that currently require grad
        # (during warmup, CNN params are frozen so only LSTM params are included)
        cnn_params = [p for n, p in model.named_parameters() if "cnn" in n and p.requires_grad]
        lstm_params = [p for n, p in model.named_parameters() if "cnn" not in n and p.requires_grad]
        param_groups = [{"params": lstm_params, "lr": lstm_lr}]
        if cnn_params:
            param_groups.insert(0, {"params": cnn_params, "lr": cnn_lr})
        optimizer = torch.optim.Adam(param_groups, weight_decay=train_cfg["weight_decay"])
        if cnn_params:
            logger.info(f"Differential LR — CNN: {cnn_lr}, LSTM: {lstm_lr}")
        else:
            logger.info(f"LSTM-only optimizer (CNN frozen for warmup), LR: {lstm_lr}")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

    # Scheduler (operates on all param groups)
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

    # AMP (mixed precision)
    use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Mixed precision (AMP) enabled")

    # Gradient accumulation
    accum_steps = train_cfg.get("accumulation_steps", 1)
    if accum_steps > 1:
        logger.info(f"Gradient accumulation: {accum_steps} steps (effective batch={train_cfg['batch_size'] * accum_steps})")

    # CNN warmup config (end-to-end only)
    cnn_warmup_epochs = train_cfg.get("cnn_warmup_epochs", 0) if mode == "end_to_end" else 0
    if cnn_warmup_epochs > 0:
        freeze_mode = model_cfg.get("freeze_mode", "all")
        # Freeze CNN for warmup period
        for p in model.cnn.parameters():
            p.requires_grad = False
        logger.info(f"CNN warmup: {cnn_warmup_epochs} epochs (CNN frozen, then {freeze_mode})")

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

        # CNN warmup: unfreeze CNN after warmup period
        if cnn_warmup_epochs > 0 and epoch == cnn_warmup_epochs + 1:
            freeze_mode = model_cfg.get("freeze_mode", "all")
            model.cnn.unfreeze(freeze_mode)
            # Re-build optimizer and scheduler with CNN params now included
            cnn_params = [p for n, p in model.named_parameters() if "cnn" in n and p.requires_grad]
            lstm_params = [p for n, p in model.named_parameters() if "cnn" not in n and p.requires_grad]
            cnn_lr = train_cfg.get("cnn_lr", 1e-5)
            optimizer = torch.optim.Adam([
                {"params": cnn_params, "lr": cnn_lr},
                {"params": lstm_params, "lr": train_cfg["lr"]},
            ], weight_decay=train_cfg["weight_decay"])
            # Rebuild scheduler for new optimizer
            if scheduler_type == "cosine_warm":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=train_cfg.get("cosine_T0", 10),
                    T_mult=train_cfg.get("cosine_T_mult", 2),
                    eta_min=train_cfg.get("cosine_eta_min", 1e-6),
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min",
                    factor=train_cfg["scheduler_factor"],
                    patience=train_cfg["scheduler_patience"],
                )
            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"  -> CNN unfrozen (epoch {epoch}), trainable params: {trainable_now:,}")

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for step, (inputs, labels) in enumerate(tqdm(loaders["train"],
                                                     desc=f"Epoch {epoch}/{epochs}",
                                                     leave=False)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Apply mixup if enabled (feature-based only)
            if mixup_alpha > 0:
                inputs, soft_targets = mixup_batch(inputs, labels, mixup_alpha, num_classes)

            # Forward pass (with optional AMP)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(inputs)
                    if mixup_alpha > 0:
                        loss = criterion(logits, soft_targets)
                    else:
                        loss = criterion(logits, labels)
            else:
                logits = model(inputs)
                if mixup_alpha > 0:
                    loss = criterion(logits, soft_targets)
                else:
                    loss = criterion(logits, labels)

            # Scale loss for gradient accumulation
            loss_scaled = loss / accum_steps

            # Backward
            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Optimizer step at accumulation boundary
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loaders["train"]):
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            if mixup_alpha > 0:
                train_correct += (preds == soft_targets.argmax(dim=1)).sum().item()
            else:
                train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in loaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        elapsed = time.time() - start_time

        # Logging
        lr_str = f"LR: {optimizer.param_groups[-1]['lr']:.6f}"
        if mode == "end_to_end" and len(optimizer.param_groups) == 2:
            lr_str = f"CNN_LR: {optimizer.param_groups[0]['lr']:.6f} LSTM_LR: {optimizer.param_groups[1]['lr']:.6f}"
        logger.info(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"{lr_str}"
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[-1]["lr"], epoch)

        # GPU memory logging (end-to-end)
        if device.type == "cuda" and mode == "end_to_end":
            mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
            writer.add_scalar("GPU_Memory_GB", mem_alloc, epoch)
            if epoch == 1:
                logger.info(f"  GPU peak memory: {mem_alloc:.2f} GB")

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

    # For end-to-end model, evaluate using the LSTM sub-module with video input
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
