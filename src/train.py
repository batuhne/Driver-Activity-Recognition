"""Training script for ActivityLSTM on pre-extracted features.

Usage: python src/train.py --config configs/config.yaml
"""

import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import load_config, set_seed, setup_logging
from src.dataset import parse_annotations, get_dataloaders, DriveActFeatureDataset
from src.models import ActivityLSTM
from src.evaluate import evaluate_model, compute_all_metrics


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
    loaders = get_dataloaders(config, feature_based=True)
    logger.info(f"Train: {len(loaders['train'].dataset)} samples")
    logger.info(f"Val: {len(loaders['val'].dataset)} samples")
    logger.info(f"Test: {len(loaders['test'].dataset)} samples")

    # Model
    model = ActivityLSTM(
        input_dim=config["model"]["feature_dim"],
        hidden_dim=config["model"]["lstm_hidden"],
        num_layers=config["model"]["lstm_layers"],
        num_classes=num_classes,
        lstm_dropout=config["model"]["lstm_dropout"],
        fc_dropout=config["model"]["fc_dropout"],
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"])

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config["training"]["scheduler_factor"],
        patience=config["training"]["scheduler_patience"],
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=config["output"]["log_dir"])

    # Checkpointing
    checkpoint_dir = config["output"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    epochs = config["training"]["epochs"]
    grad_clip = config["training"]["gradient_clip"]

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

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1)
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
            if patience_counter >= config["training"]["early_stop_patience"]:
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
