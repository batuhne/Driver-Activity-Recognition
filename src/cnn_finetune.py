"""Per-frame CNN fine-tuning for Drive&Act IR dataset.

Fine-tunes ResNet-18 layer4 as a frame-level classifier, then saves
the backbone weights for feature re-extraction. This is the fast
alternative to end-to-end CNN+LSTM training (avoids video I/O bottleneck).

Usage: python src/cnn_finetune.py --config configs/config.yaml
"""

import os
import argparse
import time
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from tqdm import tqdm

from src.utils import (
    load_config, set_seed, setup_logging,
    build_file_id_to_video_path, compute_effective_number_weights,
)
from src.dataset import parse_annotations
from src.models import CNNFeatureExtractor


class SingleFrameDataset(Dataset):
    """Dataset that reads one random frame per segment for CNN fine-tuning.

    Much faster than reading 16 frames per segment: only 1 video seek per sample.
    Each epoch samples a different random frame from each segment.
    """

    def __init__(self, segments, file_id_to_video, config, is_train=True):
        self.segments = segments
        self.file_id_to_video = file_id_to_video
        self.frame_size = config["processing"]["frame_size"]
        self.sample_fps = config["processing"]["sample_fps"]
        self.original_fps = config["processing"]["original_fps"]
        self.frame_stride = self.original_fps // self.sample_fps
        self.is_train = is_train

        ir_mean = config["processing"]["ir_mean"]
        ir_std = config["processing"]["ir_std"]
        mean = [ir_mean, ir_mean, ir_mean]
        std = [ir_std, ir_std, ir_std]

        if is_train:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(self.frame_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(int(self.frame_size * 1.14)),
                T.CenterCrop(self.frame_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        video_path = self.file_id_to_video.get(seg["file_id"])
        if video_path is None:
            raise FileNotFoundError(f"Video not found for file_id: {seg['file_id']}")

        # Candidate frame indices at target fps
        candidates = list(range(seg["frame_start"], seg["frame_end"], self.frame_stride))
        if not candidates:
            candidates = [seg["frame_start"]]

        # Pick one random frame (train) or center frame (val/test)
        if self.is_train:
            frame_idx = random.choice(candidates)
        else:
            frame_idx = candidates[len(candidates) // 2]

        # Read single frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            frame = np.zeros((424, 512, 3), dtype=np.uint8)
        elif len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_tensor = self.transform(frame)
        label = seg["label_idx"]
        return frame_tensor, label


class CNNClassifier(nn.Module):
    """ResNet-18 backbone + temporary classification head for fine-tuning."""

    def __init__(self, num_classes, freeze_mode="layer4"):
        super().__init__()
        self.backbone = CNNFeatureExtractor(freeze_mode=freeze_mode)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # (batch, 512)
        return self.fc(features)


def finetune_cnn(config):
    """Fine-tune ResNet-18 layer4 as a per-frame classifier."""
    logger = setup_logging(config["output"]["log_dir"], "cnn_finetune")
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ft_cfg = config.get("cnn_finetune", {})
    freeze_mode = config["model"].get("freeze_mode", "layer4")

    # Parse annotations
    splits, label_to_idx, idx_to_label = parse_annotations(config)
    num_classes = len(label_to_idx)
    logger.info(f"Number of classes: {num_classes}")

    # Build video path mapping
    file_id_to_video = build_file_id_to_video_path(
        config["data"]["root"], config["data"]["video_dir"]
    )
    logger.info(f"Found {len(file_id_to_video)} video files")

    # Datasets
    train_dataset = SingleFrameDataset(splits["train"], file_id_to_video, config, is_train=True)
    val_dataset = SingleFrameDataset(splits["val"], file_id_to_video, config, is_train=False)
    logger.info(f"Train: {len(train_dataset)} segments, Val: {len(val_dataset)} segments")

    # Sampler (same EN weighting as Run 3)
    batch_size = ft_cfg.get("batch_size", 64)
    num_workers = config["training"].get("num_workers", 2)
    use_sampler = ft_cfg.get("use_weighted_sampler", True)

    if use_sampler:
        labels = [s["label_idx"] for s in splits["train"]]
        beta = config["training"].get("en_beta", 0.99)
        weights = compute_effective_number_weights(labels, num_classes, beta=beta)
        sample_weights = [weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
        logger.info(f"WeightedRandomSampler (EN beta={beta})")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Model
    model = CNNClassifier(num_classes=num_classes, freeze_mode=freeze_mode).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CNNClassifier — total: {total_params:,}, trainable: {trainable_params:,}")
    logger.info(f"freeze_mode: {freeze_mode}")

    # Log layer status
    for idx, (name, child) in enumerate(model.backbone.features.named_children()):
        child_params = sum(p.numel() for p in child.parameters())
        child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        if child_params > 0:
            status = "TRAINABLE" if child_trainable > 0 else "frozen"
            logger.info(f"  CNN [{idx}] {name}: {child_params:,} params ({status})")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=ft_cfg.get("label_smoothing", 0.1))

    # Optimizer — differential LR
    cnn_lr = ft_cfg.get("cnn_lr", 1e-4)
    fc_lr = ft_cfg.get("fc_lr", 1e-3)
    cnn_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    fc_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.Adam([
        {"params": cnn_params, "lr": cnn_lr},
        {"params": fc_params, "lr": fc_lr},
    ], weight_decay=ft_cfg.get("weight_decay", 1e-4))
    logger.info(f"Differential LR — CNN: {cnn_lr}, FC: {fc_lr}")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # AMP
    use_amp = ft_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Mixed precision (AMP) enabled")

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config["output"]["log_dir"], "cnn_finetune"))

    # Checkpointing
    checkpoint_dir = config["output"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "cnn_finetuned.pth")

    # Training loop
    epochs = ft_cfg.get("epochs", 15)
    patience = ft_cfg.get("early_stop_patience", 7)
    grad_clip = ft_cfg.get("gradient_clip", 1.0)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(frames)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(frames)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_loss += loss.item() * frames.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += frames.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(frames)
                        loss = criterion(logits, labels)
                else:
                    logits = model(frames)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * frames.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += frames.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"CNN_LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        if device.type == "cuda" and epoch == 1:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"  GPU peak memory: {mem:.2f} GB")

        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save only backbone weights (not the temporary FC head)
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": model.backbone.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "freeze_mode": freeze_mode,
                "config": config,
            }, best_model_path)
            logger.info(f"  -> Saved best backbone (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    logger.info(f"CNN fine-tuning complete. Best backbone saved to {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CNN backbone")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    finetune_cnn(config)
