"""Dataset classes for Drive&Act driver activity recognition."""

import os
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from src.utils import get_activity_labels, build_file_id_to_video_path, compute_class_weights_from_labels


def parse_annotations(config):
    """Parse Drive&Act split CSV files and return annotation dicts per split.

    Uses pre-defined split files (split_0, split_1, or split_2).
    Filters out segments shorter than min_segment_frames.

    Returns:
        splits: dict with keys 'train', 'val', 'test', each containing a list of
                dicts with keys: participant_id, file_id, annotation_id,
                frame_start, frame_end, activity, chunk_id, label_idx
        label_to_idx: dict mapping activity name -> integer index
        idx_to_label: dict mapping integer index -> activity name
    """
    data_root = config["data"]["root"]
    ann_dir = os.path.join(data_root, config["data"]["annotation_dir"])
    split_idx = config["data"]["split"]
    min_frames = config["data"]["min_segment_frames"]

    # Get label mapping from main annotation file
    main_csv = os.path.join(ann_dir, config["data"]["annotation_file"])
    label_to_idx, idx_to_label = get_activity_labels(main_csv)

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(
            ann_dir,
            f"midlevel.chunks_90.split_{split_idx}.{split_name}.csv"
        )
        df = pd.read_csv(split_file)

        # Compute segment length and filter short segments
        df["duration"] = df["frame_end"] - df["frame_start"]
        df = df[df["duration"] >= min_frames].reset_index(drop=True)

        segments = []
        for _, row in df.iterrows():
            segments.append({
                "participant_id": int(row["participant_id"]),
                "file_id": row["file_id"],
                "annotation_id": int(row["annotation_id"]),
                "frame_start": int(row["frame_start"]),
                "frame_end": int(row["frame_end"]),
                "activity": row["activity"],
                "chunk_id": int(row["chunk_id"]),
                "label_idx": label_to_idx[row["activity"]],
            })

        splits[split_name] = segments

    return splits, label_to_idx, idx_to_label


class DriveActVideoDataset(Dataset):
    """Dataset for loading video frames for CNN feature extraction.

    Performs uniform temporal sampling of T frames from each segment,
    converts grayscale IR to 3-channel, and applies augmentations.
    """

    def __init__(self, segments, file_id_to_video, config, is_train=True):
        """
        Args:
            segments: list of segment dicts from parse_annotations
            file_id_to_video: dict mapping file_id -> video path
            config: loaded config dict
            is_train: whether to apply training augmentations
        """
        self.segments = segments
        self.file_id_to_video = file_id_to_video
        self.seq_length = config["processing"]["seq_length"]
        self.frame_size = config["processing"]["frame_size"]
        self.sample_fps = config["processing"]["sample_fps"]
        self.original_fps = config["processing"]["original_fps"]
        self.is_train = is_train

        # Frame sampling stride: pick every Nth frame to simulate lower fps
        self.frame_stride = self.original_fps // self.sample_fps  # 30/5 = 6

        mean = config["processing"]["imagenet_mean"]
        std = config["processing"]["imagenet_std"]

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
                T.Resize(int(self.frame_size * 1.14)),  # 256 for 224 crop
                T.CenterCrop(self.frame_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.segments)

    def _read_frames(self, video_path, frame_start, frame_end):
        """Read and uniformly sample T frames from a video segment.

        Applies temporal subsampling (stride) and uniform sampling
        to get exactly seq_length frames.
        """
        # Compute candidate frame indices at target fps
        candidate_indices = list(range(frame_start, frame_end, self.frame_stride))

        if len(candidate_indices) == 0:
            candidate_indices = [frame_start]

        # Uniform temporal sampling to get exactly seq_length frames
        if len(candidate_indices) >= self.seq_length:
            # Uniformly sample seq_length frames
            indices = np.linspace(0, len(candidate_indices) - 1,
                                  self.seq_length, dtype=int)
            selected = [candidate_indices[i] for i in indices]
        else:
            # Pad by repeating last frame
            selected = candidate_indices + [candidate_indices[-1]] * (
                self.seq_length - len(candidate_indices)
            )

        # Add temporal jitter during training
        if self.is_train:
            jitter = np.random.randint(-2, 3, size=len(selected))
            selected = [max(frame_start, min(frame_end - 1, s + j))
                        for s, j in zip(selected, jitter)]

        # Read frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        for frame_idx in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert grayscale IR to 3-channel if needed
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame read fails, duplicate last good frame or use black
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((424, 512, 3), dtype=np.uint8))
        cap.release()

        return frames

    def __getitem__(self, idx):
        seg = self.segments[idx]
        video_path = self.file_id_to_video.get(seg["file_id"])

        if video_path is None:
            raise FileNotFoundError(
                f"Video not found for file_id: {seg['file_id']}"
            )

        frames = self._read_frames(video_path, seg["frame_start"], seg["frame_end"])

        # Apply transforms to each frame
        transformed = []
        for frame in frames:
            transformed.append(self.transform(frame))

        # Stack to (T, C, H, W)
        frames_tensor = torch.stack(transformed, dim=0)
        label = seg["label_idx"]

        return frames_tensor, label


class DriveActFeatureDataset(Dataset):
    """Dataset for loading pre-extracted CNN features for LSTM training.

    Loads .npy feature files (T, 512) from a manifest CSV.
    """

    def __init__(self, manifest_path, feature_dir):
        """
        Args:
            manifest_path: path to manifest CSV with columns: filename, label
            feature_dir: directory containing .npy feature files
        """
        self.feature_dir = feature_dir
        self.samples = []

        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "filename": row["filename"],
                    "label": int(row["label"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feature_path = os.path.join(self.feature_dir, sample["filename"])
        features = np.load(feature_path).astype(np.float32)
        label = sample["label"]
        return torch.from_numpy(features), label


def get_dataloaders(config, splits=None, feature_based=True):
    """Create DataLoaders with WeightedRandomSampler for class imbalance.

    Args:
        config: loaded config dict
        splits: output of parse_annotations (if feature_based=False)
        feature_based: if True, use DriveActFeatureDataset; else DriveActVideoDataset

    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    data_root = config["data"]["root"]

    loaders = {}

    if feature_based:
        feature_dir = os.path.join(data_root, config["features"]["save_dir"])

        for split_name in ["train", "val", "test"]:
            manifest = os.path.join(feature_dir, split_name, "manifest.csv")
            split_feature_dir = os.path.join(feature_dir, split_name)
            dataset = DriveActFeatureDataset(manifest, split_feature_dir)

            if split_name == "train":
                # Weighted sampling for class imbalance
                labels = [s["label"] for s in dataset.samples]
                num_classes = max(labels) + 1
                weights = compute_class_weights_from_labels(labels, num_classes)
                sample_weights = [weights[l] for l in labels]
                sampler = WeightedRandomSampler(
                    sample_weights, len(sample_weights), replacement=True
                )
                loaders[split_name] = DataLoader(
                    dataset, batch_size=batch_size, sampler=sampler,
                    num_workers=num_workers, pin_memory=True,
                )
            else:
                loaders[split_name] = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                )
    else:
        assert splits is not None, "splits required for video-based loading"
        file_id_to_video = build_file_id_to_video_path(
            data_root, config["data"]["video_dir"]
        )

        for split_name in ["train", "val", "test"]:
            is_train = (split_name == "train")
            dataset = DriveActVideoDataset(
                splits[split_name], file_id_to_video, config, is_train=is_train
            )

            if is_train:
                labels = [s["label_idx"] for s in splits[split_name]]
                num_classes = max(labels) + 1
                weights = compute_class_weights_from_labels(labels, num_classes)
                sample_weights = [weights[l] for l in labels]
                sampler = WeightedRandomSampler(
                    sample_weights, len(sample_weights), replacement=True
                )
                loaders[split_name] = DataLoader(
                    dataset, batch_size=batch_size, sampler=sampler,
                    num_workers=num_workers, pin_memory=True,
                )
            else:
                loaders[split_name] = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                )

    return loaders
