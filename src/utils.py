"""Utility functions for Driver Activity Recognition pipeline."""

import os
import random
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def load_config(config_path="configs/config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir="results/logs", name="train"):
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def get_activity_labels(annotation_path):
    """Extract sorted unique activity labels and create label-to-index mapping.

    Returns:
        label_to_idx: dict mapping activity name -> integer index
        idx_to_label: dict mapping integer index -> activity name
    """
    df = pd.read_csv(annotation_path)
    activities = sorted(df["activity"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(activities)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def compute_class_weights_from_labels(labels, num_classes):
    """Compute balanced class weights using sklearn.

    Args:
        labels: array of integer class labels
        num_classes: total number of classes

    Returns:
        torch.FloatTensor of shape (num_classes,) with class weights
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    return torch.FloatTensor(weights)


def build_file_id_to_video_path(data_root, video_dir="kinect_ir"):
    """Map annotation file_id to actual video file path.

    Annotation file_id format: 'vp1/run1b_2018-05-29-14-02-47.kinect_ir'
    Video file format:          'kinect_ir/vp1/run1b_2018-05-29-14-02-47.kinect_ir.mp4'

    Returns:
        dict mapping file_id -> absolute video path
    """
    video_root = os.path.join(data_root, video_dir)
    mapping = {}

    for vp_dir in sorted(Path(video_root).iterdir()):
        if not vp_dir.is_dir():
            continue
        for video_file in vp_dir.glob("*.mp4"):
            # file_id = "vp1/run1b_2018-05-29-14-02-47.kinect_ir"
            file_id = f"{vp_dir.name}/{video_file.stem}"
            mapping[file_id] = str(video_file)

    return mapping


def override_config_paths(config, colab_root="/content/drive/MyDrive/DriveAndAct"):
    """Override data paths for Google Colab environment.

    Args:
        config: loaded config dict
        colab_root: path to data on Google Drive
    """
    config["data"]["root"] = colab_root
    return config


def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False
