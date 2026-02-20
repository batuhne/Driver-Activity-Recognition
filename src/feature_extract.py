"""Offline CNN feature extraction for Drive&Act dataset.

Extracts ResNet-18 features from video segments and saves as .npy files.
Usage: python src/feature_extract.py --config configs/config.yaml
"""

import os
import csv
import argparse

import numpy as np
import torch
from tqdm import tqdm

from src.utils import load_config, set_seed, setup_logging, build_file_id_to_video_path
from src.dataset import parse_annotations, DriveActVideoDataset
from src.models import CNNFeatureExtractor


def extract_features(config):
    """Extract CNN features for all segments and save to disk."""
    logger = setup_logging(config["output"]["log_dir"], "feature_extract")
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Parse annotations
    splits, label_to_idx, idx_to_label = parse_annotations(config)
    logger.info(f"Classes: {len(label_to_idx)}")
    for name, segs in splits.items():
        logger.info(f"  {name}: {len(segs)} segments")

    # Build video path mapping
    file_id_to_video = build_file_id_to_video_path(
        config["data"]["root"], config["data"]["video_dir"]
    )
    logger.info(f"Found {len(file_id_to_video)} video files")

    # Initialize CNN
    cnn = CNNFeatureExtractor().to(device)
    cnn.eval()

    data_root = config["data"]["root"]
    feature_dir = os.path.join(data_root, config["features"]["save_dir"])
    save_dtype = np.float16 if config["features"]["dtype"] == "float16" else np.float32

    for split_name, segments in splits.items():
        split_dir = os.path.join(feature_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Create dataset (no augmentation for feature extraction)
        dataset = DriveActVideoDataset(
            segments, file_id_to_video, config, is_train=False
        )

        manifest_rows = []

        for idx in tqdm(range(len(dataset)), desc=f"Extracting {split_name}"):
            seg = segments[idx]

            try:
                frames_tensor, label = dataset[idx]
            except (FileNotFoundError, Exception) as e:
                logger.warning(f"Skipping segment {idx}: {e}")
                continue

            # frames_tensor: (T, C, H, W)
            frames_tensor = frames_tensor.to(device)

            with torch.no_grad():
                features = cnn(frames_tensor)  # (T, 512)

            # Save features
            feature_filename = f"seg_{idx:06d}.npy"
            feature_path = os.path.join(split_dir, feature_filename)
            np.save(feature_path, features.cpu().numpy().astype(save_dtype))

            manifest_rows.append({
                "filename": feature_filename,
                "label": label,
                "activity": seg["activity"],
                "file_id": seg["file_id"],
                "frame_start": seg["frame_start"],
                "frame_end": seg["frame_end"],
            })

        # Write manifest
        manifest_path = os.path.join(split_dir, "manifest.csv")
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "label", "activity",
                               "file_id", "frame_start", "frame_end"]
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        logger.info(f"{split_name}: extracted {len(manifest_rows)} segments -> {split_dir}")

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CNN features")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    extract_features(config)
