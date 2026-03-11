"""Merge consecutive chunk features for longer temporal context.

Concatenates pre-extracted per-chunk .npy feature files into longer sequences.
Consecutive chunks are identified by matching file_id, activity, and zero frame gap.

Usage: python src/merge_features.py --source features_finetuned --output features_merged_k2 --merge_k 2
"""

import os
import csv
import argparse
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def find_consecutive_groups(df):
    """Find groups of consecutive chunks in a sorted manifest.

    Consecutive means:
    - Same file_id
    - Same activity (label)
    - frame_end[i] == frame_start[i+1] (zero gap)

    Args:
        df: DataFrame sorted by (file_id, frame_start) with columns:
            filename, label, activity, file_id, frame_start, frame_end

    Returns:
        List of lists, where each inner list contains row indices
        of consecutive chunks forming a group.
    """
    groups = []
    current_group = [0]

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if (curr["file_id"] == prev["file_id"]
                and curr["activity"] == prev["activity"]
                and curr["frame_start"] == prev["frame_end"]):
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]

    groups.append(current_group)
    return groups


def merge_features(source_dir, output_dir, merge_k, pad_mode="repeat"):
    """Merge pre-extracted chunk features into longer sequences.

    Args:
        source_dir: path to source features directory (e.g. features_finetuned/)
            Must contain {split}/manifest.csv and {split}/seg_*.npy
        output_dir: path to output directory for merged features
        merge_k: number of consecutive chunks to merge (2 or 3)
        pad_mode: padding for remainder/solo chunks ("repeat" = repeat last frame)
    """
    target_length = merge_k * 16  # Each chunk is 16 frames

    for split_name in ["train", "val", "test"]:
        split_source = os.path.join(source_dir, split_name)
        split_output = os.path.join(output_dir, split_name)
        manifest_path = os.path.join(split_source, "manifest.csv")

        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest not found: {manifest_path}, skipping {split_name}")
            continue

        # Resume support: skip if output manifest already exists with data
        out_manifest_path = os.path.join(split_output, "manifest.csv")
        if os.path.exists(out_manifest_path):
            with open(out_manifest_path) as f:
                existing_rows = sum(1 for _ in f) - 1  # minus header
            if existing_rows > 0:
                logger.info(f"{split_name}: already merged ({existing_rows} segments), skipping")
                continue

        os.makedirs(split_output, exist_ok=True)

        # Read source manifest
        df = pd.read_csv(manifest_path)
        # Ensure correct types
        df["frame_start"] = df["frame_start"].astype(int)
        df["frame_end"] = df["frame_end"].astype(int)
        df["label"] = df["label"].astype(int)

        # Sort by (file_id, frame_start) to find consecutive chunks
        df = df.sort_values(["file_id", "frame_start"]).reset_index(drop=True)

        # Find consecutive groups
        groups = find_consecutive_groups(df)

        merged_count = 0
        solo_count = 0
        output_rows = []
        seg_idx = 0

        for group_indices in groups:
            # Split group into non-overlapping sub-groups of size merge_k
            for start in range(0, len(group_indices), merge_k):
                subgroup = group_indices[start:start + merge_k]
                rows = [df.iloc[i] for i in subgroup]
                num_chunks = len(subgroup)

                # Load and concatenate features
                features_list = []
                for row in rows:
                    npy_path = os.path.join(split_source, row["filename"])
                    feat = np.load(npy_path)  # (16, 512) typically
                    features_list.append(feat)

                merged = np.concatenate(features_list, axis=0)  # (num_chunks*16, 512)

                # Pad if shorter than target length (solo/remainder chunks)
                if merged.shape[0] < target_length:
                    pad_count = target_length - merged.shape[0]
                    if pad_mode == "repeat":
                        pad = np.tile(merged[-1:], (pad_count, 1))
                    else:
                        pad = np.zeros((pad_count, merged.shape[1]), dtype=merged.dtype)
                    merged = np.concatenate([merged, pad], axis=0)
                    solo_count += 1
                else:
                    merged_count += 1

                # Save merged features
                out_filename = f"seg_{seg_idx:06d}.npy"
                np.save(os.path.join(split_output, out_filename), merged)

                # Manifest row — use first chunk's metadata, span full range
                output_rows.append({
                    "filename": out_filename,
                    "label": int(rows[0]["label"]),
                    "activity": rows[0]["activity"],
                    "file_id": rows[0]["file_id"],
                    "frame_start": int(rows[0]["frame_start"]),
                    "frame_end": int(rows[-1]["frame_end"]),
                    "num_chunks": num_chunks,
                })
                seg_idx += 1

        # Write output manifest
        with open(out_manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "label", "activity",
                               "file_id", "frame_start", "frame_end", "num_chunks"]
            )
            writer.writeheader()
            writer.writerows(output_rows)

        logger.info(
            f"{split_name}: {len(output_rows)} merged segments "
            f"(from {len(df)} source chunks, "
            f"{merged_count} fully merged, {solo_count} padded)"
        )

    logger.info(f"Merge complete -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge consecutive chunk features")
    parser.add_argument("--source", required=True,
                        help="Source features directory (e.g. features_finetuned)")
    parser.add_argument("--output", required=True,
                        help="Output directory (e.g. features_merged_k2)")
    parser.add_argument("--merge_k", type=int, required=True,
                        help="Number of consecutive chunks to merge (2 or 3)")
    parser.add_argument("--data_root", default="./data",
                        help="Data root directory")
    parser.add_argument("--pad_mode", default="repeat", choices=["repeat", "zero"],
                        help="Padding mode for solo/remainder chunks")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    source_dir = os.path.join(args.data_root, args.source)
    output_dir = os.path.join(args.data_root, args.output)

    merge_features(source_dir, output_dir, args.merge_k, args.pad_mode)
