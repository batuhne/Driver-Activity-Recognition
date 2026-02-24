# Driver Activity Recognition Using Deep Temporal Models

Fine-grained driver activity recognition from the **Drive&Act** dataset using a CNN+LSTM architecture. Recognizes 39 mid-level activity classes from Kinect IR video.

## Architecture

Two-stage pipeline:

1. **Feature Extraction (offline):** Frozen pretrained ResNet-18 extracts 512-dim features per frame
2. **Temporal Modeling:** 2-layer BiLSTM (hidden=256) with attention pooling processes sequences of 16 frames, followed by FC layer with softmax over activity classes

The CNN is kept frozen to prevent overfitting on limited data. Temporal and spatial modeling are separated for interpretability. Attention weights reveal which timesteps drive each prediction.

## Project Structure

```
├── configs/config.yaml          # Central configuration
├── src/
│   ├── utils.py                 # Config loading, seed, logging, IR stats
│   ├── dataset.py               # Data loading, annotation parsing, mixup
│   ├── models.py                # CNNFeatureExtractor, ActivityLSTM
│   ├── feature_extract.py       # Offline CNN feature extraction
│   ├── train.py                 # Training loop with early stopping
│   └── evaluate.py              # Metrics, confusion matrix, plots
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_lstm_training.ipynb
│   └── 04_results_visualization.ipynb
├── data/                        # Not tracked in git
│   ├── kinect_ir/vp{1-15}/      # 30 Kinect IR videos (512x424, 30fps)
│   ├── activities_3s/           # Annotation CSV files
│   └── features/                # Extracted .npy features (generated)
└── results/                     # Not tracked in git
    ├── checkpoints/
    ├── logs/
    └── figures/
```

## Setup

### Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation

Place the Drive&Act data under `data/`:

```
data/
├── kinect_ir/
│   ├── vp1/
│   │   ├── run1b_2018-05-29-14-02-47.kinect_ir.mp4
│   │   └── run2_2018-05-29-14-33-44.kinect_ir.mp4
│   ├── vp2/ ...
│   └── vp15/ ...
└── activities_3s/
    └── kinect_ir/
        ├── midlevel.chunks_90.csv
        ├── midlevel.chunks_90.split_0.train.csv
        ├── midlevel.chunks_90.split_0.val.csv
        └── midlevel.chunks_90.split_0.test.csv
```

### Google Colab

1. Upload `data/` to Google Drive at `MyDrive/DriveAndAct/`
2. Open notebooks from `notebooks/` in Colab
3. Each notebook auto-detects Colab and mounts Drive

## Usage

### Step 1: Compute IR Statistics

The `02_feature_extraction.ipynb` notebook computes dataset-specific mean/std from training videos before extraction. This replaces ImageNet normalization which is incorrect for grayscale IR images.

### Step 2: Feature Extraction

```bash
python src/feature_extract.py --config configs/config.yaml
```

Extracts ResNet-18 features (float32) for all segments → `data/features/{train,val,test}/`

### Step 3: LSTM Training

```bash
python src/train.py --config configs/config.yaml
```

Trains BiLSTM with attention pooling, early stopping, saves best model to `results/checkpoints/`

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 16 frames (~3s at 5fps) |
| Frame size | 224x224, IR normalization (computed from data) |
| LSTM | 2-layer BiLSTM, hidden=256, dropout=0.5 |
| Pooling | Attention |
| Batch size | 32 |
| Learning rate | 0.001 (Adam) |
| Loss | CrossEntropyLoss + label smoothing 0.1 |
| Class balancing | WeightedRandomSampler |
| Augmentation | Mixup (alpha=0.2), Gaussian noise (std=0.1) |
| Gradient clipping | 1.0 |
| Early stopping | patience=15 |

## Data Split

Subject-based split (Split 0):
- **Train:** vp1-10 (6559 segments, ~65%)
- **Val:** vp11-12 (1405 segments, ~14%)
- **Test:** vp13-15 (2184 segments, ~21%)

## Results

| Metric | Run 3 |
|--------|-------|
| Overall Accuracy | 44.8% |
| Mean Per-Class Accuracy | 39.2% |
| Macro F1 | 0.33 |
| Weighted F1 | 0.48 |

**Literature reference:** I3D on IR-only achieves ~65% MPCA (Martin et al., ICCV 2019). The gap is expected, I3D learns spatiotemporal features end-to-end with 3D convolutions, while our frozen 2D CNN + LSTM architecture separates spatial and temporal modeling for interpretability.

## References

- Martin, M., et al. (2019). "Drive&Act: A Multi-Modal Dataset for Fine-Grained Driver Behavior Recognition in Autonomous Vehicles." ICCV 2019.
