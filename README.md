# Driver Activity Recognition Using Deep Temporal Models

Fine-grained driver activity recognition from the **Drive&Act** dataset using a CNN+LSTM architecture. Recognizes 34+ mid-level activity classes from Kinect IR video.

## Architecture

Two-stage pipeline:

1. **Feature Extraction (offline):** Frozen pretrained ResNet-18 extracts 512-dim features per frame
2. **Temporal Modeling:** 2-layer LSTM (hidden=256) processes sequences of 16 frames, followed by FC layer with softmax over activity classes

The CNN is kept frozen to prevent overfitting on limited data. Temporal and spatial modeling are separated for interpretability.

## Project Structure

```
├── configs/config.yaml          # Central configuration
├── src/
│   ├── utils.py                 # Config loading, seed, logging, helpers
│   ├── dataset.py               # Data loading, annotation parsing, augmentation
│   ├── models.py                # CNNFeatureExtractor, ActivityLSTM, CNNLSTMModel
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

### Step 1: Feature Extraction

```bash
python src/feature_extract.py --config configs/config.yaml
```

Extracts ResNet-18 features for all segments → `data/features/{train,val,test}/`

### Step 2: LSTM Training

```bash
python src/train.py --config configs/config.yaml
```

Trains ActivityLSTM with early stopping, saves best model to `results/checkpoints/`

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 16 frames (~3s at 5fps) |
| Frame size | 224x224, ImageNet normalization |
| LSTM | 2 layers, hidden=256, dropout=0.3 |
| FC dropout | 0.5 |
| Batch size | 32 |
| Learning rate | 0.001 (Adam) |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Early stopping | patience=10 |

## Expected Results

- **Target:** 50-60% mean per-class accuracy
- **Upper bound:** ~65% (I3D on IR-only, Drive&Act paper)
- **Primary metric:** Mean per-class accuracy (equal weight per class)

## References

- Martin, M., et al. (2019). "Drive&Act: A Multi-Modal Dataset for Fine-Grained Driver Behavior Recognition in Autonomous Vehicles." ICCV 2019.
