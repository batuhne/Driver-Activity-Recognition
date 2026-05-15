# Driver Activity Recognition Using Deep Temporal Models

Fine-grained driver activity recognition on the **Drive&Act** dataset, combining a CNN spatial backbone with a BiLSTM temporal model. The best configuration fuses Kinect IR and Depth streams at the feature level before temporal modeling and recognizes 34 mid-level driver activity classes.

> **Undergraduate thesis project (Spring 2026).**
> **Author:** A. Batuhan Erdoğan.
> **Advisor:** Barış Çiçek.
> **Department of Mathematics, Izmir Institute of Technology.**

## Architecture

The model has two stages.

**Stage 1 (per-modality spatial encoding).** Each modality uses its own ResNet-18, ImageNet pretrained, with the final residual block (`layer4`) fine-tuned on Drive&Act while earlier blocks stay frozen. The CNN outputs a 512-dim feature per frame. Two CNNs are trained independently, one on IR clips and one on Depth clips.

**Stage 2 (temporal fusion and classification).** For every frame, the IR and Depth feature vectors are concatenated into a 1024-dim representation. The resulting sequence is fed to a 2-layer BiLSTM (hidden=256), followed by attention pooling that aggregates the timesteps into a single clip embedding, which a fully connected layer maps to 34 activity logits via softmax.

Why this design: keeping the early CNN layers frozen prevents overfitting on a relatively small dataset, fine-tuning `layer4` adapts the high-level features from natural-image ImageNet statistics to single-channel in-cabin Kinect imagery, and attention pooling exposes which timesteps drive each prediction.

**Chunk merging.** The Drive&Act annotations are 90-frame chunks (3s at 30fps). Consecutive same-activity chunks with no frame gap are merged before training, extending the effective context window to 6s (k=2). This also cuts zero-padding from 53% to 6% of clip frames, so the BiLSTM sees real temporal evidence instead of padding tokens. Chunk merging contributes a large slice of the final accuracy at zero added parameters.

## Dataset

Drive&Act is the largest publicly available multi-modal driver behavior benchmark (Martin et al., ICCV 2019).

| Property | Value |
|---|---|
| Subjects | 15 (`vp1` through `vp15`), 2 driving sessions each |
| Total recordings | 30 continuous videos |
| Activity classes | 34 mid-level fine-grained labels |
| Total frames | ~9.6 million |
| Frame rate | 30 fps |
| Kinect modalities used | IR and Depth (each single-channel 512x424) |

Each modality captures complementary information. IR encodes texture and appearance (hand positions, object shapes, clothing detail). Depth encodes 3D geometry (body pose, spatial relationships, distances). Frames are resized to 224x224 before being fed to the CNN.

## Project Structure

```
├── configs/config.yaml          # Central configuration
├── src/
│   ├── utils.py                 # Config loading, seed, logging, IR/Depth stats
│   ├── dataset.py               # Data loading, annotations, fusion dataset, mixup
│   ├── models.py                # CNNFeatureExtractor, ActivityLSTM, CNNLSTMModel
│   ├── cnn_finetune.py          # CNN fine-tuning loop (layer4 unfrozen)
│   ├── feature_extract.py       # Offline CNN feature extraction
│   ├── merge_features.py        # Chunk merging utility
│   ├── train.py                 # LSTM training loop with early stopping
│   └── evaluate.py              # Metrics, confusion matrix, plots
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_lstm_training.ipynb         # Run 7b: IR-only LSTM on merged features
│   ├── 03a_cnn_finetune.ipynb         # CNN fine-tuning experiment
│   ├── 04_results_visualization.ipynb
│   ├── 05_run8_depth_fusion.ipynb     # Run 8: IR + Depth fusion (best run)
│   └── 06_depth_only_ablation.ipynb   # Depth-only ablation
└── data/                        # Not tracked in git
    ├── kinect_ir/vp{1-15}/      # 30 Kinect IR videos (512x424, 30fps)
    ├── kinect_depth/vp{1-15}/   # 30 Kinect Depth videos (Run 8)
    ├── activities_3s/           # Annotation CSV files
    └── features/                # Extracted .npy features (generated)
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
│   └── vp2/ ... vp15/
├── kinect_depth/                # Required for Run 8 fusion
│   └── vp{1-15}/ ...
└── activities_3s/
    └── kinect_ir/
        ├── midlevel.chunks_90.csv
        ├── midlevel.chunks_90.split_0.train.csv
        ├── midlevel.chunks_90.split_0.val.csv
        └── midlevel.chunks_90.split_0.test.csv
```

### Google Colab

1. Upload `data/` to Google Drive at `MyDrive/DriveAndAct/`.
2. Open the notebooks in `notebooks/` from Colab.
3. Each notebook auto-detects Colab and mounts Drive.

## Usage

### Step 1: Compute Modality Statistics

`02_feature_extraction.ipynb` computes per-modality mean and std on the training videos. These replace ImageNet normalization, which is not appropriate for single-channel IR or Depth frames.

### Step 2: Fine-tune CNN per Modality

`03a_cnn_finetune.ipynb` (or `src/cnn_finetune.py`) fine-tunes ResNet-18 with `freeze_mode="layer4"`, so only the last residual block and a classification head are trained. For Run 8 this is done once for IR and once for Depth, producing two checkpoints.

### Step 3: Extract Features

```bash
python src/feature_extract.py --config configs/config.yaml
```

Runs the fine-tuned CNN over all clips and writes 512-dim features as `.npy` files under `data/features/`. Run once per modality.

### Step 4: Train the Temporal Model

For the IR + Depth fusion run (the project's best result), see `notebooks/05_run8_depth_fusion.ipynb`. It loads pre-extracted features from both modalities, concatenates them to a 1024-dim representation, and trains the BiLSTM with attention pooling.

For the IR-only baseline, see `notebooks/03_lstm_training.ipynb` (Run 7b), which uses the same temporal model on IR features only.

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Frame sampling | 16 frames per chunk; chunks merged (k=2) for ~6s context |
| Frame size | 224x224, modality-specific normalization (computed from data) |
| CNN | ResNet-18, ImageNet pretrained, `layer4` fine-tuned, earlier layers frozen |
| CNN fine-tune | 15 epochs, AdamW, cnn_lr=1e-4, fc_lr=1e-3, weight_decay=1e-4 |
| LSTM | 2-layer BiLSTM, hidden=256, lstm_dropout=0.3, fc_dropout=0.5 |
| Temporal pooling | Attention |
| Batch size | 32 |
| Learning rate | 0.001 (Adam) |
| Loss | CrossEntropyLoss with label smoothing 0.1 |
| Class balancing | WeightedRandomSampler |
| Augmentation | Mixup (alpha=0.2), Gaussian noise (std=0.1) |
| Gradient clipping | 1.0 |
| Early stopping | patience=15 |

## Data Split

Subject-based split (Split 0):

- **Train:** vp1-10 (6559 segments, ~65%)
- **Val:** vp11-12 (1405 segments, ~14%)
- **Test:** vp13-15 (2184 segments, ~21%)

Subject-based evaluation is much stricter than random splits: the model never sees test subjects during training, so a noticeable train/test gap is expected.

## Results

### Systematic Improvement

| Run | Key change | MPCA | Overall | I3D gap |
|---|---|---|---|---|
| Run 3 | Frozen CNN baseline | 39.20% | 44.80% | 25.78 pt |
| Run 6a | CNN fine-tuning (`layer4`) | 47.90% | 61.60% | 17.08 pt |
| Run 7b (k=2) | Chunk merging (6s context) | 52.47% | 66.28% | 12.51 pt |
| **Run 8** | **IR + Depth fusion** | **56.51%** | **63.98%** | **8.47 pt** |

The overall MPCA gain is **+17.31 pt** over the baseline, decomposed cleanly across three independent improvement axes:

1. **Spatial feature quality, +8.70 pt.** Fine-tuning `layer4` of ResNet-18. This was the largest single gain, adapting ImageNet features to single-channel in-cabin imagery.
2. **Temporal context, +4.57 pt.** Chunk merging extends usable context from 3s to 6s with zero added parameters; padding drops from 53% to 6%, so the BiLSTM sees real signal instead of zeros.
3. **Modality complementarity, +4.04 pt.** Fusing IR (texture) and Depth (geometry) gives the BiLSTM evidence that neither modality contains alone.

We optimize mean per-class accuracy (MPCA) as the primary metric because the dataset is heavily imbalanced; overall accuracy is reported for context.

### Comparison with Published Methods

| Method | Type | Input | MPCA |
|---|---|---|---|
| I3D (Carreira & Zisserman, 2017) | 3D CNN (inflated) | Kinect IR+Depth+Color | 68.51% |
| I3D | 3D CNN (inflated) | Kinect IR | 64.98% |
| I3D | 3D CNN (inflated) | Kinect Depth | 60.52% |
| **Ours (Run 8)** | **2D CNN + LSTM** | **Kinect IR+Depth** | **56.51%** |
| **Ours (Run 7b k=2)** | **2D CNN + LSTM** | **Kinect IR** | **52.47%** |
| Three-Stream (Martin et al.) | Body pose | Kinect skeleton | 46.95% |
| P3D (Qiu et al., 2017) | 3D CNN (pseudo) | NIR front-top | 45.32% |
| C3D (Tran et al., 2015) | 3D CNN | NIR front-top | 43.41% |

Our pipeline surpasses every body-pose baseline as well as the C3D and P3D 3D-CNN baselines, while staying lightweight: roughly 4.2M parameters versus I3D's ~12M. The remaining 8.47 pt gap to I3D is the gap to the only method that matches our modality (Kinect IR/Depth).

> Caveats: C3D and P3D use the NIR front-top sensor rather than Kinect; only the I3D rows match our modality. Published Drive&Act numbers average three subject splits, whereas our results use split 0 only.

### Modality Ablation

| Modality | MPCA | Overall |
|---|---|---|
| IR-only (Run 7b k=2) | 52.47% | 66.28% |
| Depth-only (Run 8a) | 45.54% | 57.85% |
| **IR + Depth fusion (Run 8)** | **56.51%** | **63.98%** |

Fusion gain is not just an artifact of added parameters: the two streams are genuinely complementary.

- **IR excels at** texture-dependent activities: hand-object interactions, phone use, reading.
- **Depth excels at** geometry-dependent activities: door operations, entering and exiting the car, body-pose-driven actions.

Per-class examples (recall on the test set, IR-only + Depth-only → Fusion):

- `closing_bottle`: 52% + 0% → 77%.
- `taking_off_sunglasses`: 0% + 80% → 80%.
- `entering_car`: 100% across all three configurations.

## Limitations

1. **Temporal direction confusion.** Activities that differ only in motion direction (opening vs. closing doors, putting on vs. taking off sunglasses) are systematically confused. The 2D CNN encodes per-frame appearance but does not encode motion direction; resolving these pairs would require a 3D CNN or an optical-flow stream.
2. **MPCA vs. overall accuracy tradeoff.** Optimizing for MPCA redistributes model capacity toward rare classes. This is mostly a win for rare-class recall, but a few frequent classes regress: `eating` drops from 56% (Run 7b IR-only) to 26% (Run 8 fusion), confused with `preparing_food` because the two have nearly identical depth geometry.
3. **Structural overfitting from strict evaluation.** A noticeable train/test gap is observed under subject-based splitting. Test drivers are never seen in training and individuals differ in posture, body shape, and movement style. Standard regularization (dropout, weight decay, label smoothing) was applied but cannot fully bridge the subject-domain gap. A random split would leak driver identity into the test set and inflate accuracy without testing real generalization, so this split is both the benchmark standard and the only scientifically valid setup.

## Future Work

1. **Three-fold cross-validation.** Aligns our evaluation with the full Drive&Act protocol and produces directly comparable numbers against published methods.
2. **Transformer-based temporal encoder.** Replacing the BiLSTM with self-attention would model long-range dependencies more flexibly and may close part of the remaining gap to I3D without growing the parameter budget excessively.
3. **Class-conditional adaptive fusion.** A learned per-activity weighting of IR vs. Depth (favoring depth for door operations, IR for phone use) instead of static concatenation, addressing the few classes where one modality is actively harmful for fusion.

## References

1. Martin, M., et al. (2019). *Drive&Act: A Multi-Modal Dataset for Fine-Grained Driver Behavior Recognition in Autonomous Vehicles.* ICCV 2019.
2. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
3. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
4. Zaki, M.J. & Meira, W. (2020). *Data Mining and Machine Learning.* Cambridge University Press.
5. Carreira, J. & Zisserman, A. (2017). *Quo Vadis, Action Recognition?* CVPR 2017.
6. Bahdanau, D., et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
7. Yosinski, J., et al. (2014). *How Transferable Are Features in Deep Neural Networks?* NeurIPS 2014.
8. Tran, D., et al. (2015). *Learning Spatiotemporal Features with 3D Convolutional Networks.* ICCV 2015.
9. Qiu, Z., et al. (2017). *Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks.* ICCV 2017.
