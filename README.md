# HeadGDAU-Net: A Lightweight Hybrid Network with Group-Dilated-Attention and BiLSTM for Prostate mp-MRI Segmentation

Official PyTorch implementation of **HeadGDAU-Net**, as described in the paper:

> **HeadGDAU-Net: A Lightweight Hybrid Network with Group-Dilated-Attention and BiLSTM for Prostate mp-MRI Segmentation**  
> Meili Ren, Mengxing Huang, Yu Zhang, Meiyan Ren, Zhenwei Qiao  
> Hainan University, Shanxi University of Finance and Economics, Shanxi Datong University, Shanxi Provincial Integrated TCM and WMH Hospital

---

## Overview

HeadGDAU-Net addresses three critical challenges in prostate mp-MRI segmentation:
1. **Blurred boundaries** at apex/base via BiLSTM cross-slice modeling + sFDloss shape regularization
2. **Computational efficiency** via grouped-dilated-attention (GDA) with mutual compensation design
3. **Multi-sequence fusion** via channel-wise attention on T2WI + DWI + DCE-MRI

**Key Results:**
- **89.22% DSC** on proprietary dataset (248 patients)
- **26.215M parameters** (38.5% reduction vs. SwinUnet)
- **11.89G FLOPs** (54.5% reduction vs. SwinUnet)
- Generalization validated on PROMISE12, ProstateX, and PI-CAI

---

## Repository Structure

```
headgdau_net_code/
├── model.py              # Complete model implementation
│   ├── HeadNet           # BiLSTM cross-slice context aggregation
│   ├── GDABlock          # Group-Dilated-Attention block
│   ├── GDAUNet           # U-Net backbone with GDA blocks
│   └── HeadGDAUNet       # Complete dual-stream architecture
│
├── sfd_loss.py           # Shape-Sensitive Fourier Descriptor Loss
│   ├── SFDLoss           # Frequency-domain shape regularization
│   └── HybridLoss        # sBFDloss: omega1*L_head + omega2*L_GDA
│
├── train.py              # Training script with 5-fold cross-validation
│   ├── ProstateMRIDataset # Multi-parametric MRI data loader
│   ├── MetricsCalculator # DSC, HD95, AUC-ROC, boundary metrics
│   └── Trainer           # Complete training loop with logging
│
├── inference.py          # Inference engine
│   ├── predict_single    # Single slice with neighbors
│   ├── predict_volume    # Complete 3D volume processing
│   └── benchmark_speed   # Clinical deployment benchmarking
│
├── evaluate.py           # Comprehensive evaluation
│   ├── evaluate_all      # All models vs. all metrics
│   ├── plot_roc_curves   # ROC curve generation
│   ├── plot_pr_curves    # PR curve generation
│   └── statistical_comparison  # t-test + Cohen's d
│
├── requirements.txt      # Python dependencies
├── Dockerfile            # Reproducible environment
├── config.yaml           # Hyperparameter configuration
└── README.md             # This file
```

---



### requirements.txt
```
torch==1.13.1
torchvision==0.14.1
numpy==1.23.5
scipy==1.9.3
scikit-image==0.19.3
scikit-learn==1.1.3
opencv-python==4.6.0.66
matplotlib==3.6.2
pandas==1.5.2
tensorboard==2.11.0
pyyaml==6.0
tqdm==4.64.1
pillow==9.3.0
```

---

## Data Preparation

### Directory Structure
```
data/prostate/
├── patient_001/
│   ├── t2wi/
│   │   ├── slice_000.png
│   │   ├── slice_001.png
│   │   └── ...
│   ├── dwi/
│   │   ├── slice_000.png
│   │   └── ...
│   ├── dce/
│   │   ├── slice_000.png
│   │   └── ...
│   └── mask/
│       ├── slice_000.png
│       └── ...
├── patient_002/
│   └── ...
```

### Quick Start
```bash
# Single fold training
python train.py \
    --data_root ./data/prostate \
    --fold 0 \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.001 \
    --seed 42
```

### 5-Fold Cross-Validation
```bash
# Run all folds
for fold in {0..4}; do
    python train.py --fold $fold --config config.yaml
done
```

### Configuration (config.yaml)
```yaml
# Data
data_root: ./data/prostate
img_size: 256
num_slices: 3

# Model
headnet_hidden_dim: 128
headnet_num_layers: 2
gda_groups: 8
gda_num_heads: 4
gda_window_size: 7

# Loss
omega1: 0.4  # HeadNet loss weight
omega2: 0.6  # GDAU-Net loss weight
sfd_k: 32
sfd_beta: 10.0

# Training
epochs: 200
batch_size: 16
learning_rate: 0.001
weight_decay: 0.0001
n_folds: 5

# Augmentation
rotation_range: 15
flip_prob: 0.5
```

### Training Logs
Training progress is logged to TensorBoard:
```bash
tensorboard --logdir ./logs
```

Access at `http://localhost:6006` to view:
- Loss curves (train/validation)
- DSC, HD95, AUC-ROC evolution
- Boundary metrics (ABD, Boundary DSC)
- Learning rate schedule

---





## Evaluation

### Comprehensive Evaluation
```bash
python evaluate.py \
    --model ./checkpoints/fold_0/best_model.pth \
    --baselines ./baselines/swinunet.pth ./baselines/ma-sam.pth \
    --baseline_names SwinUnet MA-SAM \
    --data ./data/prostate/test \
    --output ./evaluation_results \
    --plot
```

### Outputs
- `quantitative_results.csv`: All metrics for all models
- `roc_curves.png`: ROC curves with AUC values
- `pr_curves.png`: Precision-Recall curves
- `statistical_report.json`: t-test results with Cohen's d

---

## Model Components

### HeadNet (Cross-Slice Context Aggregation)
```python
from model import HeadNet

headnet = HeadNet(
    input_channels=64,
    hidden_dim=128,      # Empirically optimal (Section 3.1.2)
    num_layers=2,
    dropout=0.3
)
```




## Reproducibility

### Random Seeds
All experiments use fixed seeds for reproducibility:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

### Docker Environment
```dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "train.py", "--config", "config.yaml"]
```

Build and run:
```bash
docker build -t headgdau .
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/output:/output headgdau
```




