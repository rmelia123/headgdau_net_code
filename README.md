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

## Installation

### Requirements
- Python 3.9+
- PyTorch 1.13.1
- CUDA 11.7 (for GPU training)
- 12+ GB GPU memory (NVIDIA A100 40GB recommended, RTX 3060 12GB for clinical deployment)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/HeadGDAU-Net.git
cd HeadGDAU-Net

# Create conda environment
conda create -n headgdau python=3.9
conda activate headgdau

# Install dependencies
pip install -r requirements.txt

# Or use Docker for bit-for-bit reproducibility
docker build -t headgdau .
docker run --gpus all -it -v $(pwd)/data:/data headgdau
```

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

### Preprocessing
1. **Spatial registration**: Rigid transformation using Elastix toolbox
2. **Normalization**: Min-max scaling to [0, 1] per modality
3. **Resizing**: 256x256 via bilinear interpolation
4. **Quality control**: Inter-observer agreement DSC > 0.90

---

## Training

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

## Inference

### Single Slice
```bash
python inference.py \
    --model ./checkpoints/fold_0/best_model.pth \
    --input ./data/prostate/patient_001/t2wi/slice_010.png \
    --output ./results \
    --mode single
```

### Complete Volume
```bash
python inference.py \
    --model ./checkpoints/fold_0/best_model.pth \
    --input ./data/prostate/patient_001 \
    --output ./results/patient_001 \
    --mode volume
```

### Speed Benchmark
```bash
python inference.py \
    --model ./checkpoints/fold_0/best_model.pth \
    --benchmark
```

Expected performance on NVIDIA A100:
- **68.7 slices/second** (batch_size=1)
- **<0.6 seconds** for 40-slice prostate MRI

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

### GDA Block (Group-Dilated-Attention)
```python
from model import GDABlock

gda_block = GDABlock(
    in_channels=64,
    out_channels=128,
    groups=8,            # Grid search optimal (Section 3.2.3)
    num_heads=4,         # 4 attention heads
    window_size=7,       # 7x7 windows
    dilation_rates=(1, 2, 3)  # Multi-scale receptive fields
)
```

### sFDloss (Shape-Sensitive Fourier Descriptor Loss)
```python
from sfd_loss import HybridLoss

loss_fn = HybridLoss(
    omega1=0.4,          # HeadNet weight (ablation optimal)
    omega2=0.6,          # GDAU-Net weight
    sfd_k=32,            # Retained Fourier coefficients
    sfd_beta=10.0        # Sigmoid steepness
)
```

---

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

---

## Pre-trained Weights

Download pre-trained weights (5-fold cross-validation):

| Fold | DSC | HD95 (mm) | AUC-ROC | Download |
|------|-----|-----------|---------|----------|
| Fold 0 | 89.45+/-0.82 | 3.21+/-0.24 | 92.78+/-0.91 | [weights](link) |
| Fold 1 | 89.18+/-0.88 | 3.31+/-0.26 | 92.65+/-0.97 | [weights](link) |
| Fold 2 | 89.35+/-0.85 | 3.25+/-0.25 | 92.71+/-0.93 | [weights](link) |
| Fold 3 | 89.12+/-0.90 | 3.35+/-0.28 | 92.58+/-1.02 | [weights](link) |
| Fold 4 | 89.01+/-0.92 | 3.38+/-0.29 | 92.52+/-1.05 | [weights](link) |
| **Mean** | **89.22+/-0.85** | **3.27+/-0.25** | **92.69+/-0.95** | -- |

---

## Results

### Proprietary Dataset (248 patients)

| Model | Params (M) | FLOPs (G) | DSC (%) | HD95 (mm) | AUC-ROC (%) |
|-------|-----------|-----------|---------|-----------|-------------|
| U-Net | 32.08 | 18.25 | 80.67+/-1.36 | 6.82+/-0.53 | 87.31+/-1.06 |
| AttU-Net | 42.35 | 25.63 | 86.15+/-1.49 | 4.07+/-0.74 | 92.23+/-1.43 |
| TransU-Net | 52.31 | 32.87 | 87.43+/-2.03 | 4.12+/-0.52 | 92.15+/-1.72 |
| SwinUnet | 42.60 | 26.15 | 88.21+/-1.03 | 3.85+/-0.36 | 92.40+/-1.27 |
| 3DSqU2Net | 35.22 | 20.34 | 89.05+/-1.09 | 3.45+/-0.28 | 92.57+/-1.14 |
| MA-SAM | 38.76 | 22.68 | 88.92+/-1.17 | 3.62+/-0.31 | 92.53+/-1.19 |
| **HeadGDAU-Net** | **26.22** | **11.89** | **89.22+/-0.85** | **3.27+/-0.25** | **92.69+/-0.95** |

### Public Datasets

| Dataset | HeadGDAU-Net DSC | Best Baseline DSC | Improvement |
|---------|-----------------|-------------------|-------------|
| PROMISE12 | 89.40+/-1.12% | 88.73+/-1.15% (3DSqU2Net) | +0.67% |
| ProstateX | 88.76+/-1.21% | 87.92+/-1.32% (3DSqU2Net) | +0.84% |
| PI-CAI | 87.92+/-1.31% | 87.15+/-1.35% (PI-CAI Top-1) | +0.77% |

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{ren2026headgdau,
  title={HeadGDAU-Net: A Lightweight Hybrid Network with Group-Dilated-Attention and BiLSTM for Prostate mp-MRI Segmentation},
  author={Ren, Meili and Huang, Mengxing and Zhang, Yu and Ren, Meiyan and Qiao, Zhenwei},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```

---

## License

This project is licensed under the MIT License. See LICENSE for details.

---

## Acknowledgments

This work was supported by:
- Key Research and Development Program of Hainan Province (Grant No. ZDYF2021SHFZ243)
- Regional Project of the National Natural Science Foundation of China (Grant No. 82260362)
- Hainan Provincial Key Laboratory of Big Data & Smart Service
- Center of Network and Information Education Technology at Shanxi University of Finance and Economics

We thank all patients who participated in this study and the urologists who contributed to data annotation.

---

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact corresponding author: Meili Ren (renml@sxufe.edu.cn)

---

## Changelog

### v1.0.0 (2026-05-12)
- Initial release
- Complete model implementation
- Training and inference scripts
- Evaluation framework with statistical analysis
- Pre-trained weights (5-fold cross-validation)
- Docker environment for reproducibility
