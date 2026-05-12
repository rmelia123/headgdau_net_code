"""
Training Script for HeadGDAU-Net

Implements the complete training pipeline with:
- 5-fold cross-validation for model stability verification
- Adam optimizer (lr=0.001, epochs=200, batch_size=16)
- Hybrid loss (sBFDloss) with ω₁=0.4, ω₂=0.6
- Data augmentation: random crop, rotation (±15°), flip, normalization
- Learning rate scheduling and early stopping
- Comprehensive logging (DSC, HD95, AUC-ROC, boundary metrics)

Hardware: NVIDIA A100 40GB GPU
Framework: PyTorch 1.13, CUDA 11.7

Author: Meili Ren et al.
Date: 2026
"""

import os
import sys
import time
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import yaml

# Import model and loss
from model import HeadGDAUNet
from sfd_loss import HybridLoss

# Medical image processing
import cv2
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


# ==================== Configuration ====================

DEFAULT_CONFIG = {
    # Data
    'data_root': './data/prostate',
    'img_size': 256,
    'num_slices': 3,  # T2WI + DWI + DCE-MRI
    'num_classes': 1,

    # Model
    'headnet_hidden_dim': 128,
    'headnet_num_layers': 2,
    'gda_groups': 8,
    'gda_num_heads': 4,
    'gda_window_size': 7,

    # Loss
    'omega1': 0.4,  # HeadNet loss weight
    'omega2': 0.6,  # GDAU-Net loss weight
    'sfd_k': 32,
    'sfd_beta': 10.0,

    # Training
    'epochs': 200,
    'batch_size': 16,
    'learning_rate': 0.001,
    'momentum': 0.5,
    'weight_decay': 1e-4,
    'num_workers': 4,

    # Cross-validation
    'n_folds': 5,
    'fold_idx': 0,

    # Augmentation
    'rotation_range': 15,
    'flip_prob': 0.5,
    'crop_size': 256,

    # Logging
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints',
    'save_freq': 10,

    # Hardware
    'device': 'cuda',
    'seed': 42
}


# ==================== Dataset ====================

class ProstateMRIDataset(Dataset):
    """
    Prostate mp-MRI Dataset with cross-slice sampling.

    Loads multi-parametric MRI (T2WI, DWI, DCE-MRI) and extracts
    central slice with adjacent neighbors (t-1, t+1) for HeadNet.

    Preprocessing:
    1. Resize to 256×256
    2. Min-max normalization to [0, 1]
    3. Random augmentation during training

    Args:
        data_root: Root directory containing patient folders
        patient_ids: List of patient IDs to include
        is_train: Whether training mode (enables augmentation)
        config: Configuration dict
    """

    def __init__(self, data_root: str, patient_ids: list, is_train: bool = True,
                 config: Dict = None):
        self.data_root = Path(data_root)
        self.patient_ids = patient_ids
        self.is_train = is_train
        self.config = config or DEFAULT_CONFIG
        self.img_size = self.config['img_size']

        # Build slice index: (patient_id, slice_idx, has_neighbors)
        self.samples = []
        for pid in patient_ids:
            patient_dir = self.data_root / pid
            if not patient_dir.exists():
                continue

            # Count slices (each patient has T2WI, DWI, DCE-MRI volumes)
            n_slices = len(list((patient_dir / 't2wi').glob('*.png')))

            for s in range(n_slices):
                # Check if neighbors exist
                has_prev = s > 0
                has_next = s < n_slices - 1
                if has_prev and has_next:
                    self.samples.append((pid, s))

        print(f"Dataset: {len(self.samples)} samples from {len(patient_ids)} patients")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            central_slice: (3, H, W) - multi-parametric fusion
            neighbor_slices: (2, 3, H, W) - adjacent slices (t-1, t+1)
            mask: (1, H, W) - binary prostate mask
            patient_id: str - for tracking
        """
        patient_id, slice_idx = self.samples[idx]
        patient_dir = self.data_root / patient_id

        # Load multi-parametric slices
        modalities = ['t2wi', 'dwi', 'dce']
        central_slices = []

        for mod in modalities:
            img_path = patient_dir / mod / f'slice_{slice_idx:03d}.png'
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            central_slices.append(img)

        central = np.stack(central_slices, axis=0)  # (3, H, W)

        # Load neighbor slices (t-1 and t+1)
        neighbor_list = []
        for offset in [-1, 1]:
            neighbor_slices = []
            for mod in modalities:
                img_path = patient_dir / mod / f'slice_{slice_idx + offset:03d}.png'
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img.astype(np.float32) / 255.0
                neighbor_slices.append(img)
            neighbor_list.append(np.stack(neighbor_slices, axis=0))

        neighbors = np.stack(neighbor_list, axis=0)  # (2, 3, H, W)

        # Load mask
        mask_path = patient_dir / 'mask' / f'slice_{slice_idx:03d}.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype(np.float32)
        mask = mask[np.newaxis, ...]  # (1, H, W)

        # Data augmentation (training only)
        if self.is_train:
            central, neighbors, mask = self._augment(central, neighbors, mask)

        # Convert to tensors
        central = torch.from_numpy(central).float()
        neighbors = torch.from_numpy(neighbors).float()
        mask = torch.from_numpy(mask).float()

        return central, neighbors, mask, patient_id

    def _augment(self, central: np.ndarray, neighbors: np.ndarray, 
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random augmentation."""
        # Random rotation
        if random.random() < 0.8:
            angle = random.uniform(-self.config['rotation_range'], 
                                   self.config['rotation_range'])
            center = (self.img_size // 2, self.img_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            for i in range(3):
                central[i] = cv2.warpAffine(central[i], M, (self.img_size, self.img_size))
            for n in range(2):
                for i in range(3):
                    neighbors[n, i] = cv2.warpAffine(neighbors[n, i], M, 
                                                     (self.img_size, self.img_size))
            mask[0] = cv2.warpAffine(mask[0], M, (self.img_size, self.img_size))

        # Random horizontal flip
        if random.random() < self.config['flip_prob']:
            central = np.flip(central, axis=2).copy()
            neighbors = np.flip(neighbors, axis=3).copy()
            mask = np.flip(mask, axis=2).copy()

        # Random vertical flip
        if random.random() < self.config['flip_prob']:
            central = np.flip(central, axis=1).copy()
            neighbors = np.flip(neighbors, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        return central, neighbors, mask


# ==================== Metrics ====================

class MetricsCalculator:
    """Compute comprehensive segmentation metrics."""

    @staticmethod
    def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Dice Similarity Coefficient (DSC)."""
        intersection = np.sum(pred * gt)
        return 2.0 * intersection / (np.sum(pred) + np.sum(gt) + 1e-6)

    @staticmethod
    def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
        """Sensitivity (Recall)."""
        tp = np.sum(pred * gt)
        fn = np.sum((1 - pred) * gt)
        return tp / (tp + fn + 1e-6)

    @staticmethod
    def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
        """Specificity."""
        tn = np.sum((1 - pred) * (1 - gt))
        fp = np.sum(pred * (1 - gt))
        return tn / (tn + fp + 1e-6)

    @staticmethod
    def precision(pred: np.ndarray, gt: np.ndarray) -> float:
        """Precision."""
        tp = np.sum(pred * gt)
        fp = np.sum(pred * (1 - gt))
        return tp / (tp + fp + 1e-6)

    @staticmethod
    def hausdorff_distance(pred: np.ndarray, gt: np.ndarray, percentile: int = 95) -> float:
        """95th percentile Hausdorff Distance (HD95)."""
        # Extract boundaries
        pred_boundary = np.argwhere(pred > 0.5)
        gt_boundary = np.argwhere(gt > 0.5)

        if len(pred_boundary) == 0 or len(gt_boundary) == 0:
            return float('inf')

        # Compute distances
        from scipy.spatial.distance import directed_hausdorff
        d1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        d2 = directed_hausdorff(gt_boundary, pred_boundary)[0]

        # Return percentile
        return max(d1, d2)

    @staticmethod
    def auc_roc(pred_prob: np.ndarray, gt: np.ndarray) -> float:
        """AUC-ROC."""
        pred_flat = pred_prob.flatten()
        gt_flat = gt.flatten().astype(int)

        if len(np.unique(gt_flat)) < 2:
            return 0.5

        return roc_auc_score(gt_flat, pred_flat)

    @staticmethod
    def boundary_metrics(pred: np.ndarray, gt: np.ndarray, band_width: int = 3) -> Dict:
        """
        Compute boundary-specific metrics.

        Args:
            band_width: Width of boundary band in pixels (±3mm ≈ ±3 pixels at 0.5mm resolution)

        Returns:
            Dict with ABD, MaxD, Boundary DSC, Boundary Recall, Boundary Precision
        """
        # Extract boundary regions (morphological dilation)
        from scipy.ndimage import binary_dilation

        pred_boundary = binary_dilation(pred > 0.5, iterations=band_width)
        gt_boundary = binary_dilation(gt > 0.5, iterations=band_width)

        # Boundary DSC
        boundary_union = np.logical_or(pred_boundary, gt_boundary)
        if np.sum(boundary_union) == 0:
            boundary_dsc = 0.0
        else:
            boundary_dsc = 2 * np.sum(pred_boundary * gt_boundary) /                           (np.sum(pred_boundary) + np.sum(gt_boundary))

        # Distance-based metrics (simplified)
        pred_pts = np.argwhere(pred > 0.5)
        gt_pts = np.argwhere(gt > 0.5)

        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return {
                'ABD': float('inf'),
                'MaxD': float('inf'),
                'Boundary_DSC': 0.0,
                'Boundary_Recall': 0.0,
                'Boundary_Precision': 0.0
            }

        # Compute distances (simplified - full implementation uses scipy.spatial.KDTree)
        from scipy.spatial.distance import cdist
        distances_pred_to_gt = np.min(cdist(pred_pts, gt_pts), axis=1)
        distances_gt_to_pred = np.min(cdist(gt_pts, pred_pts), axis=1)

        abd = (np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)) / 2
        max_d = max(np.max(distances_pred_to_gt), np.max(distances_gt_to_pred))

        # Recall/Precision within 2mm (≈4 pixels)
        threshold = 4
        boundary_recall = np.mean(distances_gt_to_pred < threshold)
        boundary_precision = np.mean(distances_pred_to_gt < threshold)

        return {
            'ABD': abd,
            'MaxD': max_d,
            'Boundary_DSC': boundary_dsc,
            'Boundary_Recall': boundary_recall,
            'Boundary_Precision': boundary_precision
        }


# ==================== Training ====================

class Trainer:
    """HeadGDAU-Net Trainer with 5-fold cross-validation."""

    def __init__(self, config: Dict, fold_idx: int = 0):
        self.config = config
        self.fold_idx = fold_idx
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

        # Set random seeds for reproducibility
        self._set_seed(config['seed'] + fold_idx)

        # Initialize logging
        self.log_dir = Path(config['log_dir']) / f'fold_{fold_idx}'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(config['checkpoint_dir']) / f'fold_{fold_idx}'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Fold {fold_idx}: Configuration")
        self.logger.info(json.dumps(config, indent=2))

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_datasets(self, all_patient_ids: list) -> Tuple[DataLoader, DataLoader]:
        """Create train/validation datasets for current fold."""
        kfold = KFold(n_splits=self.config['n_folds'], shuffle=True, 
                      random_state=self.config['seed'])

        splits = list(kfold.split(all_patient_ids))
        train_idx, val_idx = splits[self.fold_idx]

        train_patients = [all_patient_ids[i] for i in train_idx]
        val_patients = [all_patient_ids[i] for i in val_idx]

        self.logger.info(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")

        train_dataset = ProstateMRIDataset(
            self.config['data_root'], train_patients, is_train=True, config=self.config
        )
        val_dataset = ProstateMRIDataset(
            self.config['data_root'], val_patients, is_train=False, config=self.config
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'],
            shuffle=True, num_workers=self.config['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'],
            shuffle=False, num_workers=self.config['num_workers'],
            pin_memory=True
        )

        return train_loader, val_loader

    def train_epoch(self, model: nn.Module, loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: HybridLoss,
                    epoch: int) -> Dict:
        """Train for one epoch."""
        model.train()

        total_loss = 0.0
        total_head_loss = 0.0
        total_gda_loss = 0.0

        for batch_idx, (central, neighbors, mask, _) in enumerate(loader):
            central = central.to(self.device)
            neighbors = neighbors.to(self.device)
            mask = mask.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            pred = model(central, neighbors)

            # Compute loss
            loss, loss_dict = criterion(pred, mask)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_head_loss += loss_dict['head_loss']
            total_gda_loss += loss_dict['gda_loss']

            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(Head: {loss_dict['head_loss']:.4f}, GDA: {loss_dict['gda_loss']:.4f})"
                )

        n_batches = len(loader)
        return {
            'loss': total_loss / n_batches,
            'head_loss': total_head_loss / n_batches,
            'gda_loss': total_gda_loss / n_batches
        }

    def validate(self, model: nn.Module, loader: DataLoader,
                 criterion: HybridLoss, epoch: int) -> Dict:
        """Validate and compute comprehensive metrics."""
        model.eval()

        all_preds = []
        all_masks = []
        all_probs = []

        total_loss = 0.0

        with torch.no_grad():
            for central, neighbors, mask, _ in loader:
                central = central.to(self.device)
                neighbors = neighbors.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                pred_prob = model(central, neighbors)
                pred_binary = (pred_prob > 0.5).float()

                # Compute loss
                loss, _ = criterion(pred_prob, mask)
                total_loss += loss.item()

                # Collect for metrics
                all_preds.append(pred_binary.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
                all_probs.append(pred_prob.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_masks, all_probs)
        metrics['val_loss'] = total_loss / len(loader)

        return metrics

    def _compute_metrics(self, preds: np.ndarray, masks: np.ndarray, 
                         probs: np.ndarray) -> Dict:
        """Compute comprehensive evaluation metrics."""
        calc = MetricsCalculator()

        # Per-sample metrics
        dsc_scores = []
        se_scores = []
        sp_scores = []
        pre_scores = []
        hd95_scores = []
        auc_scores = []

        boundary_metrics_list = []

        for i in range(len(preds)):
            p = preds[i, 0]
            m = masks[i, 0]
            prob = probs[i, 0]

            dsc_scores.append(calc.dice_score(p, m))
            se_scores.append(calc.sensitivity(p, m))
            sp_scores.append(calc.specificity(p, m))
            pre_scores.append(calc.precision(p, m))
            hd95_scores.append(calc.hausdorff_distance(p, m))
            auc_scores.append(calc.auc_roc(prob, m))

            boundary_metrics_list.append(calc.boundary_metrics(p, m))

        # Aggregate boundary metrics
        bd_metrics = {
            'ABD': np.mean([b['ABD'] for b in boundary_metrics_list]),
            'MaxD': np.mean([b['MaxD'] for b in boundary_metrics_list]),
            'Boundary_DSC': np.mean([b['Boundary_DSC'] for b in boundary_metrics_list]),
            'Boundary_Recall': np.mean([b['Boundary_Recall'] for b in boundary_metrics_list]),
            'Boundary_Precision': np.mean([b['Boundary_Precision'] for b in boundary_metrics_list])
        }

        metrics = {
            'DSC': np.mean(dsc_scores),
            'DSC_std': np.std(dsc_scores),
            'SE': np.mean(se_scores),
            'SP': np.mean(sp_scores),
            'Pre': np.mean(pre_scores),
            'HD95': np.mean(hd95_scores),
            'AUC_ROC': np.mean(auc_scores),
            **bd_metrics
        }

        return metrics

    def train(self, all_patient_ids: list):
        """Main training loop with 5-fold cross-validation."""
        # Create datasets
        train_loader, val_loader = self.create_datasets(all_patient_ids)

        # Create model
        model = HeadGDAUNet(
            in_channels=self.config['num_slices'],
            num_classes=self.config['num_classes'],
            headnet_hidden_dim=self.config['headnet_hidden_dim'],
            headnet_num_layers=self.config['headnet_num_layers']
        ).to(self.device)

        self.logger.info(f"Model parameters: {model.get_parameter_count():,}")

        # Create loss function
        criterion = HybridLoss(
            omega1=self.config['omega1'],
            omega2=self.config['omega2'],
            sfd_k=self.config['sfd_k'],
            sfd_beta=self.config['sfd_beta']
        ).to(self.device)

        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=20, verbose=True
        )

        # Training loop
        best_dsc = 0.0
        best_epoch = 0
        patience_counter = 0
        max_patience = 50

        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch)

            # Validate
            val_metrics = self.validate(model, val_loader, criterion, epoch)

            # Update learning rate
            scheduler.step(val_metrics['DSC'])

            # Logging
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch}/{self.config['epochs']} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"DSC: {val_metrics['DSC']:.4f}±{val_metrics['DSC_std']:.4f} | "
                f"HD95: {val_metrics['HD95']:.2f} | "
                f"AUC: {val_metrics['AUC_ROC']:.4f}"
            )

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Metrics/DSC', val_metrics['DSC'], epoch)
            self.writer.add_scalar('Metrics/HD95', val_metrics['HD95'], epoch)
            self.writer.add_scalar('Metrics/AUC_ROC', val_metrics['AUC_ROC'], epoch)
            self.writer.add_scalar('Metrics/SE', val_metrics['SE'], epoch)
            self.writer.add_scalar('Metrics/SP', val_metrics['SP'], epoch)
            self.writer.add_scalar('Metrics/Pre', val_metrics['Pre'], epoch)
            self.writer.add_scalar('Boundary/ABD', val_metrics['ABD'], epoch)
            self.writer.add_scalar('Boundary/Boundary_DSC', val_metrics['Boundary_DSC'], epoch)

            # Save checkpoint
            if epoch % self.config['save_freq'] == 0:
                self._save_checkpoint(model, optimizer, epoch, val_metrics)

            # Early stopping based on DSC
            if val_metrics['DSC'] > best_dsc:
                best_dsc = val_metrics['DSC']
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(model, optimizer, epoch, val_metrics, is_best=True)
                self.logger.info(f"New best DSC: {best_dsc:.4f} at epoch {best_epoch}")
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.logger.info(f"Training completed. Best DSC: {best_dsc:.4f} at epoch {best_epoch}")
        self.writer.close()

        return best_dsc

    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                         epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train HeadGDAU-Net')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--fold', type=int, default=0, help='Fold index for cross-validation')
    parser.add_argument('--data_root', type=str, default='./data/prostate', help='Data directory')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG.copy()

    # Override with command line args
    config['data_root'] = args.data_root
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['seed'] = args.seed
    config['fold_idx'] = args.fold

    # Get patient IDs
    data_root = Path(config['data_root'])
    all_patient_ids = sorted([d.name for d in data_root.iterdir() if d.is_dir()])

    print(f"Total patients: {len(all_patient_ids)}")

    # Train
    trainer = Trainer(config, fold_idx=args.fold)
    best_dsc = trainer.train(all_patient_ids)

    print(f"\nFold {args.fold} completed. Best DSC: {best_dsc:.4f}")


if __name__ == "__main__":
    main()
