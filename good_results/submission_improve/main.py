import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import FPRDataset
from model import UNet
from utils import (
    bartlett_window,
    finetune_test_sample,
    plot_loss_curve,
    plot_results,
    train_model,
)


TRAIN_MODE = False
SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"
MODELS_ROOT = SCRIPT_DIR.parent / "models"


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint payload or raw state_dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def build_loaders(data_root, batch_size=16, train_size=5000, val_size=1000):
    """Build train and validation dataloaders from MNIST-based FPR dataset."""
    train_dataset = FPRDataset(mnist_root=str(data_root), train=True, size=128)
    val_dataset = FPRDataset(mnist_root=str(data_root), train=False, size=128)

    train_subset = Subset(train_dataset, list(range(min(train_size, len(train_dataset)))))
    val_subset = Subset(val_dataset, list(range(min(val_size, len(val_dataset)))))

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, val_subset


def run_train(device):
    """Train model and save best checkpoint to best_model_so_far.pth."""
    train_loader, val_loader, _ = build_loaders(data_root=DATA_ROOT)
    model = UNet().to(device)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=20,
        lr=1e-3,
        scheduler_patience=2,
        early_stopping_patience=4,
        save_path=SCRIPT_DIR / "best_model_so_far.pth",
    )
    print(f"Saved best model to: {history['save_path']}")


def run_eval(device):
    """Evaluate 5 random MNIST test samples with and without TTA."""
    _, _, val_subset = build_loaders(data_root=DATA_ROOT)

    model = UNet().to(device)
    weight_candidates = [SCRIPT_DIR / "best_model_so_far.pth", MODELS_ROOT / "best_model_so_far.pth"]
    chosen_path = next((p for p in weight_candidates if p.exists()), None)

    if chosen_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Put best_model_so_far.pth in submission_improve/ or models/."
        )

    load_checkpoint(model, chosen_path, device)
    print(f"Loaded weights from: {chosen_path}")
    checkpoint = torch.load(chosen_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        base_state_dict = checkpoint["model_state_dict"]
    else:
        base_state_dict = checkpoint

    window = bartlett_window(128, device=device)
    criterion = nn.MSELoss()

    n_samples = min(5, len(val_subset))
    random_indices = random.sample(range(len(val_subset)), n_samples)

    metrics_rows = []
    for idx in random_indices:
        model.load_state_dict(base_state_dict)
        model.eval()
        measured_intensity, target = val_subset[idx]
        measured_intensity = measured_intensity.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        with torch.no_grad():
            pretrained_pred = model(measured_intensity)

        mse_pre = criterion(pretrained_pred, target).item()

        finetuned_pred, ft_losses = finetune_test_sample(
            model=model,
            measured_intensity=measured_intensity,
            window=window,
            iterations=200,
            lr=1e-5,
        )

        mse_ft = criterion(finetuned_pred, target).item()
        improvement = (mse_pre - mse_ft) / (mse_pre + 1e-12) * 100.0

        print(
            f"Sample {idx} | MSE Pre: {mse_pre:.6f} | "
            f"MSE FT: {mse_ft:.6f} | Improvement: {improvement:.2f}%"
        )

        plot_results(
            [
                {
                    "sample_id": idx,
                    "target": target,
                    "measured_intensity": measured_intensity,
                    "pretrained_pred": pretrained_pred,
                    "finetuned_pred": finetuned_pred,
                    "mse_pre": mse_pre,
                    "mse_ft": mse_ft,
                }
            ],
            save_path=SCRIPT_DIR / f"results_sample_{idx}.png",
        )
        plot_loss_curve(
            losses=ft_losses,
            title=f"Fine-tuning Loss - Sample {idx}",
            filename=SCRIPT_DIR / f"finetuning_loss_sample_{idx}.png",
        )

        metrics_rows.append(
            {
                "sample_id": idx,
                "mse_pre": mse_pre,
                "mse_ft": mse_ft,
                "improvement_%": improvement,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = SCRIPT_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")


def main():
    """Entry point for training or evaluation."""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if TRAIN_MODE:
        run_train(device)
    else:
        run_eval(device)


if __name__ == "__main__":
    main()
