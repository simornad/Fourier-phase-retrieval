"""
Two-stage training script for:
"Fourier phase retrieval using physics-enhanced deep learning".

This script is intentionally anchored to the repository's official code:
- Network architecture: model/unet.py -> UNet
- Data generation: generate_datasets.py -> mnist_dataset / emnist_dataset
- Physics forward model: model/model.py -> realFFT (with Bartlett window option)

Stage 1 (self-supervised pre-training):
    Train R_theta to map simulated diffraction intensity I = H(O) to object amplitude O.

Stage 2 (physics-driven fine-tuning):
    Initialize from Stage-1 weights and optimize:
        MSE(H(R_theta(I), window=W), I)
    where W is the Bartlett window option implemented in realFFT(window=True).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from generate_datasets import emnist_dataset, mnist_dataset
from model.model import realFFT
from model.unet import UNet


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def enable_cpu_fallback_for_repo_cuda_calls() -> None:
    """
    Repository compatibility patch:
    Several repo modules call `.cuda()` directly inside forward/data code.
    On a CPU-only machine this would crash, so we patch `.cuda()` to no-op.
    """
    if torch.cuda.is_available():
        return

    if getattr(torch.Tensor, "_repo_cuda_patched", False):
        return

    def _tensor_cuda_noop(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        del device, non_blocking, memory_format
        return self

    def _module_cuda_noop(self, device=None):
        del device
        return self

    torch.Tensor.cuda = _tensor_cuda_noop  # type: ignore[assignment]
    torch.Tensor._repo_cuda_patched = True  # type: ignore[attr-defined]
    torch.nn.Module.cuda = _module_cuda_noop  # type: ignore[assignment]
    print("[CPU fallback] Patched repository `.cuda()` calls to no-op.")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """Handle both plain and DataParallel ('module.') checkpoint key formats."""
    if not state_dict:
        return state_dict

    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    model_has_module = model_keys[0].startswith("module.")
    ckpt_has_module = ckpt_keys[0].startswith("module.")

    if ckpt_has_module and not model_has_module:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if not ckpt_has_module and model_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device, strict: bool = False) -> None:
    """Load checkpoint with repo-compatible formats (plain state_dict or dict containing state_dict)."""
    payload = torch.load(str(ckpt_path), map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} is not a valid state_dict format.")

    state_dict = normalize_state_dict_keys(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(
        f"Loaded checkpoint: {ckpt_path}\n"
        f"  missing_keys={len(missing)} unexpected_keys={len(unexpected)} strict={strict}"
    )


def build_dataloaders(
    dataset_name: str,
    data_folder: Path,
    crop_size: int,
    fft_size: int,
    batch_size: int,
    workers: int,
    train_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val dataloaders using repository dataset generation utilities.
    This keeps forward simulation API identical to official code.
    """
    if not torch.cuda.is_available() and workers != 0:
        print("[CPU fallback] Overriding workers to 0 to keep `.cuda()` patch active in data loading.")
        workers = 0

    if dataset_name.lower() == "mnist":
        full_dataset = mnist_dataset(str(data_folder), crop_size=crop_size, fourier_size=fft_size)
    elif dataset_name.lower() == "emnist":
        full_dataset = emnist_dataset(str(data_folder), crop_size=crop_size, fourier_size=fft_size)
    else:
        raise ValueError("dataset_name must be one of: mnist, emnist")

    total_len = len(full_dataset)
    if total_len < 2:
        raise ValueError(f"Dataset is too small for train/val split. Found {total_len} image(s).")

    train_len = max(1, int(total_len * train_split))
    val_len = total_len - train_len
    if val_len == 0:
        val_len = 1
        train_len = total_len - 1

    split_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=split_gen)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        # Keep False to match repo scripts and avoid pinning CUDA tensors produced by dataset code.
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        # Keep False to match repo scripts and avoid pinning CUDA tensors produced by dataset code.
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


def evaluate_stage1(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Validation loss for Stage-1 supervised/self-supervised pretraining."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for I_meas, A_real in loader:
            I_meas = I_meas.to(device=device, dtype=torch.float32)
            A_real = A_real.to(device=device, dtype=torch.float32)
            A_pred = model(I_meas)
            loss = criterion(A_pred, A_real)
            total += loss.item() * I_meas.size(0)
            count += I_meas.size(0)
    return total / max(1, count)


def evaluate_stage2_physics(
    model: nn.Module,
    loader: DataLoader,
    physics_model: realFFT,
    criterion: nn.Module,
    device: torch.device,
    crop_size: int,
    fft_size: int,
    use_window: bool,
) -> float:
    """Validation loss for Stage-2 physics-consistency fine-tuning."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for _, A_real in loader:
            A_real = A_real.to(device=device, dtype=torch.float32)
            I_meas_full, _ = physics_model(
                A_real, torch.ones_like(A_real), out_size=fft_size, window=use_window
            )
            I_input = F.interpolate(
                I_meas_full.to(torch.float32),
                size=(crop_size, crop_size),
                mode="bilinear",
                align_corners=False,
            )
            A_pred = model(I_input)
            I_pred_full, _ = physics_model(
                A_pred, torch.ones_like(A_pred), out_size=fft_size, window=use_window
            )
            loss = criterion(I_pred_full.to(torch.float32), I_meas_full.to(torch.float32))
            total += loss.item() * A_real.size(0)
            count += A_real.size(0)
    return total / max(1, count)


def save_stage_sample_plot(
    out_path: Path,
    model: nn.Module,
    sample: Tuple[torch.Tensor, torch.Tensor],
    physics_model: realFFT,
    device: torch.device,
    crop_size: int,
    fft_size: int,
    use_window: bool,
    stage_name: str,
) -> None:
    """
    Save a visualization with:
    GT amplitude, network input diffraction, predicted amplitude, predicted diffraction.
    """
    model.eval()
    I_dataset, A_real = sample
    del I_dataset  # Stage-2 sample uses freshly simulated diffraction from the physical model.

    with torch.no_grad():
        A_real = A_real.to(device=device, dtype=torch.float32)
        I_meas_full, _ = physics_model(A_real, torch.ones_like(A_real), out_size=fft_size, window=use_window)
        I_input = F.interpolate(
            I_meas_full.to(torch.float32),
            size=(crop_size, crop_size),
            mode="bilinear",
            align_corners=False,
        )
        A_pred = model(I_input)
        I_pred_full, _ = physics_model(A_pred, torch.ones_like(A_pred), out_size=fft_size, window=use_window)

    # Save a direct reconstruction image (hard requirement: sample reconstruction output).
    save_image(A_pred[0].detach().cpu(), str(out_path.parent / f"{stage_name}_sample_reconstruction.png"))

    # Save a compact diagnostic figure.
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(A_real[0, 0].detach().cpu().numpy(), cmap="gray")
    axes[0].set_title("GT amplitude O")
    axes[0].axis("off")

    axes[1].imshow(I_input[0, 0].detach().cpu().numpy(), cmap="magma")
    axes[1].set_title("Input I (resized)")
    axes[1].axis("off")

    axes[2].imshow(A_pred[0, 0].detach().cpu().numpy(), cmap="gray")
    axes[2].set_title("Prediction R_theta(I)")
    axes[2].axis("off")

    axes[3].imshow(I_pred_full[0, 0].detach().cpu().numpy(), cmap="magma")
    axes[3].set_title("Forward H(R_theta(I))")
    axes[3].axis("off")

    fig.suptitle(f"{stage_name} sample visualization", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


def save_loss_curve(
    out_path: Path,
    stage1_train: List[float],
    stage1_val: List[float],
    stage2_train: List[float],
    stage2_val: List[float],
) -> None:
    """Save one combined loss curve figure for both training stages."""
    plt.figure(figsize=(10, 5))

    x1 = np.arange(1, len(stage1_train) + 1)
    x2 = np.arange(1, len(stage2_train) + 1) + len(stage1_train)

    plt.plot(x1, stage1_train, label="Stage1 train (MSE: A_pred vs A_real)")
    plt.plot(x1, stage1_val, label="Stage1 val")
    plt.plot(x2, stage2_train, label="Stage2 train (Physics MSE)")
    plt.plot(x2, stage2_val, label="Stage2 val")

    plt.xlabel("Epoch (Stage1 + Stage2)")
    plt.ylabel("Loss")
    plt.title("Two-stage training loss curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=180)
    plt.close()


def train_stage1_pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    ckpt_dir: Path,
) -> Dict[str, List[float]]:
    """
    Stage-1 training:
    Supervised/self-supervised mapping from simulated diffraction I to amplitude O.
    """
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, cooldown=2
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")

    best_ckpt = ckpt_dir / "stage1_best.pth.tar"
    last_ckpt = ckpt_dir / "stage1_last.pth.tar"

    print("\n========== Stage 1: Self-supervised pre-training ==========")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        seen = 0

        for batch_idx, (I_meas, A_real) in enumerate(train_loader):
            # Data to device.
            I_meas = I_meas.to(device=device, dtype=torch.float32)
            A_real = A_real.to(device=device, dtype=torch.float32)

            # Forward pass: R_theta(I).
            optimizer.zero_grad(set_to_none=True)
            A_pred = model(I_meas)

            # Stage-1 loss: reconstruction target in amplitude domain.
            loss = criterion(A_pred, A_real)

            # Backpropagation + optimizer step.
            loss.backward()
            optimizer.step()

            running += loss.item() * I_meas.size(0)
            seen += I_meas.size(0)

            # Print important tensor shapes.
            if batch_idx == 0:
                print(
                    f"[Stage1][Epoch {epoch + 1}] "
                    f"I_meas={tuple(I_meas.shape)} A_real={tuple(A_real.shape)} A_pred={tuple(A_pred.shape)}"
                )

        train_loss = running / max(1, seen)
        val_loss = evaluate_stage1(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Checkpointing: save best + last in repository-compatible state_dict format.
        torch.save(model.state_dict(), str(last_ckpt))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(best_ckpt))

        print(
            f"[Stage1][Epoch {epoch + 1}/{epochs}] "
            f"train={train_loss:.6e} val={val_loss:.6e} best_val={best_val:.6e} "
            f"time={time.time() - t0:.2f}s"
        )

    return {"train_loss": train_losses, "val_loss": val_losses}


def train_stage2_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    crop_size: int,
    fft_size: int,
    use_window: bool,
    ckpt_dir: Path,
) -> Dict[str, List[float]]:
    """
    Stage-2 training:
    Physics-consistency fine-tuning from Stage-1 weights.
    """
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, cooldown=2
    )

    physics_model = realFFT()

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")

    best_ckpt = ckpt_dir / "stage2_best.pth"
    last_ckpt = ckpt_dir / "stage2_last.pth"

    print("\n========== Stage 2: Physics-driven fine-tuning ==========")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        seen = 0

        for batch_idx, (_, A_real) in enumerate(train_loader):
            # Ground-truth object amplitude from dataset (used to synthesize measured diffraction).
            A_real = A_real.to(device=device, dtype=torch.float32)

            # Generate physically consistent measured intensity with official forward model H(.).
            with torch.no_grad():
                I_meas_full, _ = physics_model(
                    A_real, torch.ones_like(A_real), out_size=fft_size, window=use_window
                )
                # Resize to network input size, as done in repo scripts.
                I_input = F.interpolate(
                    I_meas_full.to(torch.float32),
                    size=(crop_size, crop_size),
                    mode="bilinear",
                    align_corners=False,
                )

            optimizer.zero_grad(set_to_none=True)

            # Network reconstruction R_theta(I).
            A_pred = model(I_input)

            # Physics consistency:
            # H(R_theta(I), window=W) ~= I_measured
            I_pred_full, _ = physics_model(
                A_pred, torch.ones_like(A_pred), out_size=fft_size, window=use_window
            )
            loss_physics = criterion(I_pred_full.to(torch.float32), I_meas_full.to(torch.float32))

            # Backpropagation + optimizer step.
            loss_physics.backward()
            optimizer.step()

            running += loss_physics.item() * A_real.size(0)
            seen += A_real.size(0)

            # Print important tensor shapes.
            if batch_idx == 0:
                print(
                    f"[Stage2][Epoch {epoch + 1}] "
                    f"I_meas_full={tuple(I_meas_full.shape)} I_input={tuple(I_input.shape)} "
                    f"A_pred={tuple(A_pred.shape)} I_pred_full={tuple(I_pred_full.shape)}"
                )

        train_loss = running / max(1, seen)
        val_loss = evaluate_stage2_physics(
            model=model,
            loader=val_loader,
            physics_model=physics_model,
            criterion=criterion,
            device=device,
            crop_size=crop_size,
            fft_size=fft_size,
            use_window=use_window,
        )
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Checkpointing: save best + last in repository-compatible state_dict format.
        torch.save(model.state_dict(), str(last_ckpt))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(best_ckpt))

        print(
            f"[Stage2][Epoch {epoch + 1}/{epochs}] "
            f"train={train_loss:.6e} val={val_loss:.6e} best_val={best_val:.6e} "
            f"time={time.time() - t0:.2f}s"
        )

    return {"train_loss": train_losses, "val_loss": val_losses}


def parse_args() -> argparse.Namespace:
    """CLI configuration."""
    parser = argparse.ArgumentParser(description="Two-stage FPR training (repo-anchored)")
    parser.add_argument("--dataset-name", type=str, default="mnist", choices=["mnist", "emnist"])
    parser.add_argument("--data-folder", type=str, default="train_MINST_128")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--fft-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train-split", type=float, default=0.9)

    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--disable-window", action="store_true", help="Disable Bartlett window in realFFT")

    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="",
        help="Optional initial checkpoint before Stage-1 (e.g., models/checkpoint/checkpoint_Unet_MNIST_v5window.pth.tar)",
    )
    parser.add_argument(
        "--stage2-init-ckpt",
        type=str,
        default="",
        help="Optional checkpoint to start Stage-2 (if empty, uses Stage-1 best checkpoint).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/two_stage_train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1) Reproducibility + hardware setup.
    # -------------------------------------------------------------------------
    enable_cpu_fallback_for_repo_cuda_calls()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2) Path checks and output directories.
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_folder = (base_dir / args.data_folder).resolve()
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    init_ckpt = (base_dir / args.init_ckpt).resolve() if args.init_ckpt else None
    stage2_init_ckpt = (base_dir / args.stage2_init_ckpt).resolve() if args.stage2_init_ckpt else None
    if init_ckpt is not None and not init_ckpt.exists():
        raise FileNotFoundError(f"init-ckpt not found: {init_ckpt}")
    if stage2_init_ckpt is not None and not stage2_init_ckpt.exists():
        raise FileNotFoundError(f"stage2-init-ckpt not found: {stage2_init_ckpt}")

    output_dir = (base_dir / args.output_dir).resolve()
    ckpt_dir = output_dir / "checkpoints"
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 3) Build dataset and dataloaders using repository dataset generator.
    # -------------------------------------------------------------------------
    train_loader, val_loader = build_dataloaders(
        dataset_name=args.dataset_name,
        data_folder=data_folder,
        crop_size=args.crop_size,
        fft_size=args.fft_size,
        batch_size=args.batch_size,
        workers=args.workers,
        train_split=args.train_split,
        seed=args.seed,
    )

    print(
        f"Dataset prepared: train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"batch_size={args.batch_size}"
    )

    # Fixed sample for before/after visualization across both stages.
    fixed_sample = next(iter(val_loader))

    # -------------------------------------------------------------------------
    # 4) Initialize model architecture exactly as in repo.
    # -------------------------------------------------------------------------
    net = UNet(in_channels=1).to(device)
    if init_ckpt is not None:
        load_checkpoint(net, init_ckpt, device=device, strict=False)

    # -------------------------------------------------------------------------
    # 5) Stage-1: pre-training on (I, O) pairs from the repo's forward model.
    # -------------------------------------------------------------------------
    stage1_logs = train_stage1_pretrain(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        ckpt_dir=ckpt_dir,
    )

    # Save Stage-1 sample output.
    physics_model = realFFT()
    save_stage_sample_plot(
        out_path=image_dir / "stage1_sample_panel.png",
        model=net,
        sample=fixed_sample,
        physics_model=physics_model,
        device=device,
        crop_size=args.crop_size,
        fft_size=args.fft_size,
        use_window=not args.disable_window,
        stage_name="stage1",
    )

    # -------------------------------------------------------------------------
    # 6) Stage-2 initialization from Stage-1 (or provided checkpoint).
    # -------------------------------------------------------------------------
    if stage2_init_ckpt is not None:
        load_checkpoint(net, stage2_init_ckpt, device=device, strict=False)
    else:
        load_checkpoint(net, ckpt_dir / "stage1_best.pth.tar", device=device, strict=False)

    # -------------------------------------------------------------------------
    # 7) Stage-2: physics consistency fine-tuning.
    # -------------------------------------------------------------------------
    stage2_logs = train_stage2_finetune(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        crop_size=args.crop_size,
        fft_size=args.fft_size,
        use_window=not args.disable_window,
        ckpt_dir=ckpt_dir,
    )

    # Save Stage-2 sample output.
    save_stage_sample_plot(
        out_path=image_dir / "stage2_sample_panel.png",
        model=net,
        sample=fixed_sample,
        physics_model=physics_model,
        device=device,
        crop_size=args.crop_size,
        fft_size=args.fft_size,
        use_window=not args.disable_window,
        stage_name="stage2",
    )

    # -------------------------------------------------------------------------
    # 8) Save combined loss curve + compact JSON log.
    # -------------------------------------------------------------------------
    loss_curve_path = output_dir / "loss_curve.png"
    save_loss_curve(
        out_path=loss_curve_path,
        stage1_train=stage1_logs["train_loss"],
        stage1_val=stage1_logs["val_loss"],
        stage2_train=stage2_logs["train_loss"],
        stage2_val=stage2_logs["val_loss"],
    )

    log_dict = {
        "config": {
            "dataset_name": args.dataset_name,
            "data_folder": str(data_folder),
            "crop_size": args.crop_size,
            "fft_size": args.fft_size,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "train_split": args.train_split,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "pretrain_lr": args.pretrain_lr,
            "finetune_lr": args.finetune_lr,
            "window_enabled": not args.disable_window,
            "seed": args.seed,
            "device": str(device),
        },
        "stage1": stage1_logs,
        "stage2": stage2_logs,
    }
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2)

    print("\nTraining complete.")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Sample images: {image_dir}")
    print(f"Loss curve: {loss_curve_path}")
    print(f"JSON log: {output_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
