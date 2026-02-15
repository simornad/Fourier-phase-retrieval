"""Physics and fine-tuning utilities for V1 pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


def bartlett_window(size: int, device: torch.device) -> torch.Tensor:
    """Builds a 2D Bartlett window for spectral leakage reduction."""
    window_1d = torch.bartlett_window(size, device=device)
    return window_1d.unsqueeze(1) * window_1d.unsqueeze(0)


def physics_forward(image: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Calculates normalized log-intensity from an image prediction."""
    if image.dim() == 4:
        windowed = image[:, 0] * window.unsqueeze(0)
        fourier = torch.fft.fft2(windowed)
        shifted = torch.fft.fftshift(fourier, dim=(-2, -1))
        intensity = torch.abs(shifted) ** 2
        log_intensity = torch.log1p(intensity)
        norm = log_intensity / 20.0
        return torch.clamp(norm, 0.0, 1.0).unsqueeze(1)

    if image.dim() == 3:
        windowed = image[0] * window
        fourier = torch.fft.fft2(windowed)
        shifted = torch.fft.fftshift(fourier, dim=(-2, -1))
        intensity = torch.abs(shifted) ** 2
        log_intensity = torch.log1p(intensity)
        norm = log_intensity / 20.0
        return torch.clamp(norm, 0.0, 1.0).unsqueeze(0)

    raise ValueError("Unsupported image shape for physics_forward")


def finetune_test_sample(
    model: nn.Module,
    measured_intensity: torch.Tensor,
    window: torch.Tensor,
    iterations: int = 50,
    lr: float = 1e-5,
    lambda_tv: float = 1e-5,
) -> tuple[torch.Tensor, list[float]]:
    """Performs V1 test-time adaptation with partial encoder freezing."""
    for name, param in model.named_parameters():
        if "enc" in name or "pool" in name or "bottleneck" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.eval()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for _ in range(iterations):
        optimizer.zero_grad()
        pred_image = model(measured_intensity)
        pred_intensity = physics_forward(pred_image, window)

        img1 = pred_intensity[0, 0].detach().cpu().float().numpy()
        img2 = measured_intensity[0, 0].detach().cpu().float().numpy()

        def _ensure_2d(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 2:
                return arr
            if arr.ndim == 1:
                length = arr.size
                side = int(np.sqrt(length))
                if side * side == length:
                    return arr.reshape((side, side))
                return arr[np.newaxis, :]
            if arr.ndim == 3:
                return arr[0]
            return arr.squeeze()

        # img1 = _ensure_2d(img1)
        # img2 = _ensure_2d(img2)
        # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # axs[0].imshow(img1, cmap="gray")
        # axs[0].set_title("Image 1")
        # axs[0].axis("off")
        # axs[1].imshow(img2, cmap="gray")
        # axs[1].set_title("Image 2")
        # axs[1].axis("off")

        tv_h = torch.abs(pred_image[:, :, 1:, :] - pred_image[:, :, :-1, :]).sum()
        tv_w = torch.abs(pred_image[:, :, :, 1:] - pred_image[:, :, :, :-1]).sum()
        tv_loss = tv_h + tv_w

        data_loss = criterion(pred_intensity, measured_intensity)
        # total_loss = data_loss + lambda_tv * tv_loss
        total_loss = data_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        losses.append(total_loss.item())

    model.eval()
    with torch.no_grad():
        refined = model(measured_intensity).detach()

    for param in model.parameters():
        param.requires_grad = True

    return refined, losses


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: str | Path,
) -> float:
    """Trains the V1 model and saves the best checkpoint by average epoch loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for measured_intensity, target in dataloader:
            measured_intensity = measured_intensity.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            prediction = model(measured_intensity)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(dataloader), 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )

    plot_loss_curve(
        epoch_losses,
        "Pre-Training Loss Convergence",
        "pretraining_loss_curve.png",
    )

    return best_loss


def plot_loss_curve(losses, title, filename):
    """Plots and saves a generic loss convergence curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss", color="blue")
    plt.title(title)
    plt.xlabel("Epoch / Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_results(
    target: torch.Tensor,
    measured: torch.Tensor,
    pre_trained: torch.Tensor,
    fine_tuned: torch.Tensor,
    output_path: str | Path | None = None,
) -> None:
    """Plots and optionally saves a 1x4 comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    def _to_2d_tensor(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().cpu().numpy()
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr[0]
        if arr.ndim == 1:
            length = arr.size
            side = int(np.sqrt(length))
            if side * side == length:
                return arr.reshape((side, side))
            return arr[np.newaxis, :]
        return arr.squeeze()

    axes[0].imshow(_to_2d_tensor(target[0, 0]), cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(_to_2d_tensor(measured[0, 0]), cmap="viridis")
    axes[1].set_title("Input Diffraction")
    axes[1].axis("off")
    axes[2].imshow(_to_2d_tensor(pre_trained[0, 0]), cmap="gray")
    axes[2].set_title("Pre-trained")
    axes[2].axis("off")
    axes[3].imshow(_to_2d_tensor(fine_tuned[0, 0]), cmap="gray")
    axes[3].set_title("Fine-tuned")
    axes[3].axis("off")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)

    plt.show()
