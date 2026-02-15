"""Entry point for V1 Fourier phase retrieval test-time adaptation."""

import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dataset import FPRDataset
from model import UNet
from utils import (
    bartlett_window,
    finetune_test_sample,
    plot_loss_curve,
    plot_results,
    train_model,
    physics_forward
)


TRAIN_MODE = False
PRETRAIN_EPOCHS = 50
PRETRAIN_LR = 1e-3


def load_weights(model: UNet, weights_path: Path, device: torch.device) -> None:
    """Loads model weights from either a plain state dict or a checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def resolve_weights_path() -> Path | None:
    """Finds an available pre-trained weights file in the project folder."""
    candidates = [
        BASE_DIR / "v1_best_model.pth",
        BASE_DIR / "best_model_so_far.pth",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def main() -> None:
    """Runs either V1 pre-training or V1 evaluation with test-time adaptation."""
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet().to(device)
    weights_path = BASE_DIR / "v1_best_model.pth"

    if TRAIN_MODE:
        print("Starting pre-training phase...")

        train_dataset = FPRDataset(mnist_root="./data", train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
        )

        best_loss = train_model(
            model=model,
            dataloader=train_loader,
            epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            device=device,
            save_path=weights_path,
        )
        print(f"Pre-training finished. Best loss: {best_loss:.6f}")
        print(f"Saved best weights to: {weights_path}")
        return

    print("Loading pre-trained weights for evaluation...")

    dataset = FPRDataset(mnist_root="./data", train=False)

    weights_path = resolve_weights_path()
    if weights_path is None:
        print(
            "Weights file not found. Expected one of:\n"
            "- v1_best_model.pth\n"
            "- best_model_so_far.pth\n"
            "Please place a pre-trained V1 weights file in the submission_v1 folder."
        )
        return

    print(f"Using weights file: {weights_path}")

    window = bartlett_window(size=128, device=device)
    window_rect = torch.ones_like(window, device=device)  

    sample_count = min(3, len(dataset))
    indices = random.sample(range(len(dataset)), sample_count)

    for rank, idx in enumerate(indices, start=1):
        load_weights(model, weights_path, device)
        model.eval()

        measured_intensity, target = dataset[idx]
        measured_intensity = physics_forward(target.to(window_rect.device, dtype=target.dtype),window_rect)
        measured_intensity = measured_intensity.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        with torch.no_grad():
            pre_trained = model(measured_intensity)

        # img1 = target[0, 0].detach().cpu().float().numpy()
        # img2 = measured_intensity[0, 0].detach().cpu().float().numpy()
        # # img1, img2 are 2D arrays (H,W) or tensors converted to numpy
        # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # axs[0].imshow(img1, cmap="gray")
        # axs[0].set_title("Image 1")
        # axs[0].axis("off")
        # axs[1].imshow(img2, cmap="gray")
        # axs[1].set_title("Image 2")
        # axs[1].axis("off")


        fine_tuned, losses = finetune_test_sample(
            model=model,
            measured_intensity=measured_intensity,
            window=window,
            iterations=50,
            lr=1e-5,
            lambda_tv=1e-5,
        )
        plot_loss_curve(
            losses,
            f"Fine-Tuning Loss (Sample {idx})",
            f"finetuning_loss_sample_{idx}.png",
        )

        mse_pre = torch.nn.functional.mse_loss(pre_trained, target).item()
        mse_ft = torch.nn.functional.mse_loss(fine_tuned, target).item()
        improvement = (mse_pre - mse_ft) / (mse_pre + 1e-12) * 100.0

        print(
            f"Sample {idx} | "
            f"MSE pre: {mse_pre:.6f} | "
            f"MSE fine-tuned: {mse_ft:.6f} | "
            f"Improvement: {improvement:.2f}% | "
            f"Loss start/end: {losses[0]:.6f}/{losses[-1]:.6f}"
        )

        plot_results(
            target=target,
            measured=measured_intensity,
            pre_trained=pre_trained,
            fine_tuned=fine_tuned,
            output_path=BASE_DIR / "results" / f"v1_sample_{rank}_idx_{idx}.png",
        )


if __name__ == "__main__":
    main()
