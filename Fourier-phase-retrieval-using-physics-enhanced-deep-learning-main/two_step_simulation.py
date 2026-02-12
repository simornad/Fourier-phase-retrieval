"""
Two-step Fourier phase retrieval simulation (paper-style workflow).

This script follows the repository's existing implementation style:
1) Step-1 (self-supervised pretraining stage simulation):
   - Use the physical forward model `realFFT` to synthesize diffraction intensity.
   - Recover amplitude with a pre-trained UNet checkpoint from `models/checkpoint`.
2) Step-2 (physics-enhanced refinement stage simulation):
   - Start from Step-1 prediction.
   - Run per-sample physics-constrained optimization:
       L = ||H(A) - I_measured||_2^2 + lambda * ||A - A_step1||_2^2
   - Optionally initialize the network with a fine-tuned checkpoint from
     `models/face_best_pn_ft.pth` if it is architecture-compatible.

The script is intentionally verbose and heavily commented to explain each stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from model.model import realFFT
from model.unet import UNet


# -------------------------------
# Utility helpers
# -------------------------------
def load_grayscale_image(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    """Load an image as normalized tensor [1, 1, H, W] in [0, 1]."""
    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(image_path)
    amp = tfm(img).unsqueeze(0).to(device=device, dtype=torch.float32)
    return amp


def load_unet_checkpoint(net: UNet, checkpoint_path: Path, device: torch.device) -> None:
    """Load UNet weights from a plain state_dict checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)

    # Existing repository checkpoints are usually direct state_dict files.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Info] Non-strict load for {checkpoint_path.name}")
        print(f"       missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def synthesize_diffraction(amp: torch.Tensor, fft_size: int, use_window: bool = True) -> torch.Tensor:
    """
    Apply physics forward model from this repository:
    I_measured = H(amp, phase=ones).
    """
    H = realFFT()
    I_measured, _ = H(amp, torch.ones_like(amp), out_size=fft_size, window=use_window)
    return I_measured


# -------------------------------
# Step-1: Pretrained network inference
# -------------------------------
def step1_reconstruct(pretrained_net: UNet, measured_I: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Stage-1 reconstruction:
    Resize measured diffraction to network input resolution, then infer amplitude.
    """
    I_for_net = F.interpolate(measured_I, size=(image_size, image_size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        amp_stage1 = pretrained_net(I_for_net)
    return torch.clamp(amp_stage1, 0.0, 1.0)


# -------------------------------
# Step-2: Physics-enhanced refinement
# -------------------------------
def step2_physics_refine(
    measured_I: torch.Tensor,
    amp_init: torch.Tensor,
    fft_size: int,
    refine_steps: int,
    refine_lr: float,
    prior_lambda: float,
    use_window: bool,
) -> torch.Tensor:
    """
    Stage-2 refinement (physics-consistency optimization).

    We optimize a free amplitude variable initialized by Step-1 output.
    This directly mirrors the "physics-enhanced" idea: enforce consistency with
    measured diffraction through the same forward model used in training.
    """
    H = realFFT()

    # Trainable variable initialized with stage-1 estimate.
    amp_var = torch.nn.Parameter(amp_init.detach().clone())
    optimizer = torch.optim.Adam([amp_var], lr=refine_lr)

    for step in range(refine_steps):
        optimizer.zero_grad()

        amp_pos = torch.clamp(amp_var, 0.0, 1.0)
        I_pred, _ = H(amp_pos, torch.ones_like(amp_pos), out_size=fft_size, window=use_window)

        loss_physics = F.mse_loss(I_pred, measured_I)
        loss_prior = F.mse_loss(amp_pos, amp_init)
        loss = loss_physics + prior_lambda * loss_prior

        loss.backward()
        optimizer.step()

        if (step + 1) % max(1, refine_steps // 10) == 0 or step == 0:
            print(
                f"[Step-2] iter {step+1:04d}/{refine_steps} "
                f"physics={loss_physics.item():.6e} prior={loss_prior.item():.6e} total={loss.item():.6e}"
            )

    return torch.clamp(amp_var.detach(), 0.0, 1.0)


# -------------------------------
# Main workflow
# -------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Two-step FPR simulation using repository checkpoints")
    parser.add_argument(
        "--input-image",
        type=Path,
        default=Path("train_MINST_128/1.png"),
        help="Input object amplitude image used for simulation",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=Path("models/checkpoint/checkpoint_Unet_MNIST_v5window.pth.tar"),
        help="Step-1 pretrained UNet checkpoint",
    )
    parser.add_argument(
        "--finetuned-ckpt",
        type=Path,
        default=Path("models/face_best_pn_ft.pth"),
        help="Optional Step-2 fine-tuned checkpoint used as alternative initialization",
    )
    parser.add_argument("--image-size", type=int, default=128, help="UNet input/output resolution")
    parser.add_argument("--fft-size", type=int, default=300, help="Diffraction plane crop size")
    parser.add_argument("--refine-steps", type=int, default=200, help="Step-2 optimization iterations")
    parser.add_argument("--refine-lr", type=float, default=1e-2, help="Step-2 optimization learning rate")
    parser.add_argument("--prior-lambda", type=float, default=0.1, help="Weight for stage-1 prior regularization")
    parser.add_argument("--no-window", action="store_true", help="Disable Bartlett window in forward model")
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_outputs"), help="Output folder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    if not args.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")
    if not args.pretrained_ckpt.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_ckpt}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Stage A: simulate data generation with the physical forward model.
    # -----------------------------------------------------------------
    amp_gt = load_grayscale_image(args.input_image, args.image_size, device)
    measured_I = synthesize_diffraction(amp_gt, fft_size=args.fft_size, use_window=not args.no_window)

    # -----------------------------------------------------------------
    # Stage B (Step-1): pretrained model inference.
    # -----------------------------------------------------------------
    net_pre = UNet(in_channels=1).to(device)
    net_pre.eval()
    load_unet_checkpoint(net_pre, args.pretrained_ckpt, device)

    amp_stage1 = step1_reconstruct(net_pre, measured_I, image_size=args.image_size)

    # Optionally load fine-tuned checkpoint for additional initialization check.
    # If incompatible, we keep using stage-1 output as initialization.
    if args.finetuned_ckpt.exists():
        try:
            net_ft = UNet(in_channels=1).to(device)
            net_ft.eval()
            load_unet_checkpoint(net_ft, args.finetuned_ckpt, device)
            with torch.no_grad():
                amp_stage1_ft = net_ft(F.interpolate(measured_I, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False))
            # Blend two predictions to use both available repository weights.
            amp_stage1 = torch.clamp(0.5 * amp_stage1 + 0.5 * amp_stage1_ft, 0.0, 1.0)
            print("[Info] Fine-tuned checkpoint loaded and blended for stage-1 initialization.")
        except Exception as exc:
            print(f"[Warning] Could not use fine-tuned checkpoint ({args.finetuned_ckpt.name}): {exc}")

    # -----------------------------------------------------------------
    # Stage C (Step-2): physics-enhanced refinement.
    # -----------------------------------------------------------------
    amp_stage2 = step2_physics_refine(
        measured_I=measured_I,
        amp_init=amp_stage1,
        fft_size=args.fft_size,
        refine_steps=args.refine_steps,
        refine_lr=args.refine_lr,
        prior_lambda=args.prior_lambda,
        use_window=not args.no_window,
    )

    # Save outputs for quick inspection.
    save_image(amp_gt, str(args.output_dir / "amp_ground_truth.png"))
    save_image(F.interpolate(measured_I, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False), str(args.output_dir / "diffraction_resized.png"))
    save_image(amp_stage1, str(args.output_dir / "amp_stage1_pretrained.png"))
    save_image(amp_stage2, str(args.output_dir / "amp_stage2_physics_refined.png"))

    # Numeric summary.
    mse_stage1 = F.mse_loss(amp_stage1, amp_gt).item()
    mse_stage2 = F.mse_loss(amp_stage2, amp_gt).item()
    print("\n=== Two-step simulation summary ===")
    print(f"Input image:           {args.input_image}")
    print(f"Pretrained checkpoint: {args.pretrained_ckpt}")
    print(f"Fine-tuned checkpoint: {args.finetuned_ckpt} (optional)")
    print(f"MSE(stage-1, GT):      {mse_stage1:.6e}")
    print(f"MSE(stage-2, GT):      {mse_stage2:.6e}")
    print(f"Saved outputs in:      {args.output_dir}")


if __name__ == "__main__":
    main()
