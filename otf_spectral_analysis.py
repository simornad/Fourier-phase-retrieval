"""
Optical Transfer Function (OTF) & Modulation Transfer Function (MTF) Analysis.
This script analyzes the Spatial Frequency Response of the phase retrieval algorithm,
demonstrating how Physics-Informed Test-Time Adaptation widens the effective MTF
by recovering high spatial frequencies that the U-Net filters out.
"""

import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

from dataset import FPRDataset
from model import UNet
from utils import finetune_test_sample, bartlett_window


def compute_spectral_response(gt_img: np.ndarray, pred_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D frequency error map and the 1D radial response profile (Pseudo-MTF).
    """
    F_gt = np.fft.fftshift(np.fft.fft2(gt_img))
    F_pred = np.fft.fftshift(np.fft.fft2(pred_img))
    
    # 2D Error/PSNR Map (Spatial Frequency Response)
    err_map = np.abs(F_gt - F_pred)**2 / (28 * 28)**2
    psnr_map = 10 * np.log10(1.0 / (err_map + 1e-8))
    
    # 1D Radial Profile (Modulation Transfer Function equivalent)
    Y, X = np.indices((28, 28))
    R = np.sqrt((X - 14)**2 + (Y - 14)**2)
    R = np.round(R).astype(int)
    
    radii = []
    radial_psnr = []
    max_radius = 14  
    
    for r in range(max_radius + 1):
        mask = (R == r)
        if np.sum(mask) > 0:
            mse = np.mean(err_map[mask])
            psnr = 10 * np.log10(1.0 / (mse + 1e-8))
            radii.append(r)
            radial_psnr.append(psnr)
            
    return psnr_map, np.array(radii), np.array(radial_psnr)


def main() -> None:
    print("Initializing OTF / Spatial Frequency Analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialization
    dataset = FPRDataset(train=False)
    model = UNet().to(device)
    
    # Robust checkpoint path: always look in the script's directory
    checkpoint_path = Path(__file__).parent / "best_model_so_far.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure it is in the correct directory.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    window = bartlett_window(128, device)

    # 2. Select Sample and Run Inference
    idx = random.randint(0, len(dataset) - 1)
    measured_intensity, target = dataset[idx]
    
    measured_intensity = measured_intensity.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    print(f"Processing Sample {idx}...")
    with torch.no_grad():
        pretrained_pred = model(measured_intensity)

    print("Executing Physics Fine-Tuning (PINN)...")
    finetuned_pred, _ = finetune_test_sample(
        model=model,
        measured_intensity=measured_intensity,
        window=window,
        iterations=50,
        lr=1e-5
    )

    # 3. Spatial Cropping
    gt_img = target[0, 0, 50:78, 50:78].detach().cpu().numpy()
    pre_img = pretrained_pred[0, 0, 50:78, 50:78].detach().cpu().numpy()
    ft_img = finetuned_pred[0, 0, 50:78, 50:78].detach().cpu().numpy()

    # 4. Compute Frequency Responses (2D Maps and 1D Radial MTF)
    psnr_pre_2d, radii, psnr_pre_1d = compute_spectral_response(gt_img, pre_img)
    psnr_ft_2d, _, psnr_ft_1d = compute_spectral_response(gt_img, ft_img)
    psnr_diff_2d = psnr_ft_2d - psnr_pre_2d

    # 5. Plotting Architecture
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    x = np.arange(-14, 14)
    y = np.arange(-14, 14)
    X, Y = np.meshgrid(x, y)

    # --- TOP ROW: 3D Spatial Frequency Response ---
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf1 = ax1.plot_surface(X, Y, psnr_pre_2d, cmap=cm.coolwarm, edgecolor='none', alpha=0.9)
    ax1.set_title("Step 1: Pre-Trained SFR (3D)", fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    surf2 = ax2.plot_surface(X, Y, psnr_ft_2d, cmap=cm.coolwarm, edgecolor='none', alpha=0.9)
    ax2.set_title("Step 2: Fine-Tuned SFR (3D)", fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    surf3 = ax3.plot_surface(X, Y, psnr_diff_2d, cmap=cm.viridis, edgecolor='none', alpha=0.9)
    ax3.set_title(r"$\Delta$ SFR (Physics Correction)", fontweight='bold')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Fx (Cycles/Pixel)")
        ax.set_ylabel("Fy (Cycles/Pixel)")
        ax.set_zlabel("Response (dB)")
        ax.view_init(elev=35, azim=45)

    # --- BOTTOM ROW: 2D Heatmaps & 1D MTF Curve ---
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(psnr_pre_2d, cmap='coolwarm', extent=[-14, 14, -14, 14])
    ax4.set_title("Pre-Trained (2D Spectrum Heatmap)")
    ax4.set_xlabel("Fx"); ax4.set_ylabel("Fy")
    fig.colorbar(im4, ax=ax4, shrink=0.7)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(psnr_ft_2d, cmap='coolwarm', extent=[-14, 14, -14, 14])
    ax5.set_title("Fine-Tuned (2D Spectrum Heatmap)")
    ax5.set_xlabel("Fx"); ax5.set_ylabel("Fy")
    fig.colorbar(im5, ax=ax5, shrink=0.7)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(radii, psnr_pre_1d, marker='o', linewidth=2.5, label='Pre-Trained (U-Net Only)', color='#E63946')
    ax6.plot(radii, psnr_ft_1d, marker='s', linewidth=2.5, label='Fine-Tuned (PINN)', color='#1D3557')
    ax6.set_title("Radial Spatial Frequency Response (Pseudo-MTF)", fontweight='bold')
    ax6.set_xlabel("Spatial Frequency (Radial Distance)")
    ax6.set_ylabel("Reconstruction Response (PSNR dB)")
    ax6.grid(True, linestyle='--', alpha=0.7)
    ax6.legend()

    plt.suptitle("Optical Transfer Function (OTF) Analysis of the Phase Retrieval Algorithm", 
                 fontsize=18, fontweight='bold', y=0.96)

    output_filename = "otf_spectral_analysis.png"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete. Figure saved to '{output_filename}'.")


if __name__ == "__main__":
    main()