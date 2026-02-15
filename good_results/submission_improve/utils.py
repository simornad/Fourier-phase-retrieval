from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def bartlett_window(size, device):
    """Create a 2D Bartlett window."""
    w = torch.bartlett_window(size, device=device)
    return w.unsqueeze(1) * w.unsqueeze(0)


def physics_forward(image, window):
    """Forward physics model using fixed normalization."""
    if image.dim() == 4:
        img = image[:, 0]
        windowed = img * window.unsqueeze(0)
        fourier = torch.fft.fft2(windowed)
        shifted = torch.fft.fftshift(fourier, dim=(-2, -1))
        intensity = torch.abs(shifted) ** 2
        log_intensity = torch.log1p(intensity)
        norm = log_intensity / 20.0
        return torch.clamp(norm, 0.0, 1.0).unsqueeze(1)

    if image.dim() == 3:
        img = image[0]
        windowed = img * window
        fourier = torch.fft.fft2(windowed)
        shifted = torch.fft.fftshift(fourier, dim=(-2, -1))
        intensity = torch.abs(shifted) ** 2
        log_intensity = torch.log1p(intensity)
        norm = log_intensity / 20.0
        return torch.clamp(norm, 0.0, 1.0).unsqueeze(0)

    raise ValueError("Unsupported image shape for physics_forward")


def finetune_test_sample(
    model,
    measured_intensity,
    window,
    iterations=200,
    lr=1e-5,
    lambda_tv=1e-5,
):
    """Run test-time adaptation on model weights."""
    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    target_amplitude = torch.sqrt(torch.expm1(measured_intensity * 20.0))
    window_batch = window.unsqueeze(0).unsqueeze(0)

    for _ in range(iterations):
        optimizer.zero_grad()
        pred_image = model(measured_intensity)

        windowed_pred = pred_image * window_batch
        fourier_pred = torch.fft.fft2(windowed_pred, dim=(-2, -1))
        pred_amplitude = torch.abs(torch.fft.fftshift(fourier_pred, dim=(-2, -1)))

        tv_h = torch.abs(pred_image[:, :, 1:, :] - pred_image[:, :, :-1, :]).sum()
        tv_w = torch.abs(pred_image[:, :, :, 1:] - pred_image[:, :, :, :-1]).sum()
        tv_loss = tv_h + tv_w

        data_loss = criterion(pred_amplitude, target_amplitude)
        total_loss = data_loss + lambda_tv * tv_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(total_loss.item())

    model.eval()
    with torch.no_grad():
        refined = model(measured_intensity).detach()

    return refined, losses


def plot_loss_curve(losses, title, filename):
    """Plot and save a loss curve."""
    plt.figure(figsize=(7, 4))
    plt.plot(losses, marker="o")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    scheduler_patience,
    early_stopping_patience,
    save_path="best_model_so_far.pth",
):
    """Pre-train model with train/validation loop and checkpointing."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
    )

    train_losses = []
    val_losses = []

    save_path = Path(save_path)
    best_val = float("inf")
    best_for_patience = float("inf")
    early_stop_counter = 0

    model = model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0

        for inp, tgt in train_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)

            pred = model(inp)
            loss = criterion(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_sum += loss.item()

        avg_train = train_sum / len(train_loader)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                pred = model(inp)
                val_sum += criterion(pred, tgt).item()

        avg_val = val_sum / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            payload = {
                "epoch": epoch,
                "val_loss": avg_val,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "version": "baseline",
            }
            torch.save(payload, save_path)

        if avg_val < best_for_patience:
            best_for_patience = avg_val
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
            f"Best Val: {best_val:.6f} | LR: {current_lr:.2e} | "
            f"Patience: {early_stop_counter}/{early_stopping_patience}"
        )

        if early_stop_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    plot_loss_curve(
        losses=train_losses,
        title="Pre-training Train Loss",
        filename=save_path.parent / "pretraining_loss_curve.png",
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val": best_val,
        "save_path": str(save_path),
    }


def plot_results(results, save_path="random_eval.png"):
    """Plot GT, diffraction, pre-trained and fine-tuned outputs."""
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.2 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, row in enumerate(results):
        target = row["target"]
        measured_intensity = row["measured_intensity"]
        pretrained_pred = row["pretrained_pred"]
        finetuned_pred = row["finetuned_pred"]
        mse_pre = row["mse_pre"]
        mse_ft = row["mse_ft"]
        sample_id = row["sample_id"]

        cropped_target = target[0, 0, 50:78, 50:78].detach().cpu().numpy()
        cropped_pretrained = pretrained_pred[0, 0, 50:78, 50:78].detach().cpu().numpy()
        cropped_finetuned = finetuned_pred[0, 0, 50:78, 50:78].detach().cpu().numpy()

        axes[r, 0].imshow(cropped_target, cmap="gray")
        axes[r, 0].set_title(f"Sample {sample_id} GT (28x28)")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(measured_intensity[0, 0].detach().cpu().numpy(), cmap="viridis")
        axes[r, 1].set_title("Input Diffraction (128x128)")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(cropped_pretrained, cmap="gray")
        axes[r, 2].set_title(f"Pre MSE={mse_pre:.4f}")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(cropped_finetuned, cmap="gray")
        axes[r, 3].set_title(f"FT MSE={mse_ft:.4f}")
        axes[r, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
