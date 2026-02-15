"""Dataset utilities for MNIST-based Fourier phase retrieval."""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class FPRDataset(Dataset):
    """MNIST dataset with diffraction-intensity inputs and image targets."""

    def __init__(self, mnist_root: str = "./data", train: bool = True) -> None:
        self.mnist = datasets.MNIST(
            root=mnist_root,
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(128),
                    transforms.ToTensor(),
                ]
            ),
        )
        w1d = torch.bartlett_window(128)
        self.window = w1d.unsqueeze(1) * w1d.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.mnist)

    def compute_diffraction(self, image: torch.Tensor) -> torch.Tensor:
        """Calculates the normalized log-intensity diffraction pattern."""
        windowed_image = image[0] * self.window
        fourier = torch.fft.fft2(windowed_image)
        shifted = torch.fft.fftshift(fourier, dim=(-2, -1))
        intensity = torch.abs(shifted) ** 2
        log_intensity = torch.log1p(intensity)
        normalized = log_intensity / 20.0
        return torch.clamp(normalized, 0.0, 1.0).unsqueeze(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.mnist[idx]
        intensity = self.compute_diffraction(image)
        return intensity, image
