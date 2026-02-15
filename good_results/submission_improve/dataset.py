import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class FPRDataset(Dataset):
    """MNIST-based Fourier phase retrieval dataset."""

    def __init__(self, mnist_root="./data", train=True, size=128):
        self.mnist = datasets.MNIST(
            root=mnist_root,
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(50),
                    transforms.ToTensor(),
                ]
            ),
        )
        w1d = torch.bartlett_window(size)
        self.window = w1d.unsqueeze(1) * w1d.unsqueeze(0)

    def __len__(self):
        return len(self.mnist)

    def compute_diffraction(self, windowed_image):
        fourier = torch.fft.fft2(windowed_image)
        fourier_shifted = torch.fft.fftshift(fourier, dim=(-2, -1))
        intensity = torch.abs(fourier_shifted) ** 2
        log_intensity = torch.log1p(intensity)
        normalized = log_intensity / 20.0
        return torch.clamp(normalized, 0.0, 1.0).unsqueeze(0)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]
        windowed_image = image[0] * self.window
        intensity = self.compute_diffraction(windowed_image)
        return intensity, windowed_image.unsqueeze(0)
