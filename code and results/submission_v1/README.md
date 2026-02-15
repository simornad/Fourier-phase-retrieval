# Fourier Phase Retrieval via Physics-Enhanced Deep Learning

This repository contains a modular PyTorch implementation for solving the Fourier Phase Retrieval (FPR) problem using a combination of deep learning (U-Net) and physics-driven test-time adaptation.

The algorithm reconstructs an image of a digit from its diffraction intensity (Fourier magnitude), overcoming the phase ambiguity problem inherent in optical systems.

## Project Structure

* `main.py`: The entry point for execution. Loads the model, evaluates test samples, applies fine-tuning, and plots the results.
* `model.py`: Defines the U-Net architecture used for the initial phase reconstruction.
* `dataset.py`: Handles the downloading and preprocessing of the MNIST dataset, including diffraction pattern simulation and fixed scalar normalization.
* `utils.py`: Contains the core physics forward model, the test-time adaptation logic (partial freezing and optimization), and visualization helpers.
* `requirements.txt`: Lists the Python dependencies required to run the code.

## Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

## Installation

1. Clone or download this repository to your local machine.
2. Open a terminal or command prompt in the project root directory.
3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Included Weights

This submission includes the pre-trained model file copied from your local results:

- `best_model_so_far.pth`
- `v1_best_model.pth` (compatibility alias)

`main.py` automatically searches for these names in the `submission_v1` folder.

## Run

From inside `submission_v1`, run:

```bash
python main.py
```

By default, `TRAIN_MODE = False`, so the script loads pre-trained weights and runs evaluation + test-time fine-tuning on 3 random test samples.

To run pre-training instead, edit `main.py` and set:

```python
TRAIN_MODE = True
```

Then run `python main.py` again. The best checkpoint is saved as `v1_best_model.pth`.
