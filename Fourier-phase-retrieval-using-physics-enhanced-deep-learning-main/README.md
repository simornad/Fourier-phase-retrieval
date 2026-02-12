# Fourier phase retrieval using physics-enhanced deep learning
This is torch demo implementation of paper including ipynb (Fourier phase retrieval using physics-enhanced deep learning). You may get more information in the original paper:
## Paper 
Zike Zhang, Fei Wang, Qixuan Min, Ying Jin, and Guohai Situ, "Fourier phase retrieval using physics-enhanced deep learning," Opt. Lett. 49, 6129-6132 (2024)

Paper Link: https://doi.org/10.1364/OL.537792


## Framework
 Our method includes two steps: First, we leverage the image formation model of the FPR to guide the generation of data for network training in a self-supervised manner. Second, we exploit the physical model to fine-tune the pre-trained model to impose the physics-consistency constraint on the network prediction.
![image](https://github.com/user-attachments/assets/0187ef58-4b78-4d41-88a1-5bee6f278405)


## Result
 Our method provides the best image quality, more results can be found in paper and supplement.
![image](https://github.com/user-attachments/assets/1670655f-ee12-4469-969e-32aed283f8aa)


## Two-step simulation script (based on repository checkpoints)
A runnable script is provided at `two_step_simulation.py` to reproduce the two-stage idea in the paper:

1. **Step-1 (self-supervised pretraining stage simulation)**
   - Generate diffraction from an amplitude object using the repository physics model (`realFFT`).
   - Reconstruct amplitude using the pre-trained UNet checkpoint in `models/checkpoint/`.
2. **Step-2 (physics-enhanced refinement stage simulation)**
   - Initialize from the Step-1 prediction.
   - Refine by minimizing physics-consistency loss using the same forward model.
   - Optionally load and blend the fine-tuned checkpoint `models/face_best_pn_ft.pth` when compatible.

Example:

```bash
python two_step_simulation.py \
  --input-image train_MINST_128/1.png \
  --pretrained-ckpt models/checkpoint/checkpoint_Unet_MNIST_v5window.pth.tar \
  --finetuned-ckpt models/face_best_pn_ft.pth \
  --refine-steps 200 \
  --output-dir simulation_outputs
```

Outputs include the simulated diffraction, step-1 prediction, and step-2 refined reconstruction.
