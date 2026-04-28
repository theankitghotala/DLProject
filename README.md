# Latent Diffusion Models: From Theory to Implementation
### High-Resolution Image Synthesis via Perceptual Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## 📌 Project Overview
This repository contains a comprehensive implementation and independent verification of the **Latent Diffusion Model (LDM)** as proposed by Rombach et al. in *"High-Resolution Image Synthesis with Latent Diffusion Models"* (CVPR 2022). 

The project focuses on deconstructing the LDM pipeline into its constituent modules—**VAE, U-Net, and CLIP**—and evaluating its performance in text-to-image synthesis, object removal (inpainting), and super-resolution.

## 🚀 Key Features
- **Modular Pipeline:** Manual deconstruction of the LDM architecture (avoiding high-level pipelines for deeper architectural understanding).
- **Hardware Optimized:** Implementation of `fp16` precision and memory-efficient attention to run billion-parameter models on consumer-grade GPUs (tested on 6GB VRAM).
- **Multi-Task Support:**
    - **Text-to-Image:** Guided synthesis via Cross-Attention mechanisms.
    - **Object Removal (Inpainting):** Leveraging negative prompts and mask-guided denoising for seamless background reconstruction.
    - **Super-Resolution:** 4x upscaling implemented via **6-channel concatenation** (Noisy Latents + Low-Res Guide).
- **Quantitative Evaluation:** Calculation of Reconstruction PSNR to verify the perceptual compression fidelity of the VAE.

## 🛠️ Technical Architecture
The LDM operates by shifting the diffusion process from pixel space to a lower-dimensional **Latent Space**. 

1. **Perceptual Compression:** A VAE compresses $512 \times 512 \times 3$ images into $64 \times 64 \times 4$ latents, removing high-frequency redundancy while preserving semantic content.
2. **Latent Diffusion:** A Time-conditioned U-Net learns to reverse Gaussian noise within this latent manifold.
3. **Conditioning:** CLIP text embeddings guide the generation through **Cross-Attention** layers.

## 📊 Evaluation Metrics
| Metric | Result |
| :--- | :--- |
| **Reconstruction PSNR** | **33.37 dB** |
| **Peak VRAM Usage** | ~2.05 GB (Inference) |
| **Inference Time** | ~6.6s (50 steps on local GPU) |

## 📦 Installation & Setup
```bash
# Clone the repository
git clone [https://github.com/theankitghotala/DLProject.git](https://github.com/theankitghotala/DLProject.git)

# Create a virtual environment
python -m venv venv
python -m pip install --upgrade pip   
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate pillow safetensors scikit-image matplotlib
```
