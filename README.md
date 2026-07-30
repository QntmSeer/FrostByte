# FrostByte: SE(3)-Equivariant Diffusion Prior for Cryo-EM

<p align="center">
  <img src="./assets/logo.svg" alt="FrostByte Logo" width="420"/>
</p>

![License](https://img.shields.io/badge/License-MIT-blue) ![Phase](https://img.shields.io/badge/Phase-5%20Volumetric-brightgreen)

## TL;DR
An **SE(3)-equivariant diffusion prior** for 3D molecular density reconstruction from noisy 2D Cryo-EM projections.
The pipeline progresses from geometric point-cloud priors (Phase 1–3) → multi-protein generalist models (Phase 4) → continuous 3D electron density maps via volumetric diffusion (Phase 5).

---

## ⚡ Why This Matters
Cryo-EM reconstruction is an ill-posed inverse problem. Standard algorithms (RELION, cryoSPARC) struggle with noise, missing views, and conformational heterogeneity. **Learnable generative priors** provide structural regularization grounded in statistical mechanics, while remaining differentiable end-to-end.

This repository demonstrates:
- **SE(3) Geometric Correctness** — equivariance error validated at $10^{-6}$
- **Physics-Aware Forward Modeling** — CTF simulation and differentiable Radon transform
- **Volumetric Density Reconstruction** — $64^3$ voxel U-Net with DDPM training
- **Principled Failure-Mode Diagnosis** — latent-to-physical scale collapse identification and repair

---

## 🏗 System Architecture

```mermaid
graph TD;
    A[2D Projections via CTF] --> B(Inverse Problem / DPS);
    C[Diffusion Prior] --> B;
    D[Coordinate Calibration λ=1.59] -->|prevents collapse| B;
    B --> E[Reconstructed 3D Volume];
    E --> F{Metrics};
    F --> G[CC / RMSD / Rg verification];

    subgraph Phase1-3: Point Cloud
        H[SE3-GNN Score Model]
    end
    subgraph Phase4: Generalist
        I[Multi-Protein CATH-20 Dataset]
    end
    subgraph Phase5: Volumetric
        J[3D U-Net + Radon Projector]
    end

    H --> C
    I --> C
    J --> C
```

---

## 📂 Project Structure

```
diffusion-cryoem-prior/
├── data/
│   ├── volume_dataset.py       # Voxelization via Gaussian splatting (Phase 5)
│   └── triplane_dataset.py     # Continuous coord sampler (Phase 6, WIP)
├── models/
│   ├── diffusion.py            # DDPM core (noise schedule, sampling, DPS)
│   ├── unet_3d.py              # 3D U-Net backbone (Phase 5)
│   ├── unet_2d.py              # 2D Tri-Plane U-Net (Phase 6)
│   ├── triplane.py             # INR MLP decoder (Phase 6)
│   └── triplane_encoder.py     # Spatially-aware 3D encoder (Phase 6)
├── projection/
│   ├── radon.py                # Differentiable Radon Transform (Phase 5)
│   └── neural_radon.py         # Neural Ray-Marcher (Phase 6)
├── scripts/
│   ├── train_volume_prior.py   # Phase 5 DDPM training
│   ├── train_volume_overfit.py # Single-protein overfitting (sanity check)
│   ├── verify_volume_reconstruction.py
│   ├── benchmark_1mbn_volume.py
│   └── visualize_volume_hd.py
└── experiments/
    └── checkpoints/            # Saved model weights (gitignored)
```

---

## 🔬 Development Phases

### Phase 1–2: SE(3)-Equivariant Point Cloud Prior
- Custom Geometric Message Passing (SE(3)-equivariant GNN)
- CTF simulation in Fourier space
- **Result**: Equivariance error $1.0 \times 10^{-6}$

![CTF Physics](./assets/ctf_visualization.png)
*Figure: Simulated Contrast Transfer Function applied to a synthetic 2D projection. Phase flips at CTF zeros are clearly visible.*

### Phase 3: Calibration & Scale Repair
Identified a critical **scale mismatch** between the normalized latent space ($z \sim \mathcal{N}(0,I)$) and physical Angstrom coordinates.

| $\alpha$ (Guidance) | Rg (Å) | Aligned RMSD (Å) | Status |
|---|---|---|---|
| 0.01 | 0.35 | 22.8 | Collapsed |
| 0.1  | 0.59 | 14.5 | Scale Restored |
| **1.0** | **0.58** | **0.78** | **High Fidelity** |

> RMSD computed after Kabsch alignment. Coordinate scale $\lambda = 1.59$ prevents prior from contracting the structure by ~40%.

![Collapse Diagnostics](./assets/collapse_diagnostics.png)
*Figure: Radius of Gyration (Rg) as guidance strength α increases. Prior-induced collapse (left) is repaired by coordinate scaling (right).*

![Calibration Plot](./assets/calibration_plot.png)
*Figure: Ablation over guidance strength α. α=1.0 with coordinate_scale=1.59 achieves <1Å aligned RMSD.*

![Atomic Recovery](./assets/atomic_recovery.png)
*Figure: Ground truth (red) vs reconstructed (blue) point cloud after Kabsch alignment.*

### Phase 4: Generalist Multi-Protein Model
- Trained on a diverse **CATH-20 subset** (19 proteins, Lysozyme excluded as OOD test)
- Evaluated zero-shot generalization to **Myoglobin (1MBN)**
- **Finding**: OOD generalization gap requires scale, not architecture change

![Generalist Myoglobin](./assets/generalist_result.png)
*Figure: Generated structure from generalist prior vs Myoglobin ground truth. Shape is preserved but fold-specific details require more training scale.*

### Phase 5: Volumetric Electron Density (Current)
Transitioned from point-cloud Cα atoms to **continuous 3D electron density grids**, enabling direct compatibility with real Cryo-EM MRC data.

**Key changes:**
- `VolumeDataset`: PDB → $64^3$ voxel grids via Gaussian splatting
- `UNet3D`: Standard 3D encoder-decoder with skip connections, trained with DDPM
- `RadonTransform`: Differentiable Z-axis integration for 2D projection rendering
- **Reconstruction pipeline**: 3 projections → DPS-guided 3D volume reconstruction

**Verified:** Single-protein density recovery (Lysozyme overfitting benchmark).
Cross-Correlation with GT volume reaches **CC = 0.85** after coordinate scaling fix.

![Volume Reconstruction](./assets/volume_reconstruction.png)
*Figure: Left — input 2D projection. Centre — ground truth central slice. Right — reconstructed density slice from 3 projections.*

![Volume Reconstruction HD](./assets/volume_reconstruction_hd.png)
*Figure: High-resolution Z=32 slice comparison after single-protein overfitting (50 epochs). The reconstructed density closely matches the ground truth contours.*

![Volume Reconstruction Animated](./assets/volume_reconstruction_hd.gif)
*Animation: Z-axis sweep through the full reconstructed $64^3$ density volume vs ground truth. The continuous gradient confirms the model has learned the 3D structure, not just a 2D projection.*

![3D Prediction Iteration Loop](./assets/evolution.gif)
*Animation: Iteration-by-iteration 3D reverse diffusion sampling trajectory. Watch pure 3D Gaussian noise progressively denoise into the target 3D macromolecular electron density map under Radon guidance.*

![Myoglobin Benchmark](./assets/benchmark_1mbn.png)
*Figure: OOD volumetric benchmark on Myoglobin (1MBN). The model identifies the general shape but lacks fine-grained structural detail without sufficient training scale.*

```bash
# Reproduce the Phase 5 volumetric reconstruction
python scripts/verify_volume_reconstruction.py

# HD visualization of reconstruction vs ground truth
python scripts/visualize_volume_hd.py

# OOD benchmark on Myoglobin (1MBN)
python scripts/benchmark_1mbn_volume.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/QntmSeer/FrostByte.git
cd FrostByte
pip install -r requirements.txt
```

**Dependencies**: `torch`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `biotite`

---

## ⚠️ Limitations & Scope

- **Idealized Projections**: Current forward model uses noise-free projections. Real Cryo-EM introduces colored noise, beam tilt, and ice contamination.
- **Small Dataset**: CATH-20 subset (19 proteins). OOD generalization requires $10^3+$ structures.
- **No Pose Estimation**: Projection angles are assumed known. Blind angle recovery (as in RELION) is a future extension.

---

### Phase 6: Tri-Plane NeRF (Completed)
- Replaced $O(N^3)$ voxel grid with continuous Implicit Neural Representation (INR).
- Tri-Plane architecture: Three orthogonal 2D feature planes decoded by a shared MLP.
- **Result**: Successfully broke the memory bottleneck, enabling $128^3$ reconstruction with sub-Ångström precision.

### Phase 7: High-Capacity Latent Diffusion (Current)
- **Architecture Scaling**: Expanded Tri-Plane latent space to 128 channels per plane.
- **Denoising Prior**: Trained 2D U-Net prior over compressed tri-plane features.
- **HPC Optimization**: Full pipeline stabilization on L4 GPUs with optimized auto-resume logic.

### 🖼️ Structural Reconstruction Gallery
This gallery demonstrates the model's ability to encode and denoise 3D density maps for benchmark proteins.

![Tri-Plane Prior Gallery](./assets/reconstruction_gallery_v7.png)
*Figure 1: Stable baseline (v4) results. The "Diffusion-Denoised" column highlights the generative prior actively reconstructing tertiary folds at 128^3 resolution.*

---

## 📚 References

1. **DPS**: Chung et al., "Diffusion Posterior Sampling for General Inversion," ICLR 2023.
2. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
3. **EG3D / Tri-Planes**: Chan et al., "Efficient Geometry-Aware 3D GANs," CVPR 2022.
4. **CryoDRGN**: Zhong et al., "CryoDRGN: Reconstruction of Heterogeneous cryo-EM Structures," Nature Methods 2021.
5. **InstaMap**: Wes et al., "InstaMap: instant-NGP for cryo-EM density maps," IUCrJ 2024.
6. **SE(3)-EGNN**: Satorras et al., "E(n) Equivariant Graph Neural Networks," ICML 2021.
