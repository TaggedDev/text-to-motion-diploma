# Human Motion Diffusion Model (MDM)

Implementation based on [*Human Motion Diffusion Model*](https://arxiv.org/abs/2209.14916) (Tevet et al., ICLR 2023) and the [official repository](https://github.com/GuyTevet/motion-diffusion-model).

---

## Data Format

The model consumes `torch_geometric.data.Data` objects as produced by `src/graph_heatmaps.py`:

```
Data(
  pos:       Tensor[T×J, 3]   # flattened XYZ joint positions
  seq_len:   int              # T — number of frames
  num_joints: int             # J — number of joints (22 for HumanML3D)
)
```

Before entering the model the batch is reshaped to `[B, J, 3, T]` (batch × joints × features × frames), which is the canonical MDM tensor layout.

For HumanML3D:  **J = 22, nfeats = 3, data_rep = `xyz`**

---

## Architecture Overview

### 1. MDM Denoiser (Transformer)

```mermaid
flowchart TD
    subgraph Input
        DATA["Data batch\npos: [B·T·J, 3]\nseq_len, num_joints"]
        ADAPT["DataAdapter\nreshape → [B, J, 3, T]"]
        DATA --> ADAPT
    end

    subgraph InputProjection["Input Projection"]
        IP["InputProcess\nLinear(J×nfeats → latent_dim)\n[T, B, latent_dim]"]
        ADAPT --> IP
    end

    subgraph Conditioning
        direction TB
        T_IN["diffusion timestep t  [B]"]
        TSE["TimestepEmbedder\nPE[t] → Linear(d,d) → SiLU → Linear(d,d)\n[1, B, latent_dim]"]
        T_IN --> TSE

        TEXT["text prompts\nList[str]  (optional)"]
        CLIP["CLIP ViT-B/32  (frozen)\n→ float32  [1, B, 512]"]
        EMBT["Linear(512 → latent_dim)\n[1, B, latent_dim]"]
        CFG["CFG dropout\n(mask prob = 0.1 during training)"]
        TEXT --> CLIP --> EMBT --> CFG

        ACT["action class id  [B]  (optional)"]
        EMBA["EmbedAction\nLearnable table  [num_actions, latent_dim]\n[1, B, latent_dim]"]
        ACT --> EMBA

        ADD["Add  emb = time_emb + cond_emb\n[1, B, latent_dim]"]
        TSE --> ADD
        CFG --> ADD
        EMBA --> ADD
    end

    subgraph TransformerEncoder["Transformer Encoder  (default arch)"]
        CAT["Prepend condition token\ncat([emb, x_proj], dim=0)\n[T+1, B, latent_dim]"]
        PE["Sinusoidal PositionalEncoding\n[T+1, B, latent_dim]"]
        ENC["nn.TransformerEncoder\n8 layers · 4 heads · ff=1024 · GELU · dropout=0.1\n[T+1, B, latent_dim]"]
        STRIP["Drop condition token  [1:]\n[T, B, latent_dim]"]
        CAT --> PE --> ENC --> STRIP
    end

    subgraph OutputProjection["Output Projection"]
        OP["OutputProcess\nLinear(latent_dim → J×nfeats)\nreshape → [B, J, nfeats, T]"]
        STRIP --> OP
    end

    IP --> CAT
    ADD --> CAT
    OP --> XHAT["Predicted clean motion  x̂₀\n[B, J, nfeats, T]"]
```

---

### 2. Gaussian Diffusion Wrapper

```mermaid
flowchart LR
    subgraph Training
        X0["Clean motion  x₀\n[B, J, nfeats, T]"]
        NOISE["Sample noise  ε ~ N(0, I)"]
        FWD["Forward process  q(xₜ | x₀)\nxₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε\ncosine schedule  T=1000"]
        XT["Noisy motion  xₜ"]
        MDM1["MDM  →  x̂₀"]
        LOSS["MSE Loss\nL = L_motion\n  + λ_vel · L_vel"]

        X0 --> FWD
        NOISE --> FWD
        FWD --> XT --> MDM1 --> LOSS
        X0 --> LOSS
    end

    subgraph Inference
        XINIT["xₜ ~ N(0, I)"]
        LOOP["Reverse loop  t = T…1\n(50 spaced steps)"]
        MDM2["MDM  →  x̂₀"]
        POSTERIOR["Compute  μ̃(xₜ, x̂₀)\nq(xₜ₋₁ | xₜ, x̂₀)"]
        SAMPLE["Sample  xₜ₋₁"]
        OUT["Generated motion  x₀"]

        XINIT --> LOOP --> MDM2 --> POSTERIOR --> SAMPLE
        SAMPLE -->|"t > 1"| LOOP
        SAMPLE -->|"t = 1"| OUT
    end
```

---

### 3. Noise Schedule

```mermaid
flowchart LR
    SCHED["Cosine schedule\nβₜ derived from\nᾱ(t) = cos²((t/T + 0.008) / 1.008 · π/2)"]
    SCHED --> ALPHA["Cumulative product\nᾱₜ = ∏ (1 - βₛ)"]
    ALPHA --> FWD2["Forward variance\nq(xₜ|x₀) = N(√ᾱₜ·x₀, (1-ᾱₜ)·I)"]
    ALPHA --> POST2["Posterior mean\nμ̃ = (√ᾱₜ₋₁·βₜ·x̂₀ + √ᾱₜ·(1-ᾱₜ₋₁)·xₜ) / (1-ᾱₜ)"]
```

---

## Module Reference

| Module | Class | Description |
|--------|-------|-------------|
| Data adapter | `MotionDataAdapter` | Converts `Data` batch list → `[B, J, nfeats, T]` tensor with padding mask |
| Input projection | `InputProcess` | `Linear(J·nfeats → latent_dim)` for `xyz` / `rot6d`; separate velocity projection for `rot_vel` |
| Timestep embedding | `TimestepEmbedder` | Indexes sinusoidal PE at step `t`, passes through 2-layer MLP with SiLU |
| Positional encoding | `PositionalEncoding` | Standard sinusoidal PE, max_len=5000 |
| Text encoder | CLIP `ViT-B/32` | Frozen; converts text → 512-dim float; projected to `latent_dim` |
| Action encoder | `EmbedAction` | Learnable embedding table `[num_actions, latent_dim]` |
| CFG masking | `mask_cond` | Zeros out condition with probability `cond_mask_prob` during training |
| Transformer | `nn.TransformerEncoder` | 8 × `TransformerEncoderLayer`; condition token prepended to sequence |
| Output projection | `OutputProcess` | `Linear(latent_dim → J·nfeats)` → reshape `[B, J, nfeats, T]` |
| Diffusion | `GaussianDiffusion` | Cosine schedule, predict-x₀ objective, MSE + velocity loss |

---

## Hyperparameters

### Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| `latent_dim` | 512 | d_model of the Transformer |
| `ff_size` | 1024 | Feed-forward hidden dim |
| `num_layers` | 8 | Transformer encoder layers |
| `num_heads` | 4 | Attention heads |
| `dropout` | 0.1 | Applied in attention, FFN, and PE |
| `activation` | GELU | FFN activation |
| `arch` | `trans_enc` | Transformer Encoder (default) |
| `clip_version` | `ViT-B/32` | Frozen CLIP model |
| `clip_dim` | 512 | CLIP embedding dimension |
| `cond_mask_prob` | 0.1 | Classifier-free guidance dropout |

### Diffusion

| Parameter | Value | Notes |
|-----------|-------|-------|
| `diffusion_steps` | 1000 | Training timesteps |
| `noise_schedule` | `cosine` | |
| `predict_xstart` | `True` | Model predicts x₀, not ε |
| `learn_sigma` | `False` | Fixed variance |
| `sigma_small` | `True` | Use `FIXED_SMALL` posterior variance |
| Inference steps | 50 | Spaced from 1000 via `SpacedDiffusion` |

### Loss

| Term | λ | Description |
|------|---|-------------|
| `L_motion` | 1.0 | MSE on predicted x₀ vs. ground truth (masked) |
| `L_vel` | 0.5 (humanml) | MSE on finite-difference velocities |
| `L_rcxyz` | 0.0 | MSE on SMPL 3D positions (requires SMPL) |
| `L_foot_contact` | 0.0 | Foot-contact consistency at joints 7,8,10,11 |

---

## Input / Output Summary

```
Input  (per forward pass):
  x          : Tensor[B, J, nfeats, T]   — noisy motion at step t
  timesteps  : Tensor[B]                 — diffusion step indices
  y['text']  : List[str]  (optional)     — text conditions
  y['action']: Tensor[B]  (optional)     — action class indices
  y['mask']  : Tensor[B,1,1,T]           — valid-frame boolean mask

Output:
  x̂₀         : Tensor[B, J, nfeats, T]   — predicted clean motion
```

For this project's Data objects: **B = batch, J = 22, nfeats = 3, T ≤ 196**.
