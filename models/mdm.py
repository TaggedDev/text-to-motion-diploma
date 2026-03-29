from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

try:
    import clip as openai_clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class MDMConfig:
    njoints: int = 22
    nfeats: int = 3
    latent_dim: int = 512
    ff_size: int = 1024
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    cond_mode: str = "no_cond"      # "no_cond" | "text" | "action"
    cond_mask_prob: float = 0.1
    num_actions: int = 0            # required when cond_mode == "action"
    clip_version: str = "ViT-B/32"  # required when cond_mode == "text"
    clip_dim: int = 512


@dataclass
class DiffusionConfig:
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"  # "cosine" | "linear"
    sigma_small: bool = True
    lambda_vel: float = 0.0


# ── Beta schedule ─────────────────────────────────────────────────────────────

def _cosine_betas(num_steps: int) -> np.ndarray:
    def _alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    return np.array(
        [min(1.0 - _alpha_bar((i + 1) / num_steps) / _alpha_bar(i / num_steps), 0.999)
         for i in range(num_steps)],
        dtype=np.float64,
    )


def _linear_betas(num_steps: int) -> np.ndarray:
    scale = 1000.0 / num_steps
    return np.linspace(scale * 1e-4, scale * 0.02, num_steps, dtype=np.float64)


def _make_betas(schedule: str, num_steps: int) -> np.ndarray:
    if schedule == "cosine":
        return _cosine_betas(num_steps)
    if schedule == "linear":
        return _linear_betas(num_steps)
    raise ValueError(f"Unknown noise schedule {schedule!r}. Choose 'cosine' or 'linear'.")


def _posterior_params(
    betas: np.ndarray,
    acp: np.ndarray,       # ᾱ_t
    acp_prev: np.ndarray,  # ᾱ_{t-1}
    sigma_small: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = 1.0 - betas
    post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
    log_var = (
        np.log(np.append(post_var[1], post_var[1:]))
        if sigma_small
        else np.log(np.maximum(post_var, 1e-20))
    )
    coef1 = betas * np.sqrt(acp_prev) / (1.0 - acp)
    coef2 = (1.0 - acp_prev) * np.sqrt(alphas) / (1.0 - acp)
    return coef1, coef2, log_var


# ── Data collation ────────────────────────────────────────────────────────────

def collate_motion_batch(
    graphs: list[Data],
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Converts a list of Data objects into a padded motion tensor and frame mask.

    Returns:
        x    : [B, J, 3, T_max]   padded XYZ joint positions
        mask : [B, 1, 1, T_max]   True for valid frames
    """
    B = len(graphs)
    J = int(graphs[0].num_joints)
    seq_lens = [int(g.seq_len) for g in graphs]
    T_max = max(seq_lens)

    x = torch.zeros(B, J, 3, T_max)
    mask = torch.zeros(B, 1, 1, T_max, dtype=torch.bool)

    for i, (g, T) in enumerate(zip(graphs, seq_lens)):
        pos = g.pos.reshape(T, J, 3).permute(1, 2, 0)  # [J, 3, T]
        x[i, :, :, :T] = pos
        mask[i, 0, 0, :T] = True

    if device is not None:
        x, mask = x.to(device), mask.to(device)
    return x, mask


def collate_motion_batch_hml(
    graphs: list[Data],
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Collate for hml_vec format (263-dim feature vectors).

    Returns:
        x    : [B, 263, 1, T_max]   padded feature vectors
        mask : [B, 1,   1, T_max]   True for valid frames
    """
    B = len(graphs)
    seq_lens = [int(g.seq_len) for g in graphs]
    T_max = max(seq_lens)

    x = torch.zeros(B, 263, 1, T_max)
    mask = torch.zeros(B, 1, 1, T_max, dtype=torch.bool)

    for i, (g, T) in enumerate(zip(graphs, seq_lens)):
        x[i, :, 0, :T] = g.x[:T].T  # [T, 263].T → [263, T]
        mask[i, 0, 0, :T] = True

    if device is not None:
        x, mask = x.to(device), mask.to(device)
    return x, mask


# ── Model submodules ──────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    pe: Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))  # [max_len, 1, d]

    def forward(self, x: Tensor) -> Tensor:  # [S, B, d] → [S, B, d]
        return self.dropout(x + self.pe[: x.shape[0]])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim: int, pos_encoder: PositionalEncoding) -> None:
        super().__init__()
        self.pos_encoder = pos_encoder
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, t: Tensor) -> Tensor:  # [B] → [1, B, d]
        return self.mlp(self.pos_encoder.pe[t]).permute(1, 0, 2)  # pe[t]: [B, 1, d]


class InputProcess(nn.Module):
    def __init__(self, input_feats: int, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_feats, latent_dim)

    def forward(self, x: Tensor) -> Tensor:  # [B, J, F, T] → [T, B, d]
        B, J, F, T = x.shape
        return self.proj(x.permute(3, 0, 1, 2).reshape(T, B, J * F))


class OutputProcess(nn.Module):
    def __init__(self, latent_dim: int, njoints: int, nfeats: int) -> None:
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.proj = nn.Linear(latent_dim, njoints * nfeats)

    def forward(self, x: Tensor) -> Tensor:  # [T, B, d] → [B, J, F, T]
        T, B, _ = x.shape
        return self.proj(x).reshape(T, B, self.njoints, self.nfeats).permute(1, 2, 3, 0)


class EmbedAction(nn.Module):
    def __init__(self, num_actions: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_actions, latent_dim)

    def forward(self, ids: Tensor) -> Tensor:  # [B] → [1, B, d]
        return self.embedding(ids).unsqueeze(0)


# ── MDM ───────────────────────────────────────────────────────────────────────

class MDM(nn.Module):
    def __init__(self, cfg: MDMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_process = InputProcess(cfg.njoints * cfg.nfeats, cfg.latent_dim)
        self.pos_encoder = PositionalEncoding(cfg.latent_dim, cfg.dropout)
        self.timestep_embedder = TimestepEmbedder(cfg.latent_dim, self.pos_encoder)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.latent_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.ff_size,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=False,
            ),
            num_layers=cfg.num_layers,
        )
        self.output_process = OutputProcess(cfg.latent_dim, cfg.njoints, cfg.nfeats)

        if cfg.cond_mode == "text":
            assert _CLIP_AVAILABLE, "Install openai-clip to use text conditioning"
            self.clip_model = self._load_frozen_clip(cfg.clip_version)
            self.text_proj = nn.Linear(cfg.clip_dim, cfg.latent_dim)
        elif cfg.cond_mode == "action":
            assert cfg.num_actions > 0, "Set num_actions > 0 for action conditioning"
            self.action_embedder = EmbedAction(cfg.num_actions, cfg.latent_dim)

    @staticmethod
    def _load_frozen_clip(version: str) -> nn.Module:
        model, _ = openai_clip.load(version, device="cpu", jit=False)
        openai_clip.model.convert_weights(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _encode_text(self, texts: list[str]) -> Tensor:  # → [1, B, clip_dim]
        device = next(self.parameters()).device
        tokens = openai_clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            return self.clip_model.encode_text(tokens).float().unsqueeze(0)

    def _mask_cond(self, cond: Tensor, force_mask: bool = False) -> Tensor:
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cfg.cond_mask_prob > 0.0:
            B = cond.shape[1]
            keep = 1.0 - torch.bernoulli(
                torch.full((B,), self.cfg.cond_mask_prob, device=cond.device)
            ).view(1, B, 1)
            return cond * keep
        return cond

    def _build_cond_token(self, t: Tensor, y: dict) -> Tensor:  # → [1, B, d]
        time_emb = self.timestep_embedder(t)
        force = y.get("uncond", False)

        if self.cfg.cond_mode == "text":
            text_enc = self._encode_text(y["text"])  # [1, B, clip_dim]
            return time_emb + self.text_proj(self._mask_cond(text_enc, force))

        if self.cfg.cond_mode == "action":
            return time_emb + self._mask_cond(self.action_embedder(y["action"]), force)

        return time_emb

    def _build_src_key_padding_mask(
        self, x: Tensor, frame_mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        if frame_mask is None:
            return None
        B, T = x.shape[0], x.shape[3]
        frame_ignore = ~frame_mask[:, 0, 0, :T]                           # [B, T]
        cond_keep = torch.zeros(B, 1, dtype=torch.bool, device=x.device)  # [B, 1]
        return torch.cat([cond_keep, frame_ignore], dim=1)                 # [B, T+1]

    def forward(self, x: Tensor, t: Tensor, y: Optional[dict] = None) -> Tensor:
        """
        x : [B, J, nfeats, T]  noisy motion at diffusion step t
        t : [B]                diffusion timestep indices
        y : optional dict with keys:
              mask    [B, 1, 1, T] bool  — valid-frame mask
              text    list[str]          — text prompts (cond_mode == "text")
              action  Tensor[B]  long   — action class ids (cond_mode == "action")
              uncond  bool               — force-zero conditioning for CFG
        Returns: [B, J, nfeats, T]  predicted clean motion x̂₀
        """
        y = y or {}
        cond_token = self._build_cond_token(t, y)                         # [1, B, d]
        x_proj = self.input_process(x)                                    # [T, B, d]
        seq = self.pos_encoder(torch.cat([cond_token, x_proj], dim=0))    # [T+1, B, d]
        padding_mask = self._build_src_key_padding_mask(x, y.get("mask"))
        out = self.transformer(seq, src_key_padding_mask=padding_mask)
        return self.output_process(out[1:])                               # [B, J, F, T]

    def parameters_wo_clip(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]


# ── Loss helpers ──────────────────────────────────────────────────────────────

def _masked_mse(pred: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
    loss = F.mse_loss(pred, target, reduction="none")  # [B, J, F, T]
    if mask is None:
        return loss.mean()
    expanded = mask.expand_as(loss)
    return (loss * expanded).sum() / expanded.sum().clamp(min=1)


def _velocity_mse(pred: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
    return _masked_mse(
        pred[..., 1:] - pred[..., :-1],
        target[..., 1:] - target[..., :-1],
        mask[..., 1:] if mask is not None else None,
    )


# ── Gaussian diffusion ────────────────────────────────────────────────────────

class GaussianDiffusion:
    def __init__(self, cfg: DiffusionConfig) -> None:
        betas = _make_betas(cfg.noise_schedule, cfg.diffusion_steps)
        acp = np.cumprod(1.0 - betas)        # ᾱ_t
        acp_prev = np.append(1.0, acp[:-1])  # ᾱ_{t-1}

        self.num_timesteps = len(betas)
        self.lambda_vel = cfg.lambda_vel
        self._sigma_small = cfg.sigma_small
        self._noise_schedule = cfg.noise_schedule

        self._sqrt_acp = torch.tensor(np.sqrt(acp), dtype=torch.float32)
        self._sqrt_one_minus_acp = torch.tensor(np.sqrt(1.0 - acp), dtype=torch.float32)

        coef1, coef2, log_var = _posterior_params(betas, acp, acp_prev, cfg.sigma_small)
        self._post_coef1 = torch.tensor(coef1, dtype=torch.float32)
        self._post_coef2 = torch.tensor(coef2, dtype=torch.float32)
        self._post_log_var = torch.tensor(log_var, dtype=torch.float32)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        noise = noise if noise is not None else torch.randn_like(x0)
        shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        t_cpu = t.cpu()
        c1 = self._sqrt_acp[t_cpu].to(x0.device).reshape(shape)
        c2 = self._sqrt_one_minus_acp[t_cpu].to(x0.device).reshape(shape)
        return c1 * x0 + c2 * noise

    def training_losses(
        self,
        model: MDM,
        x0: Tensor,
        t: Tensor,
        model_kwargs: Optional[dict] = None,
    ) -> dict[str, Tensor]:
        y = (model_kwargs or {}).get("y")
        mask = (y or {}).get("mask")

        x0_pred = model(self.q_sample(x0, t), t, y)
        rot_mse = _masked_mse(x0_pred, x0, mask)
        losses: dict[str, Tensor] = {"rot_mse": rot_mse}

        if self.lambda_vel > 0.0:
            losses["vel_mse"] = _velocity_mse(x0_pred, x0, mask)
            losses["loss"] = rot_mse + self.lambda_vel * losses["vel_mse"]
        else:
            losses["loss"] = rot_mse

        return losses

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: MDM,
        shape: tuple[int, ...],
        model_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ) -> Tensor:
        if device is None:
            device = next(model.parameters()).device
        xt = torch.randn(shape, device=device)
        timesteps: Iterable[int] = range(self.num_timesteps - 1, -1, -1)

        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling", total=self.num_timesteps)

        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x0_pred = model(xt, t_batch, (model_kwargs or {}).get("y")).clamp(-1.0, 1.0)
            c1 = float(self._post_coef1[t])
            c2 = float(self._post_coef2[t])
            lv = float(self._post_log_var[t])
            noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            xt = c1 * x0_pred + c2 * xt + math.exp(0.5 * lv) * noise

        return xt

    @torch.no_grad()
    def sample(
        self,
        model: MDM,
        shape: tuple[int, ...],
        model_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        num_inference_steps: int = 50,
        progress: bool = False,
    ) -> Tensor:
        if num_inference_steps >= self.num_timesteps:
            return self.p_sample_loop(model, shape, model_kwargs, device, progress)
        return _SpacedSampler(self, num_inference_steps).sample(
            model, shape, model_kwargs, device, progress
        )


# ── Spaced sampler (fast inference) ──────────────────────────────────────────

class _SpacedSampler:
    def __init__(self, base: GaussianDiffusion, num_steps: int) -> None:
        T = base.num_timesteps
        step = max(T // num_steps, 1)
        self.spaced_t = list(range(0, T, step))[:num_steps]  # ascending original indices

        orig_acp = (base._sqrt_acp ** 2).numpy()  # ᾱ from original schedule
        acp = np.array([orig_acp[t] for t in self.spaced_t])
        acp_prev = np.concatenate([[1.0], acp[:-1]])
        betas = np.clip(1.0 - acp / acp_prev, 0.0, 0.999)

        coef1, coef2, log_var = _posterior_params(betas, acp, acp_prev, sigma_small=True)
        self._post_coef1 = torch.tensor(coef1, dtype=torch.float32)
        self._post_coef2 = torch.tensor(coef2, dtype=torch.float32)
        self._post_log_var = torch.tensor(log_var, dtype=torch.float32)

    @torch.no_grad()
    def sample(
        self,
        model: MDM,
        shape: tuple[int, ...],
        model_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ) -> Tensor:
        if device is None:
            device = next(model.parameters()).device
        xt = torch.randn(shape, device=device)
        K = len(self.spaced_t)
        steps: Iterable[tuple[int, int]] = list(enumerate(reversed(self.spaced_t)))

        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Sampling", total=K)

        for k, t in steps:
            idx = K - 1 - k  # maps reverse iteration index → ascending posterior index
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x0_pred = model(xt, t_batch, (model_kwargs or {}).get("y")).clamp(-1.0, 1.0)
            c1 = float(self._post_coef1[idx])
            c2 = float(self._post_coef2[idx])
            lv = float(self._post_log_var[idx])
            noise = torch.randn_like(xt) if k < K - 1 else torch.zeros_like(xt)
            xt = c1 * x0_pred + c2 * xt + math.exp(0.5 * lv) * noise

        return xt
