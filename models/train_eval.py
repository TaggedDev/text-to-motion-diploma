from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from tqdm import tqdm

try:
    from models.mdm import MDM, GaussianDiffusion, collate_motion_batch_hml
except ImportError:
    from mdm import MDM, GaussianDiffusion, collate_motion_batch_hml  # type: ignore[no-redef]


# ── HumanML3D feature-vector layout ──────────────────────────────────────────
# [0:4]    root: angular vel, linear vel (x,z), height
# [4:67]   relative joint positions  (21 joints × 3)
# [67:130] joint velocities          (21 joints × 3)
# [130:256] 6D joint rotations       (21 joints × 6)
# [256:260] foot contacts            (4 binary)
# [260:263] root linear vel (global) (3)

_HML_DIM = 263
_N_JOINTS = 22
_POS_SLICE = slice(4, 67)    # 63 dims — relative XYZ positions
_VEL_SLICE = slice(67, 130)  # 63 dims — joint velocities


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4
    checkpoint_every_k: int = 1
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/checkpoints"))
    num_inference_steps: int = 50
    n_eval_samples: int = 256    # sequences to generate for FID / diversity
    eval_seq_len: int = 60       # fixed frame length for generated sequences
    recon_noise_frac: float = 0.5  # noise level (as fraction of T) for APE / AVE
    num_workers: int = 0


# ── Epoch metrics ─────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    epoch: int
    # Training reconstruction loss
    train_loss: float
    train_rot_mse: float
    train_vel_mse: float
    # Validation reconstruction loss
    val_loss: float
    val_rot_mse: float
    val_vel_mse: float
    # Generative quality (sampling-based)
    fid: float        # Fréchet Inception Distance — real vs generated feature distributions
    diversity: float  # mean pairwise L2 distance between generated sequences
    # Reconstruction precision (partial-denoise on val set)
    ape: float  # Average Position Error on joint positions (feature indices 4:67)
    ave: float  # Average Velocity Error on joint velocities (feature indices 67:130)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MotionDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Data:
        raw = np.load(self.paths[idx]).astype(np.float32)  # [T, 263]
        T = raw.shape[0]
        root_pos = np.stack([np.zeros(T), raw[:, 3], np.zeros(T)], axis=1)[:, np.newaxis]
        joints_1_21 = raw[:, 4:67].reshape(T, 21, 3)
        joints = np.concatenate([root_pos, joints_1_21], axis=1)  # [T, 22, 3]
        return Data(
            pos=torch.from_numpy(joints.reshape(T * _N_JOINTS, 3)),
            x=torch.from_numpy(raw),
            seq_len=T,
            num_joints=_N_JOINTS,
        )


def _load_split(split_file: Path, data_dir: Path) -> list[Path]:
    ids = split_file.read_text(encoding="utf-8").splitlines()
    paths = [data_dir / f"{i.strip()}.npy" for i in ids if i.strip()]
    return [p for p in paths if p.exists()]


def make_dataloaders(
    data_dir: Path,
    humanml_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_paths = _load_split(humanml_dir / "train.txt", data_dir)
    val_paths = _load_split(humanml_dir / "val.txt", data_dir)
    test_paths = _load_split(humanml_dir / "test.txt", data_dir)
    pin = torch.cuda.is_available()

    def _loader(paths: list[Path], shuffle: bool) -> DataLoader:
        return DataLoader(
            MotionDataset(paths),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_motion_batch_hml,
            pin_memory=pin,
        )

    return _loader(train_paths, True), _loader(val_paths, False), _loader(test_paths, False)


# ── Individual metric functions ───────────────────────────────────────────────

def compute_fid(real_feats: np.ndarray, gen_feats: np.ndarray, eps: float = 1e-6) -> float:
    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sig_r = np.cov(real_feats, rowvar=False) + eps * np.eye(real_feats.shape[1])
    sig_g = np.cov(gen_feats, rowvar=False) + eps * np.eye(gen_feats.shape[1])
    diff = mu_r - mu_g
    covmean, _ = scipy.linalg.sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))


def compute_diversity(gen_feats: np.ndarray, n_pairs: int = 300) -> float:
    n = len(gen_feats)
    if n < 2:
        return 0.0
    k = min(n, n_pairs)
    rng = np.random.default_rng()
    i1 = rng.choice(n, size=k, replace=False)
    i2 = rng.choice(n, size=k, replace=False)
    return float(np.linalg.norm(gen_feats[i1] - gen_feats[i2], axis=1).mean())


def compute_ape(recon_feats: np.ndarray, gt_feats: np.ndarray) -> float:
    return float(np.linalg.norm(recon_feats[:, _POS_SLICE] - gt_feats[:, _POS_SLICE], axis=1).mean())


def compute_ave(recon_feats: np.ndarray, gt_feats: np.ndarray) -> float:
    return float(np.linalg.norm(recon_feats[:, _VEL_SLICE] - gt_feats[:, _VEL_SLICE], axis=1).mean())


# ── Internal helpers ──────────────────────────────────────────────────────────

def _mean_pool(x: Tensor, mask: Tensor) -> np.ndarray:
    # x: [B, 263, 1, T], mask: [B, 1, 1, T] → [B, 263] mean over valid frames
    feats = []
    for i in range(x.shape[0]):
        T = int(mask[i, 0, 0].sum())
        feats.append(x[i, :, 0, :T].mean(dim=1).cpu().numpy())
    return np.stack(feats)


def _train_epoch(
    model: MDM,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "rot_mse": 0.0, "vel_mse": 0.0}
    n = 0

    with tqdm(loader, desc=f"Epoch {epoch:>3} │ train", leave=False, unit="batch") as bar:
        for x, mask in bar:
            x, mask = x.to(device), mask.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            losses = diffusion.training_losses(model, x, t, {"y": {"mask": mask}})

            optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            totals["loss"] += losses["loss"].item()
            totals["rot_mse"] += losses["rot_mse"].item()
            totals["vel_mse"] += losses.get("vel_mse", losses["rot_mse"]).item()
            n += 1
            bar.set_postfix(loss=f"{losses['loss'].item():.4f}")

    return {f"train_{k}": v / n for k, v in totals.items()}


@torch.no_grad()
def _eval_epoch(
    model: MDM,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "rot_mse": 0.0, "vel_mse": 0.0}
    n = 0

    with tqdm(loader, desc=f"Epoch {epoch:>3} │ val  ", leave=False, unit="batch") as bar:
        for x, mask in bar:
            x, mask = x.to(device), mask.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            losses = diffusion.training_losses(model, x, t, {"y": {"mask": mask}})
            totals["loss"] += losses["loss"].item()
            totals["rot_mse"] += losses["rot_mse"].item()
            totals["vel_mse"] += losses.get("vel_mse", losses["rot_mse"]).item()
            n += 1
            bar.set_postfix(loss=f"{losses['loss'].item():.4f}")

    return {f"val_{k}": v / n for k, v in totals.items()}


@torch.no_grad()
def _collect_real_feats(loader: DataLoader, max_samples: int = 2048) -> np.ndarray:
    feats: list[np.ndarray] = []
    for x, mask in loader:
        feats.append(_mean_pool(x, mask))
        if sum(len(f) for f in feats) >= max_samples:
            break
    return np.concatenate(feats)[:max_samples]


@torch.no_grad()
def _generate_feats(
    model: MDM,
    diffusion: GaussianDiffusion,
    n_samples: int,
    seq_len: int,
    num_inference_steps: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    feats: list[np.ndarray] = []
    done = 0

    with tqdm(total=n_samples, desc="  Generating   ", leave=False, unit="seq") as bar:
        while done < n_samples:
            b = min(batch_size, n_samples - done)
            x = diffusion.sample(
                model, (b, _HML_DIM, 1, seq_len),
                num_inference_steps=num_inference_steps, device=device,
            )
            feats.append(x[:, :, 0, :].mean(dim=2).cpu().numpy())  # [b, 263]
            done += b
            bar.update(b)

    return np.concatenate(feats)


@torch.no_grad()
def _collect_reconstruction_feats(
    model: MDM,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    noise_frac: float,
    max_samples: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    t_val = int(diffusion.num_timesteps * noise_frac)
    recon_list: list[np.ndarray] = []
    gt_list: list[np.ndarray] = []

    with tqdm(loader, desc="  Reconstruction", leave=False, unit="batch") as bar:
        for x, mask in bar:
            if len(recon_list) * loader.batch_size >= max_samples:
                break
            x, mask = x.to(device), mask.to(device)
            t = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
            xt = diffusion.q_sample(x, t)
            x0_pred = model(xt, t, {"mask": mask})
            recon_list.append(_mean_pool(x0_pred, mask))
            gt_list.append(_mean_pool(x, mask))

    recon = np.concatenate(recon_list)[:max_samples]
    gt = np.concatenate(gt_list)[:max_samples]
    return recon, gt


def _compute_all_metrics(
    model: MDM,
    diffusion: GaussianDiffusion,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    real_feats = _collect_real_feats(val_loader)
    gen_feats = _generate_feats(
        model, diffusion, cfg.n_eval_samples, cfg.eval_seq_len,
        cfg.num_inference_steps, device, cfg.batch_size,
    )
    recon_feats, gt_feats = _collect_reconstruction_feats(
        model, diffusion, val_loader, device, cfg.recon_noise_frac,
    )
    return {
        "fid": compute_fid(real_feats, gen_feats),
        "diversity": compute_diversity(gen_feats),
        "ape": compute_ape(recon_feats, gt_feats),
        "ave": compute_ave(recon_feats, gt_feats),
    }


def _save_checkpoint(
    model: MDM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
    cfg: TrainConfig,
    run_ts: str,
) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.checkpoint_dir / f"mdm-{run_ts}-ep{epoch:03d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dataclasses.asdict(metrics),
        },
        path,
    )
    tqdm.write(f"  ✓ checkpoint → {path.name}")


# ── Public API ────────────────────────────────────────────────────────────────

def train(
    model: MDM,
    diffusion: GaussianDiffusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> list[EpochMetrics]:
    run_ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    optimizer = torch.optim.Adam(model.parameters_wo_clip(), lr=cfg.lr)
    history: list[EpochMetrics] = []

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training", unit="epoch"):
        train_stats = _train_epoch(model, diffusion, train_loader, optimizer, device, epoch)
        val_stats = _eval_epoch(model, diffusion, val_loader, device, epoch)
        gen_stats = _compute_all_metrics(model, diffusion, val_loader, device, cfg)

        m = EpochMetrics(
            epoch=epoch,
            train_loss=train_stats["train_loss"],
            train_rot_mse=train_stats["train_rot_mse"],
            train_vel_mse=train_stats["train_vel_mse"],
            val_loss=val_stats["val_loss"],
            val_rot_mse=val_stats["val_rot_mse"],
            val_vel_mse=val_stats["val_vel_mse"],
            fid=gen_stats["fid"],
            diversity=gen_stats["diversity"],
            ape=gen_stats["ape"],
            ave=gen_stats["ave"],
        )
        history.append(m)

        tqdm.write(
            f"Epoch {epoch:>3} │ loss {m.train_loss:.4f} │ val {m.val_loss:.4f} "
            f"│ FID {m.fid:.1f} │ div {m.diversity:.3f} │ APE {m.ape:.4f} │ AVE {m.ave:.4f}"
        )

        if epoch % cfg.checkpoint_every_k == 0:
            _save_checkpoint(model, optimizer, epoch, m, cfg, run_ts)

    return history
