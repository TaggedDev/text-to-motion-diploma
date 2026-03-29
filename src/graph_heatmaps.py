from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import plotly.graph_objects as go
import torch
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from torch_geometric.data import Data
from tqdm import tqdm

from display import _load_joints


_PLANES: Final[tuple[tuple[str, str, str, int, int], ...]] = (
    ("XY plane", "X (m)", "Y (m)", 0, 1),
    ("XZ plane", "X (m)", "Z (m)", 0, 2),
    ("YZ plane", "Y (m)", "Z (m)", 1, 2),
)

_COLORAXIS_IDS: Final[tuple[str, str, str]] = ("coloraxis", "coloraxis2", "coloraxis3")
_COLORBAR_X: Final[tuple[float, float, float]] = (0.27, 0.64, 1.02)
_COLORSCALE: Final[str] = "Hot"
_N_BINS: Final[int] = 60


class PlaneHistograms(NamedTuple):
    counts: NDArray[np.float64]        # (3, N_BINS, N_BINS) — density per plane
    axis_centers: NDArray[np.float64]  # (3, N_BINS)  — bin centres for X, Y, Z axes


def _points(graph: Data) -> NDArray[np.float32]:
    return graph.pos.numpy()  # type: ignore[return-value]


def _axis_ranges(points: NDArray[np.float32]) -> list[tuple[float, float]]:
    return [(float(points[:, i].min()), float(points[:, i].max())) for i in range(3)]


def _bin_centers(low: float, high: float) -> NDArray[np.float64]:
    edges = np.linspace(low, high, _N_BINS + 1)
    return (edges[:-1] + edges[1:]) / 2  # type: ignore[return-value]


def _compute_counts(
    points: NDArray[np.float32],
    ranges: list[tuple[float, float]],
) -> NDArray[np.float64]:
    counts: NDArray[np.float64] = np.zeros((3, _N_BINS, _N_BINS))
    for plane_idx, (_, _, _, xi, yi) in enumerate(_PLANES):
        h, _, _ = np.histogram2d(
            points[:, xi], points[:, yi],
            bins=_N_BINS,
            range=[ranges[xi], ranges[yi]],
        )
        counts[plane_idx] = h.T  # rows = y, cols = x
    return counts


def _make_histograms(
    counts: NDArray[np.float64],
    ranges: list[tuple[float, float]],
) -> PlaneHistograms:
    axis_centers: NDArray[np.float64] = np.array([_bin_centers(*r) for r in ranges])
    return PlaneHistograms(counts=counts, axis_centers=axis_centers)


def load_graph(npy_path: Path) -> Data:
    raw = np.load(npy_path)           # (T, 263) or (T, J, 3)
    joints = _load_joints(npy_path)   # (T, J, 3)
    n_frames, n_joints, _ = joints.shape
    pos = torch.tensor(joints.reshape(n_frames * n_joints, 3), dtype=torch.float32)
    kwargs: dict = {"pos": pos, "seq_len": n_frames, "num_joints": n_joints}
    if raw.ndim == 2 and raw.shape[1] == 263:
        kwargs["x"] = torch.tensor(raw, dtype=torch.float32)  # [T, 263] full feature vector
    return Data(**kwargs)


def compute_single(graph: Data) -> PlaneHistograms:
    pts = _points(graph)
    ranges = _axis_ranges(pts)
    return _make_histograms(_compute_counts(pts, ranges), ranges)


def compute_batch(graphs: list[Data]) -> PlaneHistograms:
    all_pts = [_points(g) for g in tqdm(graphs, desc="Extracting points")]

    ranges = _axis_ranges(np.concatenate(all_pts, axis=0))

    counts_list = [
        _compute_counts(pts, ranges)
        for pts in tqdm(all_pts, desc="Computing histograms")
    ]

    mean_counts: NDArray[np.float64] = np.stack(counts_list).mean(axis=0)

    return _make_histograms(mean_counts, ranges)

def display_heatmap(data: PlaneHistograms) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[plane[0] for plane in _PLANES],
        horizontal_spacing=0.12,
    )

    for col, (_, x_label, y_label, xi, yi) in enumerate(_PLANES, start=1):
        fig.add_trace(
            go.Heatmap(
                z=data.counts[col - 1],
                x=data.axis_centers[xi],
                y=data.axis_centers[yi],
                coloraxis=_COLORAXIS_IDS[col - 1],
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text=x_label, row=1, col=col)
        fig.update_yaxes(title_text=y_label, row=1, col=col)

    coloraxis_configs: dict[str, object] = {
        ca_id: dict(colorscale=_COLORSCALE, colorbar=dict(x=x_pos, len=0.85, thickness=12))
        for ca_id, x_pos in zip(_COLORAXIS_IDS, _COLORBAR_X)
    }

    fig.update_layout(
        title="Joint position density per plane",
        height=480,
        showlegend=False,
        **coloraxis_configs,
    )

    return fig
