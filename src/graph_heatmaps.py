from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch_geometric.data import Data

from display import _load_joints


def load_graph(npy_path: Path) -> Data:
    """Load a .npy motion file into a PyG Data object.

    Accepts both (T, 22, 3) joint arrays and (T, 263) feature vectors
    (the latter is converted to positions automatically via display._load_joints).

    Args:
        npy_path: Path to a .npy file from new_joints/ or new_joint_vecs/.

    Returns:
        Data with:
            pos (Tensor[float32]):  shape (T * num_joints, 3)
            seq_len (int):          number of frames T
            num_joints (int):       number of joints J
    """
    joints = _load_joints(npy_path)          # (T, J, 3)
    n_frames, n_joints, _ = joints.shape
    pos = torch.tensor(joints.reshape(n_frames * n_joints, 3), dtype=torch.float32)
    return Data(pos=pos, seq_len=n_frames, num_joints=n_joints)


# Each plane: (subplot title, x-axis label, y-axis label, x-coord index, y-coord index)
_PLANES: Final[tuple[tuple[str, str, str, int, int], ...]] = (
    ("XY plane", "X (m)", "Y (m)", 0, 1),
    ("XZ plane", "X (m)", "Z (m)", 0, 2),
    ("YZ plane", "Y (m)", "Z (m)", 1, 2),
)

_COLORAXIS_IDS: Final[tuple[str, str, str]] = ("coloraxis", "coloraxis2", "coloraxis3")
_COLORBAR_X: Final[tuple[float, float, float]] = (0.27, 0.64, 1.02)
_COLORSCALE: Final[str] = "Hot"
_N_BINS: Final[int] = 60


def _flat_points(graph: Data) -> np.ndarray:
    """Return all joint positions as a flat (T * num_joints, 3) array."""
    return graph.pos.numpy()


def _plane_density(
    points: np.ndarray,
    xi: int,
    yi: int,
    coloraxis_id: str,
) -> go.Histogram2d:
    """Build a 2D density histogram for one spatial plane.

    Args:
        points:       Array of shape (N, 3) — all joint positions.
        xi:           Column index for the horizontal axis (0=X, 1=Y, 2=Z).
        yi:           Column index for the vertical axis.
        coloraxis_id: Plotly coloraxis reference for shared colour scale.
    """
    return go.Histogram2d(
        x=points[:, xi],
        y=points[:, yi],
        nbinsx=_N_BINS,
        nbinsy=_N_BINS,
        coloraxis=coloraxis_id,
    )


def display_heatmap_single(graph: Data) -> go.Figure:
    """Display a 1×3 grid of spatial density heatmaps for one motion sequence.

    Each subplot shows a 2D histogram of joint positions projected onto one plane.
    Colour encodes visit frequency — cold (rare) → hot (frequent).

    Args:
        graph: PyG Data with:
            pos (Tensor[float32]):  shape (T * num_joints, 3)
            seq_len (int):          number of frames T
            num_joints (int):       number of joints J
    """
    points = _flat_points(graph)  # (T*J, 3)

    titles = [plane[0] for plane in _PLANES]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.12)

    for col, (_, x_label, y_label, xi, yi) in enumerate(_PLANES, start=1):
        coloraxis_id = _COLORAXIS_IDS[col - 1]
        fig.add_trace(_plane_density(points, xi, yi, coloraxis_id), row=1, col=col)
        fig.update_xaxes(title_text=x_label, row=1, col=col)
        fig.update_yaxes(title_text=y_label, row=1, col=col)

    coloraxis_configs: dict[str, object] = {
        ca_id: dict(
            colorscale=_COLORSCALE,
            colorbar=dict(x=x_pos, len=0.85, thickness=12),
        )
        for ca_id, x_pos in zip(_COLORAXIS_IDS, _COLORBAR_X)
    }

    fig.update_layout(
        title="Joint position density per plane",
        height=480,
        showlegend=False,
        **coloraxis_configs,
    )

    return fig
