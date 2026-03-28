
from pathlib import Path
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from numpy.typing import NDArray

FPS = 20


class MotionDisplay:
    def __init__(self, content: Union[Figure, FuncAnimation]) -> None:
        self._content = content

    def show(self) -> None:
        try:
            from IPython.display import display, HTML  
            if isinstance(self._content, FuncAnimation):
                display(HTML(self._content.to_jshtml()))
            else:
                display(self._content)
        except ImportError:
            plt.show()

    def _repr_html_(self) -> Optional[str]:
        if isinstance(self._content, FuncAnimation):
            return self._content.to_jshtml()
        return None

    def _repr_png_(self) -> Optional[bytes]:
        if isinstance(self._content, Figure):
            import io
            buf = io.BytesIO()
            self._content.savefig(buf, format="png", bbox_inches="tight")
            return buf.getvalue()
        return None


def _load_joints(npy_path: Path) -> NDArray[Any]:
    data: NDArray[Any] = np.load(npy_path)
    shape = cast(tuple[int, ...], data.shape)

    if data.ndim == 3 and shape[1] == 22 and shape[2] == 3:
        return data

    if data.ndim == 2 and shape[1] == 263:
        # index 3: root height (y); indices 4:67: joints 1-21 (x, y, z)
        n = shape[0]
        root = np.stack([np.zeros(n), data[:, 3], np.zeros(n)], axis=1)[:, np.newaxis, :]
        joints_1_21 = np.reshape(data[:, 4:67], (n, 21, 3))
        return np.concatenate([root, joints_1_21], axis=1)  # (T, 22, 3)

    raise ValueError(
        f"Unsupported shape {data.shape}. Expected (T, 22, 3) or (T, 263)."
    )


def _plot_joints_on_ax(ax: plt.Axes, pos: NDArray[Any]) -> None:
    pad = 0.15
    x_lo = float(np.min(pos[:, 0])) - pad
    x_hi = float(np.max(pos[:, 0])) + pad
    y_lo = float(np.min(pos[:, 2])) - pad  # data Z → plot Y
    y_hi = float(np.max(pos[:, 2])) + pad
    z_lo = float(np.min(pos[:, 1])) - pad  # data Y → plot Z (up)
    z_hi = float(np.max(pos[:, 1])) + pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], s=40, depthshade=False)
    ax.view_init(elev=20, azim=-90)


def _display_first_frame(npy_path: Path) -> None:
    draw_frame(npy_path).show()


# ── Public API ────────────────────────────────────────────────────────────────

def draw_frame(npy_path: Path, frame_idx: int = 0) -> MotionDisplay:
    joints = _load_joints(npy_path)
    n_frames = len(joints)
    if not (0 <= frame_idx < n_frames):
        raise IndexError(f"frame_idx {frame_idx} out of range for T={n_frames}")

    fig: Figure = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    pos: NDArray[Any] = np.asarray(joints[frame_idx])
    _plot_joints_on_ax(ax, pos)
    ax.set_title(f"{npy_path.stem}  —  frame {frame_idx}")
    fig.tight_layout()
    plt.close(fig)  # detach from pyplot to prevent Jupyter auto-display
    return MotionDisplay(fig)


def draw_frame_slice(
    npy_path: Path,
    start: int = 0,
    end: Optional[int] = None,
) -> MotionDisplay: 
    joints = _load_joints(npy_path)
    n_frames = len(joints)
    end = n_frames if end is None else end

    if not (0 <= start < end <= n_frames):
        raise ValueError(f"Invalid slice [{start}:{end}] for T={n_frames}")

    # Fixed axis limits across the full slice for a stable animation
    slice_joints = joints[start:end]
    pad = 0.15
    x_lo = float(np.min(slice_joints[:, :, 0])) - pad
    x_hi = float(np.max(slice_joints[:, :, 0])) + pad
    y_lo = float(np.min(slice_joints[:, :, 2])) - pad
    y_hi = float(np.max(slice_joints[:, :, 2])) + pad
    z_lo = float(np.min(slice_joints[:, :, 1])) - pad
    z_hi = float(np.max(slice_joints[:, :, 1])) + pad

    fig: Figure = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def _animate(i: int) -> None:
        ax.cla()
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_zlim(z_lo, z_hi)
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
        pos: NDArray[Any] = np.asarray(slice_joints[i])
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], s=40, depthshade=False)
        ax.view_init(elev=20, azim=-90)
        ax.set_title(f"{npy_path.stem}  —  frame {start + i} / {n_frames}")

    anim = FuncAnimation(
        fig, _animate,
        frames=end - start,
        interval=int(1000 / FPS),
        cache_frame_data=False,
    )
    plt.close(fig)
    return MotionDisplay(anim)
