
from pathlib import Path
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from numpy.typing import NDArray

FPS = 20
_PAD = 0.15


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
        n = shape[0]
        root = np.stack([np.zeros(n), data[:, 3], np.zeros(n)], axis=1)[:, np.newaxis, :]
        joints_1_21 = np.reshape(data[:, 4:67], (n, 21, 3))
        return np.concatenate([root, joints_1_21], axis=1)  # (T, 22, 3)

    raise ValueError(f"Unsupported shape {data.shape}. Expected (T, 22, 3) or (T, 263).")


def _compute_limits(joints: NDArray[Any]) -> tuple[tuple[float, float], ...]:
    """Compute axis limits from joints of shape (..., 22, 3)."""
    joints = np.asarray(joints)
    return (
        (float(np.min(joints[..., 0])) - _PAD, float(np.max(joints[..., 0])) + _PAD),
        (float(np.min(joints[..., 2])) - _PAD, float(np.max(joints[..., 2])) + _PAD),
        (float(np.min(joints[..., 1])) - _PAD, float(np.max(joints[..., 1])) + _PAD),
    )


def _draw_on_ax(
    ax: plt.Axes,
    pos: NDArray[Any],
    title: str = "",
    limits: Optional[tuple[tuple[float, float], ...]] = None,
) -> None:
    xlim, ylim, zlim = limits if limits is not None else _compute_limits(pos)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], s=40, depthshade=False)
    ax.view_init(elev=20, azim=-90)
    if title:
        ax.set_title(title)


def draw_frame(npy_path: Path, frame_idx: int = 0) -> MotionDisplay:
    joints = _load_joints(npy_path)
    n_frames = len(joints)
    if not (0 <= frame_idx < n_frames):
        raise IndexError(f"frame_idx {frame_idx} out of range for T={n_frames}")

    fig: Figure = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_on_ax(ax, joints[frame_idx], title=f"{npy_path.stem}  —  frame {frame_idx}")
    fig.tight_layout()
    plt.close(fig)
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

    slice_joints = joints[start:end]
    limits = _compute_limits(slice_joints)

    fig: Figure = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def _animate(i: int) -> None:
        ax.cla()
        _draw_on_ax(
            ax, slice_joints[i],
            title=f"{npy_path.stem}  —  frame {start + i} / {n_frames}",
            limits=limits,
        )

    anim = FuncAnimation(
        fig, _animate,
        frames=end - start,
        interval=int(1000 / FPS),
        cache_frame_data=False,
    )
    plt.close(fig)
    return MotionDisplay(anim)
