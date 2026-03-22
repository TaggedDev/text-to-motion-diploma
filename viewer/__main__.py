"""
HumanML3D Animation Viewer
Renders skeleton animations from new_joints/ using matplotlib + tkinter.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parents[1] / "data" / "HumanML3D" / "humanml"
JOINTS_DIR = DATA_DIR / "new_joints"
TEXTS_DIR  = DATA_DIR / "texts"
ALL_IDS    = DATA_DIR / "all.txt"

FPS = 20

# ── SMPL 22-joint skeleton bones ──────────────────────────────────────────────
# Joint indices: 0=pelvis, 1=L-hip, 2=R-hip, 3=spine1, 4=L-knee, 5=R-knee,
# 6=spine2, 7=L-ankle, 8=R-ankle, 9=spine3, 10=L-foot, 11=R-foot,
# 12=neck, 13=L-collar, 14=R-collar, 15=head, 16=L-shoulder, 17=R-shoulder,
# 18=L-elbow, 19=R-elbow, 20=L-wrist, 21=R-wrist
BONES = [
    (0, 1), (1, 4), (4, 7), (7, 10),   # left leg
    (0, 2), (2, 5), (5, 8), (8, 11),   # right leg
    (0, 3), (3, 6), (6, 9),             # spine
    (9, 12), (12, 15),                  # neck → head
    (9, 13), (13, 16), (16, 18), (18, 20),  # left arm
    (9, 14), (14, 17), (17, 19), (19, 21),  # right arm
]


def load_motion_ids():
    """Return sorted list of IDs that have a file in new_joints/."""
    with open(ALL_IDS) as f:
        ids = [line.strip() for line in f if line.strip()]
    return [mid for mid in ids if (JOINTS_DIR / f"{mid}.npy").exists()]


def load_joints(motion_id: str) -> np.ndarray:
    """Load (T, 22, 3) joint positions."""
    return np.load(JOINTS_DIR / f"{motion_id}.npy")


def load_descriptions(motion_id: str) -> list[str]:
    """Return list of plain-text descriptions for a motion ID."""
    path = TEXTS_DIR / f"{motion_id}.txt"
    if not path.exists():
        return []
    descriptions = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("#")
            if parts:
                descriptions.append(parts[0])
    return descriptions


# ── Main application ───────────────────────────────────────────────────────────
class ViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HumanML3D Viewer")
        self.geometry("1100x720")
        self.resizable(True, True)

        self._anim: FuncAnimation | None = None
        self._joints: np.ndarray | None = None
        self._playing = False
        self._frame_idx = 0

        self._build_ui()
        self._load_ids()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Left panel ────────────────────────────────────────────────────────
        left = tk.Frame(self, width=260, bg="#1e1e2e")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="Motion clips", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 11, "bold")).pack(pady=(8, 4))

        # Search box
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", self._on_search)
        search = tk.Entry(left, textvariable=self._search_var,
                          bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                          relief=tk.FLAT, font=("Segoe UI", 10))
        search.pack(fill=tk.X, padx=8, pady=(0, 6))

        # Listbox + scrollbar
        frame_lb = tk.Frame(left, bg="#1e1e2e")
        frame_lb.pack(fill=tk.BOTH, expand=True, padx=8)

        scrollbar = tk.Scrollbar(frame_lb)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._listbox = tk.Listbox(
            frame_lb,
            yscrollcommand=scrollbar.set,
            bg="#181825", fg="#cdd6f4", selectbackground="#89b4fa",
            selectforeground="#1e1e2e", activestyle="none",
            font=("Courier New", 9), relief=tk.FLAT, borderwidth=0,
        )
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_select)

        # Counter label
        self._count_label = tk.Label(left, text="", bg="#1e1e2e", fg="#6c7086",
                                     font=("Segoe UI", 8))
        self._count_label.pack(pady=(4, 8))

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(self, bg="#11111b")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Description label
        self._desc_label = tk.Label(
            right, text="Select a clip to begin.",
            bg="#11111b", fg="#a6adc8", font=("Segoe UI", 9),
            wraplength=780, justify=tk.LEFT, anchor="w",
        )
        self._desc_label.pack(fill=tk.X, padx=4, pady=(4, 0))

        # Matplotlib figure
        self._fig = plt.Figure(figsize=(8, 5.5), facecolor="#11111b")
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._style_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Controls bar
        ctrl = tk.Frame(right, bg="#11111b")
        ctrl.pack(fill=tk.X, pady=(4, 0))

        btn_kw = dict(bg="#313244", fg="#cdd6f4", relief=tk.FLAT,
                      font=("Segoe UI", 10), padx=12, pady=4,
                      activebackground="#45475a", activeforeground="#cdd6f4",
                      cursor="hand2")

        self._play_btn = tk.Button(ctrl, text="▶  Play",
                                   command=self._toggle_play, **btn_kw)
        self._play_btn.pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(ctrl, text="⏮  Restart",
                  command=self._restart, **btn_kw).pack(side=tk.LEFT, padx=(0, 6))

        # Frame counter
        self._frame_label = tk.Label(ctrl, text="Frame —", bg="#11111b",
                                     fg="#6c7086", font=("Segoe UI", 9))
        self._frame_label.pack(side=tk.LEFT, padx=8)

        # Frame slider
        self._slider_var = tk.IntVar()
        self._slider = ttk.Scale(ctrl, from_=0, to=1,
                                 variable=self._slider_var,
                                 orient=tk.HORIZONTAL,
                                 command=self._on_slider)
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    # ── Axis styling ──────────────────────────────────────────────────────────
    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor("#11111b")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#313244")
        ax.tick_params(colors="#6c7086", labelsize=6)
        ax.xaxis.label.set_color("#6c7086")
        ax.yaxis.label.set_color("#6c7086")
        ax.zaxis.label.set_color("#6c7086")
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_ids(self):
        self._all_ids = load_motion_ids()
        self._filtered_ids = list(self._all_ids)
        self._refresh_listbox()

    def _refresh_listbox(self):
        self._listbox.delete(0, tk.END)
        for mid in self._filtered_ids:
            self._listbox.insert(tk.END, f"  {mid}")
        self._count_label.config(
            text=f"{len(self._filtered_ids)} / {len(self._all_ids)} clips")

    # ── Events ────────────────────────────────────────────────────────────────
    def _on_search(self, *_):
        query = self._search_var.get().strip()
        self._filtered_ids = (
            [mid for mid in self._all_ids if query in mid]
            if query else list(self._all_ids)
        )
        self._refresh_listbox()

    def _on_select(self, _event=None):
        sel = self._listbox.curselection()
        if not sel:
            return
        motion_id = self._filtered_ids[sel[0]]
        self._load_clip(motion_id)

    def _on_slider(self, value):
        if self._joints is None:
            return
        frame = int(float(value))
        self._frame_idx = frame
        self._draw_frame(frame)

    # ── Clip loading & animation ──────────────────────────────────────────────
    def _load_clip(self, motion_id: str):
        self._stop_anim()
        self._joints = load_joints(motion_id)
        T = self._joints.shape[0]

        descriptions = load_descriptions(motion_id)
        desc_text = descriptions[0] if descriptions else "(no description)"
        self._desc_label.config(
            text=f"[{motion_id}]  {T} frames @ {FPS} fps  —  {desc_text}")

        self._frame_idx = 0
        self._slider.config(from_=0, to=T - 1)
        self._slider_var.set(0)
        self._ax.view_init(elev=20, azim=-90)  # front view: X left-right, Y up
        self._draw_frame(0)
        self._start_anim()

    def _draw_frame(self, frame: int):
        joints = self._joints  # (T, 22, 3)
        pos = joints[frame]    # (22, 3)  x, y, z

        # Preserve user rotation across frames (cla() resets the view)
        elev, azim = self._ax.elev, self._ax.azim
        self._ax.cla()
        self._style_axes()
        self._ax.view_init(elev=elev, azim=azim)

        # Compute axis limits from the full clip for stability
        # Plot as (x, z, y) so matplotlib's Z-up maps to data's Y-up (height).
        # Floor is XZ, Y is the wall/height axis.
        lo = joints.min(axis=(0, 1))
        hi = joints.max(axis=(0, 1))
        pad = 0.2
        self._ax.set_xlim(lo[0] - pad, hi[0] + pad)  # data X → plot X
        self._ax.set_ylim(lo[2] - pad, hi[2] + pad)  # data Z → plot Y (depth)
        self._ax.set_zlim(lo[1] - pad, hi[1] + pad)  # data Y → plot Z (up)

        # Bones
        for i, j in BONES:
            self._ax.plot(
                [pos[i, 0], pos[j, 0]],  # X
                [pos[i, 2], pos[j, 2]],  # Z → plot Y
                [pos[i, 1], pos[j, 1]],  # Y → plot Z (up)
                color="#89b4fa", linewidth=1.8, alpha=0.85,
            )

        # Joints
        self._ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1],
                         c="#f38ba8", s=18, zorder=5, depthshade=False)

        T = joints.shape[0]
        self._ax.set_title(f"Frame {frame + 1} / {T}",
                           color="#cdd6f4", fontsize=9, pad=4)
        self._canvas.draw_idle()

        self._frame_label.config(text=f"Frame {frame + 1} / {T}")
        self._slider_var.set(frame)

    def _animate(self, _frame):
        if not self._playing or self._joints is None:
            return
        T = self._joints.shape[0]
        self._frame_idx = (self._frame_idx + 1) % T
        self._draw_frame(self._frame_idx)

    def _start_anim(self):
        if self._joints is None:
            return
        self._playing = True
        self._play_btn.config(text="⏸  Pause")
        interval = int(1000 / FPS)
        self._anim = FuncAnimation(
            self._fig, self._animate,
            interval=interval, cache_frame_data=False
        )
        self._canvas.draw()

    def _stop_anim(self):
        self._playing = False
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None
        self._play_btn.config(text="▶  Play")

    def _toggle_play(self):
        if self._joints is None:
            return
        if self._playing:
            self._stop_anim()
        else:
            self._start_anim()

    def _restart(self):
        if self._joints is None:
            return
        self._stop_anim()
        self._frame_idx = 0
        self._draw_frame(0)
        self._start_anim()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ViewerApp()
    app.mainloop()
