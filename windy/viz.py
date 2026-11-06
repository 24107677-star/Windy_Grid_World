from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .env import HOLDING, NO_ANCHOR

ARROWS = ["↑", "↓", "←", "→"]
STATUS_TITLES = {0: "chưa có neo", 1: "đang giữ neo", 2: "đã dùng neo"}


def _smooth(x, w):
    w = max(1, min(w, len(x)))
    if w == 1:
        return np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_learning_curves(curves, title, path, window=20, ylabel="Return mỗi episode", clip="low"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo, hi = [], []
    for label, arr in curves.items():
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        m, sd = _smooth(mean, window), _smooth(std, window)
        x = np.arange(len(mean) - len(m) + 1, len(mean) + 1)
        ax.plot(x, m, lw=1.6, label=f"{label} ({arr.shape[0]} seed)")
        ax.fill_between(x, m - sd, m + sd, alpha=0.2)
        lo.append(np.percentile(m - sd, 5))
        hi.append(np.percentile(m + sd, 95))
    if clip == "low" and lo:
        ax.set_ylim(bottom=min(lo))
    elif clip == "high" and hi:
        ax.set_ylim(top=max(hi))
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(trung bình trượt {window} episode, vùng mờ = ±1 std giữa các seed)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_steps_vs_episodes(lengths, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, arr in lengths.items():
        mean = np.atleast_2d(np.asarray(arr, dtype=float)).mean(axis=0)
        ax.plot(np.cumsum(mean), np.arange(1, len(mean) + 1), label=label)
    ax.set_xlabel("Tổng số time step")
    ax.set_ylabel("Số episode")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_grid(ax, env, show_anchors=True):
    R, C = env.n_rows, env.n_cols
    img = np.ones((R, C, 3))
    wmax = max(1, int(env.wind.max()))
    for c in range(C):
        k = env.wind[c] / wmax
        img[:, c] = [1 - 0.25 * k, 1 - 0.12 * k, 1.0]
    img[env.walls] = [0.25, 0.25, 0.25]
    for (r, c) in env.goals:
        img[r, c] = [0.55, 0.85, 0.55]
    sr, sc = env.start
    img[sr, sc] = [0.7, 0.8, 1.0]
    ax.imshow(img, origin="upper")
    ax.set_xticks(np.arange(C))
    ax.set_xticklabels([str(w) for w in env.wind], fontsize=8)
    ax.set_yticks(np.arange(R))
    ax.set_yticklabels([str(r) for r in range(R)], fontsize=8)
    ax.set_xlabel("cường độ gió danh định theo cột", fontsize=8)
    ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, R, 1), minor=True)
    ax.grid(which="minor", color="lightgray", lw=0.6)
    ax.tick_params(which="minor", length=0)
    ax.text(sc - 0.42, sr - 0.28, "S", fontsize=7, color="navy")
    for (r, c) in env.goals:
        ax.text(c, r, "G", ha="center", va="center", fontsize=11, fontweight="bold", color="darkgreen")
    if show_anchors and env.anchor_mode != "none":
        for (r, c) in env.anchor_cells:
            ax.add_patch(plt.Circle((c, r), 0.38, color="gold", alpha=0.6))


def plot_policy(env, Q, title, path, trajectory=None, statuses=None):
    statuses = list(range(env.n_anchor_states)) if statuses is None else statuses
    n, cw = len(statuses), 0.42
    fig, axes = plt.subplots(1, n, figsize=(max(4.0, cw * env.n_cols) * n + 0.6, cw * env.n_rows + 1.8),
                             squeeze=False)
    for ax, k in zip(axes[0], statuses):
        _draw_grid(ax, env, show_anchors=(k == NO_ANCHOR))
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                if env.walls[r, c] or (r, c) in env.goals:
                    continue
                q = Q[env.encode(r, c, k)]
                if np.all(q == q[0]):
                    ax.text(c, r, "·", ha="center", va="center", color="gray")
                    continue
                a = int(np.argmax(q))
                use = a >= 4 and k == HOLDING
                ax.text(c, r, ARROWS[a % 4], ha="center", va="center", fontsize=11,
                        color="crimson" if use else "black", fontweight="bold" if use else "normal")
        if trajectory:
            seg = [(rr, cc) for rr, cc, kk in trajectory if kk == k]
            if seg:
                ys, xs = zip(*seg)
                ax.plot(xs, ys, "-o", color="tab:red", alpha=0.45, lw=2, ms=3)
        if env.n_anchor_states > 1:
            ax.set_title(f"trạng thái neo: {STATUS_TITLES[k]}", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)