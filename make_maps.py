"""Vẽ 3 map từ file config rồi lưu thành maps.png (dùng cho báo cáo)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from windy import WindyGridWorldEnv, load_config
from windy.viz import _draw_grid

maps = [
    ("configs/map1_classic.json", "Map 1 (easy)"),
    ("configs/map2_stochastic.json", "Map 2 (medium)"),
    ("configs/map3_anchor.json", "Map 3 (hard)"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                         gridspec_kw={"width_ratios": [10, 14, 16]})
for ax, (path, title) in zip(axes, maps):
    env = WindyGridWorldEnv(load_config(path))
    _draw_grid(ax, env)
    ax.set_xlabel("wind of each column", fontsize=8)
    ax.set_title(f"{title}: {env.n_rows}x{env.n_cols}", fontsize=11)

fig.tight_layout()
fig.savefig("maps.png", dpi=150)
print("Đã lưu maps.png")