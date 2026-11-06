"""Xem agent chơi lần lượt cả 3 map trong cửa sổ pygame.
python demo_all.py
python demo_all.py --algo sarsa --seed 3
Thuật toán: qlearning, sarsa, mc, vi (Value Iteration), pi (Policy Iteration)
"""
import argparse
import time
from pathlib import Path

import numpy as np

from demo import ALL_ALGOS, load_or_train
from windy.env import WindyGridWorldEnv, load_config

MAPS = [
    ("Map 1 - kinh dien, gio co dinh (7x10)", "configs/map1_classic.json"),
    ("Map 2 - co tuong, gio ngau nhien (10x14)", "configs/map2_stochastic.json"),
    ("Map 3 - co neo, gio ngau nhien (12x16)", "configs/map3_anchor.json"),
]


def play(title, path, algo, seed):
    cfg = load_config(path)
    Q = load_or_train(cfg, algo, Path(f"results/models/{cfg['name']}_{algo}.npy"), seed)
    env = WindyGridWorldEnv(cfg, render_mode="human")
    env.metadata = dict(env.metadata, render_fps=3)
    print(f"\n===== {title} | {algo} =====")
    s, _ = env.reset(seed=seed)
    G, done, term, info = 0.0, False, False, {"t": 0}
    while not done:
        s, r, term, trunc, info = env.step(int(np.argmax(Q[s])))
        G += r
        done = term or trunc
    print(f"{'TOI DICH' if term else 'HET GIO'} sau {info['t']} buoc, return = {G:.0f}")
    time.sleep(2)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=ALL_ALGOS, default="qlearning")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for title, path in MAPS:
        play(title, path, args.algo, args.seed)


if __name__ == "__main__":
    main()