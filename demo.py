"""Demo 1 episode bằng policy đã học.
python demo.py --config configs/map3_anchor.json --algo qlearning              # cửa sổ pygame
python demo.py --config configs/map1_classic.json --algo sarsa --render ansi  # in ra terminal
python demo.py --config configs/map3_anchor.json --render gif                  # xuất results/demo.gif
python demo.py --config configs/map3_anchor.json --algo vi                     # policy của Value Iteration
Thuật toán: qlearning, sarsa, mc, vi (Value Iteration), pi (Policy Iteration)
"""
import argparse
import time
from pathlib import Path

import numpy as np

from windy.agents import AGENTS, algo_settings, make_agent, train
from windy.env import ANCHOR_STATUS_NAMES, MOVE_NAMES, WindyGridWorldEnv, load_config
from windy.planning import policy_iteration, value_iteration

DP_ALGOS = {"vi": value_iteration, "pi": policy_iteration}
ALL_ALGOS = list(AGENTS) + list(DP_ALGOS)   # mc, sarsa, qlearning, vi, pi


def load_or_train(cfg, algo, model_path: Path, seed=0):
    if model_path.exists():
        return np.load(model_path)
    env = WindyGridWorldEnv(cfg)
    if algo in DP_ALGOS:                     # Policy/Value Iteration: tính thẳng từ mô hình
        print(f"Chưa có {model_path}, đang tính bằng {algo.upper()}...")
        Q = DP_ALGOS[algo](env)[1]
    else:
        print(f"Chưa có {model_path}, đang train nhanh...")
        agent = make_agent(algo, env, seed=seed)
        train(env, agent, algo_settings(cfg, algo).get("episodes", 500), seed=seed)
        Q = agent.Q
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(model_path, Q)
    return Q


def save_gif(frames, path, fps=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(frames[0])
    anim = FuncAnimation(fig, lambda i: (im.set_data(frames[i]),), frames=len(frames))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/map3_anchor.json")
    p.add_argument("--algo", choices=ALL_ALGOS, default="qlearning")
    p.add_argument("--model", default=None)
    p.add_argument("--render", choices=["human", "ansi", "gif"], default="human")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--gif-path", default="results/demo.gif")
    args = p.parse_args()

    cfg = load_config(args.config)
    model = Path(args.model or f"results/models/{cfg['name']}_{args.algo}.npy")
    Q = load_or_train(cfg, args.algo, model, args.seed)

    mode = {"human": "human", "ansi": "ansi", "gif": "rgb_array"}[args.render]
    env = WindyGridWorldEnv(cfg, render_mode=mode)
    env.metadata = dict(env.metadata, render_fps=3)  # chậm hơn để dễ xem
    if Q.shape != (env.n_states, env.n_actions):
        raise SystemExit(f"Model {Q.shape} không khớp môi trường ({env.n_states}, {env.n_actions})")

    s, info = env.reset(seed=args.seed)
    frames = [env.render()] if args.render == "gif" else []
    if args.render == "ansi":
        print(env.render())
    G, done, term = 0.0, False, False
    while not done:
        a = int(np.argmax(Q[s]))
        s, r, term, trunc, info = env.step(a)
        G += r
        done = term or trunc
        act = MOVE_NAMES[a % 4] + (" +NEO" if a >= 4 else "")
        tags = (" [DÙNG NEO]" if info["used_anchor"] else "") + (" [NHẶT NEO]" if info["picked_anchor"] else "")
        print(f"t={info['t']:3d}  action={act:10s} gió={info['wind']}  -> ({info['row']},{info['col']})"
              f"  neo={ANCHOR_STATUS_NAMES[info['anchor']]}  r={r:.0f}{tags}")
        if args.render == "ansi":
            print(env.render())
            time.sleep(args.delay)
        elif args.render == "gif":
            frames.append(env.render())

    print(f"\nKết thúc: {'TỚI ĐÍCH' if term else 'HẾT max_steps'} sau {info['t']} bước, return = {G:.1f}")
    if args.render == "gif":
        Path(args.gif_path).parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, args.gif_path)
        print(f"Đã lưu {args.gif_path}")
    if args.render == "human":
        time.sleep(2)
    env.close()


if __name__ == "__main__":
    main()