"""Train nhanh 1 config với 1 thuật toán.
python train.py --config configs/map1_classic.json --algo sarsa
Thuật toán: mc, sarsa, qlearning  (Policy/Value Iteration chạy trong experiments.py và app.py)
"""
import argparse
from pathlib import Path

import numpy as np

from windy import viz
from windy.agents import AGENTS, algo_settings, evaluate, make_agent, run_greedy_episode, train
from windy.env import WindyGridWorldEnv, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--algo", choices=list(AGENTS), default="qlearning")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    env = WindyGridWorldEnv(cfg)
    st = algo_settings(cfg, args.algo)
    episodes = args.episodes or st.get("episodes", 500)
    print(f"{cfg['name']}: |S| = {env.n_states} (hợp lệ: {len(env.valid_states())}), |A| = {env.n_actions}")

    agent = make_agent(args.algo, env, seed=args.seed)
    log = train(env, agent, episodes, seed=args.seed)
    ev = evaluate(env, agent.Q, st.get("eval_episodes", 20), seed=999)
    print(f"{agent.name}: train {episodes} episode")
    print(f"Return 20 episode train cuối: {log['returns'][-20:].mean():.2f}")
    print(f"Greedy: return {ev['mean_return']:.2f} ± {ev['std_return']:.2f}, "
          f"{ev['mean_length']:.1f} bước, tới đích {100 * ev['success_rate']:.0f}%")

    models, figs = Path("results/models"), Path("results/figures")
    models.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg['name']}_{args.algo}"
    np.save(models / f"{tag}.npy", agent.Q)
    viz.plot_learning_curves({agent.name: log["returns"][None]}, f"Train {tag}", figs / f"train_{tag}.png")
    ep = run_greedy_episode(env, agent.Q, seed=0)
    viz.plot_policy(env, agent.Q, f"{tag} - policy greedy", figs / f"train_policy_{tag}.png",
                    trajectory=ep["trajectory"])
    print(f"Đã lưu results/models/{tag}.npy và hình trong results/figures/")


if __name__ == "__main__":
    main()