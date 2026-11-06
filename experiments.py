"""
Toàn bộ thí nghiệm của đề tài.

  python experiments.py --exp all --seeds 10                 # đầy đủ (lâu)
  python experiments.py --exp main --seeds 3                 # chạy thử nhanh
  python experiments.py --exp main --algos sarsa qlearning   # chỉ chạy vài thuật toán
  python experiments.py --exp dp | vi | reward | failure | anchor

5 thuật toán: Policy Iteration, Value Iteration (quy hoạch động, cần mô hình)
              Monte Carlo, SARSA, Q-learning (học từ kinh nghiệm, không cần mô hình)
Thí nghiệm:
  E1 main    : cả 5 thuật toán trên 3 map
  E2 dp      : Policy Iteration vs Value Iteration (số vòng lặp, thời gian)
  E3 vi      : so policy học được (MC, SARSA, Q-learning) với policy tối ưu trên Map 1
  E4 reward  : đổi reward / tham số môi trường
  E5 failure : trường hợp thất bại / hội tụ chậm
  E6 anchor  : 3 cơ chế dùng neo

Kết quả: results/figures/*.png, results/table_*.md, results/models/*.npy, results/raw/*.npz
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from windy import viz
from windy.agents import ALGO_LABELS, AGENTS, algo_settings, evaluate, make_agent, run_greedy_episode, train
from windy.env import WindyGridWorldEnv, load_config
from windy.planning import policy_agreement, policy_iteration, reachable_states, value_iteration

RESULTS = Path("results")
FIG, MODELS, RAW = RESULTS / "figures", RESULTS / "models", RESULTS / "raw"
for d in (FIG, MODELS, RAW):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIGS = ["configs/map1_classic.json", "configs/map2_stochastic.json", "configs/map3_anchor.json"]
ALL_ALGOS = list(AGENTS)          # mc, sarsa, qlearning
LABEL = ALGO_LABELS


# ---------------------------------------------------------------- tiện ích
def run_many(cfg, algo, seeds, episodes=None, overrides=None, eval_episodes=None):
    st = algo_settings(cfg, algo)
    episodes = episodes or st.get("episodes", 500)
    eval_episodes = eval_episodes or st.get("eval_episodes", 50)
    seeds = list(seeds)[: st.get("n_seeds", len(seeds))]
    out = {"returns": [], "lengths": [], "success": [], "Qs": [], "evals": [], "episodes": episodes}
    for seed in seeds:
        env = WindyGridWorldEnv(cfg)
        agent = make_agent(algo, env, seed=seed, **(overrides or {}))
        log = train(env, agent, episodes, seed=seed)
        for k in ("returns", "lengths", "success"):
            out[k].append(log[k])
        out["Qs"].append(agent.Q.copy())
        out["evals"].append(evaluate(env, agent.Q, eval_episodes, seed=100_000 + seed))
    for k in ("returns", "lengths", "success"):
        out[k] = np.array(out[k])
    out["best"] = int(np.argmax([e["mean_return"] for e in out["evals"]]))
    return out


def convergence_episode(curve, window=20, tol_frac=0.05):
    """Episode mà từ đó trung bình trượt luôn nằm trong ngưỡng của mức cuối cùng."""
    if len(curve) < window:
        return None
    ma = np.convolve(curve, np.ones(window) / window, mode="valid")
    final = ma[-max(1, len(ma) // 10):].mean()
    tol = tol_frac * abs(final) + 1.0
    bad = np.flatnonzero(ma < final - tol)
    return int(bad[-1] + 1 + window) if len(bad) else window


def summarize(res):
    ev, R = res["evals"], res["returns"]
    last = max(1, min(50, R.shape[1] // 10))
    good = [e["mean_length"] for e in ev if e["success_rate"] == 1.0]
    return {
        "episode train": res["episodes"],
        "return train (cuối)": float(R[:, -last:].mean()),
        "return greedy": float(np.mean([e["mean_return"] for e in ev])),
        "std giữa seed": float(np.std([e["mean_return"] for e in ev])),
        "số bước greedy": float(np.mean([e["mean_length"] for e in ev])),
        "tỉ lệ tới G": float(np.mean([e["success_rate"] for e in ev])),
        "seed luôn tới G": f"{len(good)}/{len(ev)}",
        "số bước (seed tới G)": float(np.mean(good)) if good else None,
        "episode hội tụ": convergence_episode(R.mean(axis=0)),
        "va biên/tường": float(np.mean([e["mean_bumps"] for e in ev])),
        "% bước ở hàng 0": 100 * float(np.mean([e["top_row_frac"] for e in ev])),
        "tỉ lệ dùng neo": float(np.mean([e["anchor_use_rate"] for e in ev])),
    }


def _fmt(v):
    if v is None:
        return "–"
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}"
    return str(v)


def write_table(tag, title, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    text = "\n".join(lines) + "\n"
    (RESULTS / f"table_{tag}.md").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)


def save_policy(env, Q, title, fname):
    ep = run_greedy_episode(env, Q, seed=0)
    viz.plot_policy(env, Q, f"{title} (return greedy = {ep['return']:.0f})", FIG / fname,
                    trajectory=ep["trajectory"])


# ---------------------------------------------------------------- E1: cả 5 thuật toán trên 3 map
def exp_main(config_paths, seeds, episodes, algos):
    rows = []
    for path in config_paths:
        cfg = load_config(path)
        name = cfg["name"]
        env = WindyGridWorldEnv(cfg)
        n_eval = cfg.get("agent", {}).get("eval_episodes", 50)
        print(f"\n=== {name}: |S| = {env.n_states} (state hợp lệ = {len(env.valid_states())}), "
              f"|A| = {env.n_actions}", flush=True)
        # --- Policy Iteration và Value Iteration: tính thẳng từ mô hình
        t0 = time.time()
        V, Q_vi, _, sweeps = value_iteration(env)
        t_vi = time.time() - t0
        t0 = time.time()
        _, Q_pi, _, n_improve, _ = policy_iteration(env)
        t_pi = time.time() - t0
        v_star = float(V[env.encode(*env.start, 0)])
        for lab, Q, dt, note in [("Policy Iteration", Q_pi, t_pi, f"{n_improve} lần cải thiện policy"),
                                 ("Value Iteration", Q_vi, t_vi, f"{sweeps} vòng quét")]:
            ev = evaluate(env, Q, n_eval, seed=100_000)
            rows.append({"config": name, "số state": env.n_states, "số action": env.n_actions, "thuật toán": lab,
                         "tối ưu (VI)": -v_star, "episode train": "– (dùng mô hình)",
                         "return greedy": ev["mean_return"], "số bước greedy": ev["mean_length"],
                         "tỉ lệ tới G": ev["success_rate"], "seed luôn tới G": "–",
                         "số bước (seed tới G)": ev["mean_length"], "tỉ lệ dùng neo": ev["anchor_use_rate"],
                         "ghi chú": note, "thời gian (s)": dt})
            key = "pi" if lab.startswith("Policy") else "vi"
            np.save(MODELS / f"{name}_{key}.npy", Q)
            print(f"  {lab}: xong sau {dt:.1f}s", flush=True)
        save_policy(env, Q_vi, f"{name} - Value Iteration", f"policy_{name}_vi.png")
        # --- Monte Carlo, SARSA, Q-learning: học từ kinh nghiệm
        curves, lens = {}, {}
        for algo in algos:
            t0 = time.time()
            res = run_many(cfg, algo, seeds, episodes)
            lab = LABEL[algo]
            curves[lab], lens[lab] = res["returns"], res["lengths"]
            Q = res["Qs"][res["best"]]
            np.save(MODELS / f"{name}_{algo}.npy", Q)
            np.savez(RAW / f"main_{name}_{algo}.npz", returns=res["returns"], lengths=res["lengths"])
            save_policy(env, Q, f"{name} - {lab}", f"policy_{name}_{algo}.png")
            rows.append({"config": name, "số state": env.n_states, "số action": env.n_actions, "thuật toán": lab,
                         "tối ưu (VI)": -v_star, **summarize(res), "thời gian (s)": time.time() - t0})
            print(f"  {lab}: xong sau {time.time() - t0:.1f}s", flush=True)
        viz.plot_learning_curves(curves, f"Đường học - {name}", FIG / f"curve_{name}.png")
        viz.plot_steps_vs_episodes(lens, f"Episode theo time step - {name}", FIG / f"steps_{name}.png")
    write_table("main", "E1. So sánh 5 thuật toán trên các setting", rows)


# ---------------------------------------------------------------- E2: Policy Iteration vs Value Iteration
def exp_dp(config_paths):
    rows = []
    for path in config_paths:
        cfg = load_config(path)
        env = WindyGridWorldEnv(cfg)
        s0 = env.encode(*env.start, 0)
        t0 = time.time()
        V, Q, pi_v, sweeps = value_iteration(env)
        tv = time.time() - t0
        t0 = time.time()
        Vp, Qp, pi_p, n_improve, eval_sweeps = policy_iteration(env)
        tp = time.time() - t0
        same = float(np.mean([Q[s, pi_p[s]] >= Q[s].max() - 1e-6 for s in range(env.n_states)
                              if env.is_valid_state(s) and not env.is_terminal_state(s)]))
        rows.append({"config": cfg["name"], "phương pháp": "Value Iteration", "V*(start)": float(V[s0]),
                     "số vòng lặp": f"{sweeps} vòng quét", "thời gian (s)": tv, "policy tối ưu": "có"})
        rows.append({"config": cfg["name"], "phương pháp": "Policy Iteration", "V*(start)": float(Vp[s0]),
                     "số vòng lặp": f"{n_improve} lần cải thiện, {eval_sweeps} vòng quét đánh giá",
                     "thời gian (s)": tp, "policy tối ưu": f"khớp VI {100 * same:.0f}%"})
        if cfg["name"] in ("map1_classic", "map3_anchor"):
            save_policy(env, Q, f"Value Iteration - {cfg['name']}", f"policy_vi_{cfg['name']}.png")
            save_policy(env, Qp, f"Policy Iteration - {cfg['name']}", f"policy_pi_{cfg['name']}.png")
        print(f"  {cfg['name']}: VI {tv:.2f}s, PI {tp:.2f}s", flush=True)
    write_table("dp", "E2. Quy hoạch động: Policy Iteration vs Value Iteration", rows)


# ---------------------------------------------------------------- E3: đối chiếu với policy tối ưu
def exp_vi(seeds, episodes, algos):
    base = load_config("configs/map1_classic.json")
    stoch = copy.deepcopy(base)
    stoch.update({"name": "map1_stochastic", "stochastic_wind": True})
    stoch["agent"].update({"alpha": 0.1, "episodes": 3000, "eval_episodes": 100})
    rows = []
    for label, cfg in {"Map 1 gió tất định": base, "Map 1 gió ngẫu nhiên": stoch}.items():
        env = WindyGridWorldEnv(cfg)
        n_eval = cfg["agent"]["eval_episodes"]
        V, Qstar, pi, iters = value_iteration(env)
        s0 = env.encode(*env.start, 0)
        ev = evaluate(env, Qstar, n_eval, seed=7)
        rows.append({"setting": label, "phương pháp": "Value Iteration", "V*(start)": float(V[s0]),
                     "return greedy": ev["mean_return"], "seed luôn tới G": "–",
                     "khớp policy tối ưu": 1.0, "ghi chú": f"{iters} vòng quét"})
        save_policy(env, Qstar, f"Value Iteration - {label}", f"policy_vi_{cfg['name']}.png")
        reach = reachable_states(env, pi, s0)
        for algo in algos:
            res = run_many(cfg, algo, seeds, episodes, eval_episodes=n_eval)
            agree = np.mean([policy_agreement(Q, Qstar, reach) for Q in res["Qs"]])
            s = summarize(res)
            rows.append({"setting": label, "phương pháp": LABEL[algo], "V*(start)": float(V[s0]),
                         "return greedy": s["return greedy"], "seed luôn tới G": s["seed luôn tới G"],
                         "số bước (seed tới G)": s["số bước (seed tới G)"],
                         "khớp policy tối ưu": float(agree), "ghi chú": f"hội tụ ~ episode {s['episode hội tụ']}"})
            print(f"  {label} - {LABEL[algo]}: xong", flush=True)
    write_table("vi", "E3. Đối chiếu policy học được với policy tối ưu (Map 1)", rows)


# ---------------------------------------------------------------- E4: reward / tham số môi trường
def exp_reward(seeds, episodes):
    base = load_config("configs/map2_stochastic.json")
    variants = {
        "R1: -1/bước (gốc)": {},
        "R2: -1/bước, -5 khi va biên/tường": {"reward": {"step": -1, "goal": 0, "wall_bump": -5}},
        "R3: thưởng thưa +10 tại G, γ=0.95": {"reward": {"step": 0, "goal": 10, "wall_bump": 0}, "gamma": 0.95},
        "E1: gió tất định (bỏ nhiễu ±1)": {"stochastic_wind": False},
    }
    rows, lens = [], {}
    for i, (label, patch) in enumerate(variants.items()):
        cfg = copy.deepcopy(base)
        cfg.update(copy.deepcopy(patch))
        cfg["name"] = f"reward_v{i + 1}"
        res = run_many(cfg, "qlearning", seeds, episodes)
        lens[label] = res["lengths"]
        env = WindyGridWorldEnv(cfg)
        save_policy(env, res["Qs"][res["best"]], f"Q-learning - {label}", f"policy_reward_v{i + 1}.png")
        s = summarize(res)
        rows.append({"biến thể": label, "số bước greedy": s["số bước greedy"], "tỉ lệ tới G": s["tỉ lệ tới G"],
                     "va biên/tường": s["va biên/tường"], "% bước ở hàng 0": s["% bước ở hàng 0"],
                     "episode hội tụ": s["episode hội tụ"]})
    viz.plot_learning_curves(lens, "Số bước mỗi episode theo cấu hình reward/môi trường (Map 2, Q-learning)",
                             FIG / "curve_reward_variants.png", ylabel="Số bước / episode", clip="high")
    write_table("reward", "E4. Ảnh hưởng của reward và tham số môi trường (Map 2)", rows)


# ---------------------------------------------------------------- E5: thất bại / hội tụ chậm
def exp_failure(seeds, episodes):
    base = load_config("configs/map2_stochastic.json")
    seeds = seeds[:5]
    episodes = episodes or 800
    variants = {
        "Q-learning chuẩn (α=0.2, ε=0.1)": ("qlearning", {}),
        "α=0.02 (học quá chậm)": ("qlearning", {"alpha": 0.02}),
        "ε=0.01 (khám phá quá ít)": ("qlearning", {"epsilon": 0.01}),
        "Q0=-50 (khởi tạo bi quan)": ("qlearning", {"q_init": -50.0}),
        "Q-learning α=0.9 (dao động)": ("qlearning", {"alpha": 0.9}),
        "SARSA α=0.9 (dao động)": ("sarsa", {"alpha": 0.9}),
    }
    rows, curves, lens = [], {}, {}
    for label, (algo, ov) in variants.items():
        t0 = time.time()
        res = run_many(base, algo, seeds, episodes, overrides={"alpha": 0.2, **ov})
        curves[label], lens[label] = res["returns"], res["lengths"]
        s = summarize(res)
        rows.append({"biến thể": label, "return greedy": s["return greedy"], "std giữa seed": s["std giữa seed"],
                     "tỉ lệ tới G": s["tỉ lệ tới G"], "episode hội tụ": s["episode hội tụ"],
                     "tổng time step": int(res["lengths"].sum(axis=1).mean()), "thời gian (s)": time.time() - t0})
        print(f"  {label}: xong", flush=True)
    viz.plot_learning_curves(curves, "Các trường hợp hội tụ chậm / thất bại (Map 2)", FIG / "curve_failure.png")
    viz.plot_steps_vs_episodes(lens, "Episode theo time step - các biến thể (Map 2)", FIG / "steps_failure.png")
    write_table("failure", "E5. Phân tích thất bại / hội tụ chậm", rows)


# ---------------------------------------------------------------- E6: biến thể chính sách dùng neo
def exp_anchor(seeds, episodes):
    base = load_config("configs/map3_anchor.json")
    modes = {"Không dùng neo": "none", "Neo tự động (auto)": "auto", "Agent tự chọn lúc dùng neo (manual)": "manual"}
    rows, curves = [], {}
    for label, mode in modes.items():
        cfg = copy.deepcopy(base)
        cfg.update({"anchor_mode": mode, "name": f"map3_{mode}"})
        env = WindyGridWorldEnv(cfg)
        V, _, _, _ = value_iteration(env)
        res = run_many(cfg, "qlearning", seeds, episodes)
        curves[label] = res["returns"]
        save_policy(env, res["Qs"][res["best"]], f"Map 3 - {label}", f"policy_anchor_{mode}.png")
        s = summarize(res)
        rows.append({"biến thể": label, "số state": env.n_states, "số action": env.n_actions,
                     "tối ưu (VI)": -float(V[env.encode(*env.start, 0)]),
                     "return greedy": s["return greedy"], "seed luôn tới G": s["seed luôn tới G"],
                     "số bước (seed tới G)": s["số bước (seed tới G)"], "tỉ lệ tới G": s["tỉ lệ tới G"],
                     "tỉ lệ dùng neo": s["tỉ lệ dùng neo"], "episode hội tụ": s["episode hội tụ"]})
        print(f"  {label}: xong", flush=True)
    viz.plot_learning_curves(curves, "Map 3 - so sánh cơ chế neo (Q-learning)", FIG / "curve_anchor_modes.png")
    write_table("anchor", "E6. Biến thể chính sách dùng neo (Map 3)", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="all",
                   choices=["all", "main", "dp", "vi", "reward", "failure", "anchor"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=None, help="ghi đè số episode trong config")
    p.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    p.add_argument("--algos", nargs="*", default=ALL_ALGOS, choices=ALL_ALGOS)
    args = p.parse_args()
    seeds = list(range(args.seeds))
    t0 = time.time()
    if args.exp in ("all", "main"):
        exp_main(args.configs, seeds, args.episodes, args.algos)
    if args.exp in ("all", "dp"):
        exp_dp(args.configs)
    if args.exp in ("all", "vi"):
        exp_vi(seeds, args.episodes, args.algos)
    if args.exp in ("all", "reward"):
        exp_reward(seeds, args.episodes)
    if args.exp in ("all", "failure"):
        exp_failure(seeds, args.episodes)
    if args.exp in ("all", "anchor"):
        exp_anchor(seeds, args.episodes)
    print(f"\nHoàn tất sau {(time.time() - t0) / 60:.1f} phút. Xem results/")


if __name__ == "__main__":
    main()