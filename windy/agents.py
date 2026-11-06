"""
Các thuật toán học tăng cường dạng bảng (không dùng function approximation).

    Monte Carlo control (first-visit, ε-greedy) - chờ hết episode mới học
    SARSA                (TD on-policy)        - học sau mỗi bước, dùng a' thật sự làm
    Q-learning           (TD off-policy)       - học sau mỗi bước, dùng max

(Policy Iteration và Value Iteration nằm trong planning.py vì cần biết mô hình.)

Mọi agent có chung giao diện:
    a  = agent.start(s)                                   # đầu episode
    a2 = agent.observe(s, a, r, s2, terminated, truncated) # sau mỗi bước, trả về action tiếp theo
    agent.end_episode()                                   # cuối episode
    agent.Q                                               # bảng Q dùng để chạy policy greedy
"""
from __future__ import annotations

import numpy as np


class TabularAgent:
    name = "base"

    def __init__(self, n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1,
                 epsilon_decay=1.0, epsilon_min=0.0, q_init=0.0, seed=None, **_):
        self.nS, self.nA = n_states, n_actions
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.Q = np.full((n_states, n_actions), float(q_init))
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- chọn action
    def greedy(self, s: int) -> int:
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())  # phá hoà ngẫu nhiên
        return int(self.rng.choice(best))

    def act(self, s: int) -> int:
        """ε-greedy: xác suất ε chọn ngẫu nhiên, còn lại chọn action tốt nhất."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.nA))
        return self.greedy(s)

    # ---------------------------------------------------------------- vòng đời episode
    def start(self, s: int) -> int:
        return self.act(s)

    def observe(self, s, a, r, s2, terminated, truncated) -> int:
        raise NotImplementedError

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ===================================================================== Monte Carlo
class MonteCarloAgent(TabularAgent):
    """On-policy first-visit Monte Carlo control với ε-greedy (Sutton & Barto, mục 5.4).
    Chờ hết episode rồi mới cập nhật: Q(s,a) ← trung bình các return G nhận được.
    mc_alpha = None  -> trung bình mẫu (1/N);  mc_alpha = số  -> constant-α MC."""
    name = "Monte Carlo"

    def __init__(self, *args, mc_alpha=None, **kw):
        super().__init__(*args, **kw)
        self.mc_alpha = mc_alpha
        self.N = np.zeros_like(self.Q)

    def start(self, s):
        self.episode = []
        return self.act(s)

    def observe(self, s, a, r, s2, terminated, truncated):
        self.episode.append((s, a, r))
        a2 = self.act(s2)
        if terminated or truncated:
            self._learn_from_episode()
        return a2

    def _learn_from_episode(self):
        first = {}
        for t, (s, a, _) in enumerate(self.episode):   # lần xuất hiện ĐẦU TIÊN của (s, a)
            first.setdefault((s, a), t)
        G = 0.0
        for t in range(len(self.episode) - 1, -1, -1):  # duyệt ngược để tính return
            s, a, r = self.episode[t]
            G = self.gamma * G + r
            if first[(s, a)] == t:
                self.N[s, a] += 1
                step = self.mc_alpha if self.mc_alpha is not None else 1.0 / self.N[s, a]
                self.Q[s, a] += step * (G - self.Q[s, a])


# ===================================================================== TD
class SarsaAgent(TabularAgent):
    """On-policy TD: Q(s,a) += α [r + γ Q(s',a') - Q(s,a)], a' chọn bằng ε-greedy."""
    name = "SARSA"

    def observe(self, s, a, r, s2, terminated, truncated):
        a2 = self.act(s2)  # chọn a' TRƯỚC vì công thức cần a'
        target = r if terminated else r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        return a2


class QLearningAgent(TabularAgent):
    """Off-policy TD: Q(s,a) += α [r + γ max_a' Q(s',a') - Q(s,a)]."""
    name = "Q-learning"

    def observe(self, s, a, r, s2, terminated, truncated):
        target = r if terminated else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        return self.act(s2)


AGENTS = {
    "mc": MonteCarloAgent,
    "sarsa": SarsaAgent,
    "qlearning": QLearningAgent,
}
ALGO_LABELS = {k: v.name for k, v in AGENTS.items()}


def algo_settings(cfg: dict, name: str) -> dict:
    """Siêu tham số chung trong config["agent"], ghi đè bởi config["agent"]["per_algo"][name]."""
    a_cfg = cfg.get("agent", {})
    merged = {k: v for k, v in a_cfg.items() if k != "per_algo"}
    merged.update(a_cfg.get("per_algo", {}).get(name.lower(), {}))
    return merged


def make_agent(name: str, env, seed=None, **overrides) -> TabularAgent:
    hp = {"alpha": 0.5, "epsilon": 0.1, "epsilon_decay": 1.0, "epsilon_min": 0.0, "q_init": 0.0,
          "mc_alpha": None}
    hp.update({k: v for k, v in algo_settings(env.cfg, name).items() if k in hp})
    hp.update(overrides)
    cls = AGENTS[name.lower()]
    if cls is not MonteCarloAgent:
        hp.pop("mc_alpha")
    return cls(env.n_states, env.n_actions, gamma=env.gamma, seed=seed, **hp)


def train(env, agent: TabularAgent, n_episodes: int, seed: int = 0):
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    success = np.zeros(n_episodes, dtype=bool)
    s, _ = env.reset(seed=seed)
    for ep in range(n_episodes):
        if ep > 0:
            s, _ = env.reset()
        a = agent.start(s)
        G, t, terminated = 0.0, 0, False
        while True:
            s2, r, terminated, truncated, _ = env.step(a)
            a2 = agent.observe(s, a, r, s2, terminated, truncated)
            G += r
            t += 1
            s, a = s2, a2
            if terminated or truncated:  # truncated: TD vẫn bootstrap (không phải terminal thật)
                break
        agent.end_episode()
        returns[ep], lengths[ep], success[ep] = G, t, terminated
    return {"returns": returns, "lengths": lengths, "success": success}


def run_greedy_episode(env, Q, seed=None):
    s, info = env.reset(seed=seed)
    traj = [(info["row"], info["col"], info["anchor"])]
    G, bumps, used, terminated, stuck = 0.0, 0, False, False, False
    while True:
        a = int(np.argmax(Q[s]))
        s2, r, terminated, truncated, info = env.step(a)
        G += r
        bumps += int(info["bumped"])
        used |= info["used_anchor"]
        traj.append((info["row"], info["col"], info["anchor"]))
        if terminated or truncated:
            break
        # Kẹt vĩnh viễn: đứng yên tại chỗ và chuyển trạng thái tất định -> Q đóng băng nên
        # sẽ lặp mãi. Dừng sớm, cộng reward như thể chạy hết max_steps (kết quả không đổi).
        if s2 == s and len(env.transition_distribution(s, a)) == 1:
            remaining = env.max_steps - info["t"]
            G += r * remaining
            bumps += int(info["bumped"]) * remaining
            stuck = True
            break
        s = s2
    top = np.mean([p[0] == 0 for p in traj])
    length = env.max_steps if stuck else len(traj) - 1
    return {"return": G, "length": length, "success": terminated, "bumps": bumps,
            "used_anchor": used, "top_row_frac": float(top), "trajectory": traj, "stuck": stuck}


def evaluate(env, Q, n_episodes=50, seed=0):
    eps = [run_greedy_episode(env, Q, seed=seed + i) for i in range(n_episodes)]
    rets = np.array([e["return"] for e in eps])
    return {
        "mean_return": float(rets.mean()),
        "std_return": float(rets.std()),
        "mean_length": float(np.mean([e["length"] for e in eps])),
        "success_rate": float(np.mean([e["success"] for e in eps])),
        "mean_bumps": float(np.mean([e["bumps"] for e in eps])),
        "top_row_frac": float(np.mean([e["top_row_frac"] for e in eps])),
        "anchor_use_rate": float(np.mean([e["used_anchor"] for e in eps])),
    }