"""
Quy hoạch động (Dynamic Programming) trên mô hình đầy đủ p(s', r | s, a) của môi trường:
    - Policy Iteration : đánh giá policy -> cải thiện policy, lặp tới khi policy không đổi
    - Value Iteration  : lặp phương trình Bellman tối ưu tới khi V hội tụ
Dùng làm chuẩn đối chiếu (đáp án tối ưu) cho các thuật toán học không cần mô hình.
"""
from __future__ import annotations

import numpy as np


def build_model(env):
    """P[s][a] = danh sách (xác suất, s', reward, terminated) lấy từ môi trường."""
    return [[env.transition_distribution(s, a) for a in range(env.n_actions)] for s in range(env.n_states)]


def _backup(P_sa, V, gamma):
    """Σ p(s'|s,a) [r + γ V(s')]  (state đích có giá trị 0)."""
    return sum(p * (r + (0.0 if term else gamma * V[s2])) for p, s2, r, term in P_sa)


def _active_states(env):
    return [s for s in range(env.n_states) if env.is_valid_state(s) and not env.is_terminal_state(s)]


def _q_from_v(env, P, V, gamma, active):
    Q = np.zeros((env.n_states, env.n_actions))
    for s in active:
        for a in range(env.n_actions):
            Q[s, a] = _backup(P[s][a], V, gamma)
    return Q


# ===================================================================== Value Iteration
def value_iteration(env, gamma=None, theta=1e-8, max_iter=10_000):
    gamma = env.gamma if gamma is None else gamma
    P = build_model(env)
    V = np.zeros(env.n_states)
    active = _active_states(env)
    it = 0
    for it in range(1, max_iter + 1):
        delta = 0.0
        for s in active:
            best = max(_backup(P[s][a], V, gamma) for a in range(env.n_actions))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    Q = _q_from_v(env, P, V, gamma, active)
    return V, Q, np.argmax(Q, axis=1), it


# ===================================================================== Policy Iteration
def policy_evaluation(P, policy, V, gamma, active, theta=1e-8, max_sweeps=1000):
    """Tính V^π bằng cách lặp V(s) ← Σ p [r + γ V(s')] với a = π(s).
    Với γ = 1, policy không bao giờ tới đích sẽ làm V giảm mãi, nên giới hạn số vòng quét."""
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        delta = 0.0
        for s in active:
            v = _backup(P[s][policy[s]], V, gamma)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V, sweeps


def policy_iteration(env, gamma=None, theta=1e-8, max_sweeps=1000, max_iter=1000):
    """Trả về V*, Q*, policy tối ưu, số lần cải thiện policy, tổng số vòng quét đánh giá."""
    gamma = env.gamma if gamma is None else gamma
    P = build_model(env)
    active = _active_states(env)
    V = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)       # policy ban đầu: luôn đi Lên
    total_sweeps, it = 0, 0
    for it in range(1, max_iter + 1):
        # 1) Đánh giá policy hiện tại
        V, sweeps = policy_evaluation(P, policy, V, gamma, active, theta, max_sweeps)
        total_sweeps += sweeps
        # 2) Cải thiện: chọn action tham lam theo V vừa tính
        stable = True
        for s in active:
            q = np.array([_backup(P[s][a], V, gamma) for a in range(env.n_actions)])
            best = int(np.argmax(q))
            if q[best] > q[policy[s]] + 1e-9:        # chỉ đổi khi tốt hơn thật (tránh lặp do hoà)
                policy[s] = best
                stable = False
        if stable:
            break
    Q = _q_from_v(env, P, V, gamma, active)
    return V, Q, policy, it, total_sweeps


# ===================================================================== công cụ so sánh
def reachable_states(env, policy, s0):
    """Các state (không terminal) có thể gặp khi đi theo policy từ s0."""
    seen, stack = {s0}, [s0]
    while stack:
        s = stack.pop()
        if env.is_terminal_state(s):
            continue
        for p, s2, _, _ in env.transition_distribution(s, int(policy[s])):
            if p > 0 and s2 not in seen:
                seen.add(s2)
                stack.append(s2)
    return [s for s in seen if not env.is_terminal_state(s)]


def policy_agreement(Q, Qstar, states, tol=1e-4):
    """Tỉ lệ state mà action greedy của Q cũng là action tối ưu theo Q*."""
    ok = [Qstar[s, int(np.argmax(Q[s]))] >= Qstar[s].max() - tol for s in states]
    return float(np.mean(ok)) if ok else float("nan")