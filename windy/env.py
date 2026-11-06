"""
Windy Grid World - môi trường tùy biến theo chuẩn gymnasium.Env.

State  : (hàng, cột, trạng_thái_neo) mã hoá thành 1 số nguyên (Discrete).
         trạng_thái_neo ∈ {0: chưa nhặt, 1: đang giữ (chưa dùng), 2: đã dùng}
         (map không có neo -> chỉ có trạng thái 0)
Action : 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
         anchor_mode="manual": 4..7 = cùng hướng 0..3 + "dùng neo ở bước này"

Thứ tự 1 bước chuyển:
  1) Di chuyển theo action (ra ngoài biên/đâm tường -> đứng yên, tính là va chạm)
  2) Lấy cường độ gió của CỘT Ô VỪA TỚI (sau khi đi), nếu gió ngẫu nhiên thì
     w = max(0, w_danh_dinh + δ), δ ∈ {-1, 0, +1} đều nhau
  3) Nếu đang giữ neo và (auto, hoặc manual + action 4..7) và w > 0 -> bỏ qua gió, neo -> "đã dùng"
  4) Nếu không: đẩy lên trên từng ô một, tối đa w ô, dừng khi gặp biên/tường
  5) Nếu kết thúc ở ô neo và trạng thái neo = 0 -> nhặt neo (=1)
  6) Kết thúc (terminated) nếu vị trí cuối là Goal; truncated khi hết max_steps
"""
from __future__ import annotations

import copy
import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
MOVE_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
NO_ANCHOR, HOLDING, USED = 0, 1, 2
ANCHOR_STATUS_NAMES = {NO_ANCHOR: "chưa nhặt", HOLDING: "đang giữ", USED: "đã dùng"}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class WindyGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 6}

    def __init__(self, config: dict, render_mode: str | None = None):
        super().__init__()
        self.cfg = copy.deepcopy(config)
        grid = self.cfg["grid"]
        self.n_rows, self.n_cols = len(grid), len(grid[0])
        if any(len(row) != self.n_cols for row in grid):
            raise ValueError("Mọi hàng trong 'grid' phải có cùng độ dài")
        self.wind = np.asarray(self.cfg["wind"], dtype=int)
        if self.wind.shape != (self.n_cols,):
            raise ValueError("'wind' phải có đúng 1 giá trị cho mỗi cột")

        self.walls = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        self.goals, self.anchor_cells, self.start = set(), set(), None
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls[r, c] = True
                elif ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goals.add((r, c))
                elif ch == "A":
                    self.anchor_cells.add((r, c))
                elif ch != ".":
                    raise ValueError(f"Ký tự không hợp lệ '{ch}' tại ({r},{c})")
        if self.start is None or not self.goals:
            raise ValueError("Map cần 1 ô 'S' và ít nhất 1 ô 'G'")

        self.stochastic = bool(self.cfg.get("stochastic_wind", False))
        self.noise_on_calm = bool(self.cfg.get("noise_on_calm", False))
        mode = self.cfg.get("anchor_mode", "manual")
        if mode not in ("none", "auto", "manual"):
            raise ValueError("anchor_mode phải là none | auto | manual")
        self.anchor_mode = mode if self.anchor_cells else "none"

        rw = self.cfg.get("reward", {})
        self.step_reward = float(rw.get("step", -1.0))
        self.goal_reward = float(rw.get("goal", 0.0))
        self.wall_penalty = float(rw.get("wall_bump", 0.0))
        self.anchor_bonus = float(rw.get("anchor_pickup", 0.0))
        self.gamma = float(self.cfg.get("gamma", 1.0))
        self.max_steps = int(self.cfg.get("max_steps", 1000))

        self.n_anchor_states = 3 if self.anchor_mode != "none" else 1
        self.n_actions = 8 if self.anchor_mode == "manual" else 4
        self.n_states = self.n_rows * self.n_cols * self.n_anchor_states
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.cell_size = int(self.cfg.get("cell_size", 48))
        self.window = self.clock = self.font = self.small_font = None

        self._pos, self._anchor, self._t = self.start, NO_ANCHOR, 0
        self._last_wind, self._trail = 0, [self.start]

    # ------------------------------------------------------------------ mã hoá
    def encode(self, r: int, c: int, anchor: int = 0) -> int:
        return (r * self.n_cols + c) * self.n_anchor_states + anchor

    def decode(self, s: int):
        anchor = s % self.n_anchor_states
        cell = s // self.n_anchor_states
        return cell // self.n_cols, cell % self.n_cols, anchor

    def is_free(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols and not self.walls[r, c]

    def is_valid_state(self, s: int) -> bool:
        r, c, _ = self.decode(s)
        return not self.walls[r, c]

    def is_terminal_state(self, s: int) -> bool:
        r, c, _ = self.decode(s)
        return (r, c) in self.goals

    def valid_states(self):
        return [s for s in range(self.n_states) if self.is_valid_state(s)]

    # ------------------------------------------------------------------ động lực học
    def wind_outcomes(self, c: int):
        """Danh sách (xác suất, cường độ gió thực tế) tại cột c."""
        nominal = int(self.wind[c])
        if self.stochastic and (nominal > 0 or self.noise_on_calm):
            return [(1 / 3, max(0, nominal + d)) for d in (-1, 0, 1)]
        return [(1.0, nominal)]

    def _move(self, r, c, action):
        """Bước 1: đi 1 ô theo hướng của action. Bị chặn (biên/tường) thì đứng yên."""
        dr, dc = MOVES[action % 4]
        nr, nc = r + dr, c + dc
        if not self.is_free(nr, nc):
            return r, c, True
        return nr, nc, False

    def _transition(self, r, c, anchor, action, wind_strength):
        """Chuyển trạng thái TẤT ĐỊNH khi đã biết cường độ gió thực tế
        (wind_strength là gió của cột Ô VỪA TỚI sau bước di chuyển)."""
        wants_anchor = action >= 4
        nr, nc, bumped = self._move(r, c, action)

        use_anchor = (
            anchor == HOLDING
            and wind_strength > 0
            and (self.anchor_mode == "auto" or (self.anchor_mode == "manual" and wants_anchor))
        )
        new_anchor = USED if use_anchor else anchor

        if not use_anchor:
            for _ in range(wind_strength):
                if self.is_free(nr - 1, nc):
                    nr -= 1
                else:
                    bumped = True
                    break

        picked = False
        if self.anchor_mode != "none" and new_anchor == NO_ANCHOR and (nr, nc) in self.anchor_cells:
            new_anchor, picked = HOLDING, True

        terminated = (nr, nc) in self.goals
        reward = self.step_reward
        if bumped:
            reward += self.wall_penalty
        if picked:
            reward += self.anchor_bonus
        if terminated:
            reward += self.goal_reward
        extra = {"bumped": bumped, "used_anchor": use_anchor, "picked_anchor": picked}
        return nr, nc, new_anchor, reward, terminated, extra

    def transition_distribution(self, state: int, action: int):
        """Mô hình p(s', r | s, a) dạng list (p, s', r, terminated) - dùng cho Value/Policy Iteration."""
        r, c, anchor = self.decode(state)
        _, dest_c, _ = self._move(r, c, action)   # gió lấy theo cột của ô vừa tới
        merged = {}
        for p, w in self.wind_outcomes(dest_c):
            nr, nc, na, rew, term, _ = self._transition(r, c, anchor, action, w)
            key = (self.encode(nr, nc, na), rew, term)
            merged[key] = merged.get(key, 0.0) + p
        return [(p, s2, rew, term) for (s2, rew, term), p in merged.items()]

    # ------------------------------------------------------------------ API gymnasium
    def _obs(self):
        return self.encode(self._pos[0], self._pos[1], self._anchor)

    def _info(self):
        return {"row": self._pos[0], "col": self._pos[1], "anchor": self._anchor, "t": self._t}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos, self._anchor, self._t = self.start, NO_ANCHOR, 0
        self._last_wind, self._trail = 0, [self.start]
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Action không hợp lệ: {action}")
        r, c = self._pos
        _, dest_c, _ = self._move(r, c, action)   # gió lấy theo cột của ô vừa tới
        outcomes = self.wind_outcomes(dest_c)
        w = outcomes[0][1] if len(outcomes) == 1 else outcomes[int(self.np_random.integers(len(outcomes)))][1]

        nr, nc, na, reward, terminated, extra = self._transition(r, c, self._anchor, action, w)
        self._pos, self._anchor, self._last_wind = (nr, nc), na, w
        self._t += 1
        self._trail.append(self._pos)
        truncated = (not terminated) and self._t >= self.max_steps

        info = self._info()
        info.update(extra)
        info["wind"] = w
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode in ("human", "rgb_array"):
            return self._render_frame()
        return None

    def _render_ansi(self):
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                if (r, c) == self._pos:
                    ch = "@"
                elif self.walls[r, c]:
                    ch = "#"
                elif (r, c) in self.goals:
                    ch = "G"
                elif (r, c) == self.start:
                    ch = "S"
                elif self.anchor_mode != "none" and self._anchor == NO_ANCHOR and (r, c) in self.anchor_cells:
                    ch = "A"
                else:
                    ch = "."
                row.append(ch)
            lines.append(" ".join(row))
        lines.append(" ".join(str(w) for w in self.wind) + "   <- gió")
        lines.append(f"t={self._t}  gió={self._last_wind}  neo={ANCHOR_STATUS_NAMES[self._anchor]}")
        return "\n".join(lines)

    @staticmethod
    def _blit_center(surface, text, center, font, color):
        img = font.render(text, True, color)
        surface.blit(img, img.get_rect(center=center))

    def _render_frame(self):
        import pygame

        cs = self.cell_size
        W, H = self.n_cols * cs, (self.n_rows + 1) * cs + 30
        if self.font is None:
            pygame.init()
            self.font = pygame.font.SysFont("arial", max(12, cs // 3), bold=True)
            self.small_font = pygame.font.SysFont("arial", 14)
        if self.render_mode == "human" and self.window is None:
            self.window = pygame.display.set_mode((W, H))
            pygame.display.set_caption(f"Windy Grid World - {self.cfg.get('name', '')}")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((W, H))
        canvas.fill((255, 255, 255))
        wmax = max(1, int(self.wind.max()))
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                if self.walls[r, c]:
                    color = (60, 60, 60)
                elif (r, c) in self.goals:
                    color = (110, 200, 110)
                elif (r, c) == self.start:
                    color = (150, 180, 240)
                else:
                    k = self.wind[c] / wmax
                    color = (int(255 - 70 * k), int(255 - 35 * k), 255)
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (190, 190, 190), rect, 1)
                if self.anchor_mode != "none" and self._anchor == NO_ANCHOR and (r, c) in self.anchor_cells:
                    pygame.draw.circle(canvas, (230, 170, 20), rect.center, cs // 4)
                    self._blit_center(canvas, "A", rect.center, self.small_font, (0, 0, 0))

        for (r, c) in self._trail[:-1]:
            pygame.draw.circle(canvas, (240, 120, 120), (c * cs + cs // 2, r * cs + cs // 2), max(2, cs // 10))
        r, c = self._pos
        center = (c * cs + cs // 2, r * cs + cs // 2)
        pygame.draw.circle(canvas, (200, 30, 30), center, cs // 3)
        if self._anchor == HOLDING:
            pygame.draw.circle(canvas, (230, 170, 20), center, cs // 3, 4)

        for c in range(self.n_cols):
            self._blit_center(canvas, str(self.wind[c]), (c * cs + cs // 2, self.n_rows * cs + cs // 2),
                              self.font, (30, 60, 160))
        status = f"t={self._t}  wind={self._last_wind}  anchor={['none', 'holding', 'used'][self._anchor]}"
        canvas.blit(self.small_font.render(status, True, (0, 0, 0)), (6, (self.n_rows + 1) * cs + 6))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None