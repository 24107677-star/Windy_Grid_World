"""
Windy Grid World - app xem agent chơi.

    python app.py

Menu: chọn map theo độ khó và 1 trong 5 thuật toán -> bấm "Bắt đầu".
Màn chơi: bấm "Bắt đầu" để agent đi, "Tạm dừng", "Chơi lại", "Menu"; chỉ tắt khi bấm "Thoát".
"""
import random
import sys
from pathlib import Path

import numpy as np
import pygame

from windy.agents import algo_settings, make_agent
from windy.env import HOLDING, MOVES, NO_ANCHOR, USED, WindyGridWorldEnv, load_config
from windy.planning import policy_iteration, value_iteration

# ------------------------------------------------------------------ cấu hình giao diện
W, H = 1200, 740
FPS = 60

LEVELS = [
    {"name": "Dễ", "title": "Map 1 · cổ điển 7×10", "desc": "Gió cố định, không có tường",
     "config": "configs/map1_classic.json", "color": (64, 168, 110)},
    {"name": "Trung bình", "title": "Map 2 · 10×14 có tường", "desc": "Gió 0–2 ô, ngẫu nhiên lệch ±1",
     "config": "configs/map2_stochastic.json", "color": (226, 140, 40)},
    {"name": "Khó", "title": "Map 3 · 12×16 có neo", "desc": "Gió ngẫu nhiên + 2 ô neo",
     "config": "configs/map3_anchor.json", "color": (204, 66, 66)},
]
ALGOS = [("qlearning", "Q-learning"), ("sarsa", "SARSA"), ("mc", "Monte Carlo"),
         ("vi", "Value Iteration"), ("pi", "Policy Iteration")]
DP_ALGOS = {"vi": value_iteration, "pi": policy_iteration}  # tính thẳng từ mô hình, không cần train
STEP_MS = 550  # thời gian mỗi bước (mili-giây): tăng số này để agent đi chậm hơn

BG = (240, 244, 249)
CARD = (255, 255, 255)
NAVY = (18, 40, 66)
TEXT = (28, 36, 48)
MUTED = (112, 122, 138)
LINE = (220, 227, 236)
WALL = (58, 66, 80)
GOAL = (86, 186, 120)
START = (126, 162, 232)
AGENT = (222, 60, 60)
GOLD = (240, 178, 40)
WIND_TINT = [(250, 252, 255), (222, 236, 252), (196, 222, 250), (172, 206, 246)]
ANCHOR_TEXT = {NO_ANCHOR: "Chưa nhặt neo", HOLDING: "Đang giữ neo", USED: "Đã dùng neo"}
ANCHOR_COLOR = {NO_ANCHOR: MUTED, HOLDING: GOLD, USED: (150, 110, 60)}
MOVE_VI = ["Lên", "Xuống", "Trái", "Phải"]


def font(size, bold=False):
    return pygame.font.SysFont("segoeui,arial,helvetica", size, bold=bold)


def draw_text(surf, text, fnt, color, pos, anchor="topleft"):
    img = fnt.render(text, True, color)
    rect = img.get_rect()
    setattr(rect, anchor, pos)
    surf.blit(img, rect)
    return rect


def mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Button:
    def __init__(self, rect, text, color, text_color=(255, 255, 255), outline=False):
        self.rect = pygame.Rect(rect)
        self.text, self.color, self.text_color, self.outline = text, color, text_color, outline

    def draw(self, surf, fnt, mouse, active=False):
        hover = self.rect.collidepoint(mouse)
        if self.outline and not active:
            fill = mix(CARD, self.color, 0.12) if hover else CARD
            pygame.draw.rect(surf, fill, self.rect, border_radius=10)
            pygame.draw.rect(surf, self.color, self.rect, width=2, border_radius=10)
            col = self.color
        else:
            fill = mix(self.color, (0, 0, 0), 0.12) if hover else self.color
            pygame.draw.rect(surf, fill, self.rect, border_radius=10)
            col = self.text_color
        draw_text(surf, self.text, fnt, col, self.rect.center, "center")

    def hit(self, pos):
        return self.rect.collidepoint(pos)


# ------------------------------------------------------------------ huấn luyện từng phần (có thanh tiến trình)
def train_iter(cfg, algo, seed=0, chunk=50):
    env = WindyGridWorldEnv(cfg)
    agent = make_agent(algo, env, seed=seed)
    n = int(algo_settings(cfg, algo).get("episodes", 500))
    s, _ = env.reset(seed=seed)
    for ep in range(n):
        if ep > 0:
            s, _ = env.reset()
        a = agent.start(s)
        while True:
            s2, r, term, trunc, _ = env.step(a)
            a2 = agent.observe(s, a, r, s2, term, trunc)
            s, a = s2, a2
            if term or trunc:
                break
        agent.end_episode()
        if (ep + 1) % chunk == 0 or ep + 1 == n:
            yield (ep + 1) / n, agent


# ------------------------------------------------------------------ app
class App:
    def __init__(self):
        if sys.platform == "win32":  # tránh Windows phóng to cửa sổ làm lẹm mép
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Windy Grid World")
        self.clock = pygame.time.Clock()
        self.f_title = font(46, True)
        self.f_h1 = font(26, True)
        self.f_h2 = font(20, True)
        self.f_body = font(17)
        self.f_small = font(14)
        self.f_cell = font(18, True)
        self.level, self.algo = 0, 0
        self.mode = "menu"
        self.running_app = True
        self.build_menu()

    # ---------------------------------------------------------------- menu
    def build_menu(self):
        self.level_cards = []
        n = len(LEVELS)
        cw, ch, gap = 320, 150, 24
        x0 = (W - (n * cw + (n - 1) * gap)) // 2
        for i in range(n):
            self.level_cards.append(pygame.Rect(x0 + i * (cw + gap), 230, cw, ch))
        bw, bh, gx, gy = 220, 44, 16, 12  # 5 nút thuật toán: hàng trên 3 nút, hàng dưới 2 nút
        self.algo_btns = []
        for i, (_, lab) in enumerate(ALGOS):
            row, col = i // 3, i % 3
            n_in_row = min(3, len(ALGOS) - row * 3)
            bx0 = W // 2 - (n_in_row * bw + (n_in_row - 1) * gx) // 2
            self.algo_btns.append(Button((bx0 + col * (bw + gx), 450 + row * (bh + gy), bw, bh), lab, NAVY, outline=True))
        self.menu_start = Button((W // 2 - 230, 600, 220, 60), "Bắt đầu", (46, 134, 193))
        self.menu_quit = Button((W // 2 + 10, 600, 220, 60), "Thoát", (150, 158, 170))

    def draw_menu(self, mouse):
        s = self.screen
        s.fill(BG)
        draw_text(s, "WINDY GRID WORLD", self.f_title, NAVY, (W // 2, 70), "center")
        draw_text(s, "Xem agent học tăng cường vượt gió để tới đích", self.f_body, MUTED, (W // 2, 120), "center")
        draw_text(s, "1. Chọn map theo độ khó", self.f_h2, TEXT, (self.level_cards[0].x, 192))
        for i, (rect, lv) in enumerate(zip(self.level_cards, LEVELS)):
            sel = i == self.level
            hover = rect.collidepoint(mouse)
            pygame.draw.rect(s, mix(CARD, lv["color"], 0.08) if (sel or hover) else CARD, rect, border_radius=16)
            pygame.draw.rect(s, lv["color"] if sel else LINE, rect, width=3 if sel else 1, border_radius=16)
            pill = pygame.Rect(rect.x + 18, rect.y + 18, 110, 30)
            pygame.draw.rect(s, lv["color"], pill, border_radius=15)
            draw_text(s, lv["name"], self.f_small, (255, 255, 255), pill.center, "center")
            for k in range(i + 1):  # số "vạch" độ khó
                pygame.draw.rect(s, lv["color"], (rect.right - 30 - k * 14, rect.y + 22, 9, 22), border_radius=3)
            draw_text(s, lv["title"], self.f_h2, TEXT, (rect.x + 18, rect.y + 66))
            draw_text(s, lv["desc"], self.f_small, MUTED, (rect.x + 18, rect.y + 100))
        draw_text(s, "2. Chọn thuật toán", self.f_h2, TEXT, (W // 2, 420), "center")
        for i, b in enumerate(self.algo_btns):
            b.draw(s, self.f_body, mouse, active=i == self.algo)
        self.menu_start.draw(s, self.f_h2, mouse)
        self.menu_quit.draw(s, self.f_h2, mouse)

    def click_menu(self, pos):
        for i, r in enumerate(self.level_cards):
            if r.collidepoint(pos):
                self.level = i
        for i, b in enumerate(self.algo_btns):
            if b.hit(pos):
                self.algo = i
        if self.menu_start.hit(pos):
            self.open_game()
        if self.menu_quit.hit(pos):
            self.running_app = False

    # ---------------------------------------------------------------- nạp model / train
    def open_game(self):
        lv = LEVELS[self.level]
        self.cfg = load_config(lv["config"])
        self.env = WindyGridWorldEnv(self.cfg)
        algo = ALGOS[self.algo][0]
        path = Path(f"results/models/{self.cfg['name']}_{algo}.npy")
        Q = np.load(path) if path.exists() else None
        if Q is None or Q.shape != (self.env.n_states, self.env.n_actions):
            if algo in DP_ALGOS:
                Q = self.solve_with_dp(algo, path)
            else:
                Q = self.train_with_progress(algo, path)
            if Q is None:
                return
        self.Q = Q
        self.setup_board()
        self.reset_episode()
        self.mode = "game"

    def solve_with_dp(self, algo, path):
        """Policy/Value Iteration: tính policy tối ưu trực tiếp từ mô hình của môi trường."""
        s = self.screen
        s.fill(BG)
        draw_text(s, f"Đang tính policy tối ưu bằng {dict(ALGOS)[algo]}...", self.f_h1, NAVY, (W // 2, 320), "center")
        draw_text(s, "Thuật toán này dùng mô hình của môi trường, không cần chơi thử.", self.f_body, MUTED,
                  (W // 2, 360), "center")
        pygame.display.flip()
        pygame.event.pump()
        Q = DP_ALGOS[algo](self.env)[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, Q)
        return Q

    def train_with_progress(self, algo, path):
        lv = LEVELS[self.level]
        quit_btn = Button((W // 2 - 90, 470, 180, 50), "Thoát", (150, 158, 170))
        agent = None
        for frac, agent in train_iter(self.cfg, algo):
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.MOUSEBUTTONDOWN and quit_btn.hit(ev.pos)):
                    self.running_app = False
                    return None
            s = self.screen
            s.fill(BG)
            draw_text(s, f"Đang huấn luyện agent cho map {lv['name']}...", self.f_h1, NAVY, (W // 2, 300), "center")
            draw_text(s, "Chỉ cần làm một lần, lần sau app dùng lại model đã lưu.", self.f_body, MUTED, (W // 2, 340), "center")
            bar = pygame.Rect(W // 2 - 300, 385, 600, 26)
            pygame.draw.rect(s, LINE, bar, border_radius=13)
            pygame.draw.rect(s, lv["color"], (bar.x, bar.y, max(26, int(bar.w * frac)), bar.h), border_radius=13)
            draw_text(s, f"{int(frac * 100)}%", self.f_body, TEXT, (W // 2, 430), "center")
            quit_btn.draw(s, self.f_h2, pygame.mouse.get_pos())
            pygame.display.flip()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, agent.Q)
        return agent.Q

    # ---------------------------------------------------------------- màn chơi
    def setup_board(self):
        env = self.env
        area_w, area_h = 780, 600
        self.cell = max(24, min(area_w // env.n_cols, (area_h - 40) // env.n_rows, 72))
        self.gw, self.gh = env.n_cols * self.cell, env.n_rows * self.cell
        self.gx = 30 + (area_w - self.gw) // 2
        self.gy = 100 + (area_h - 40 - self.gh) // 2
        px = 840
        self.btn_play = Button((px, 540, 160, 52), "Bắt đầu", (46, 134, 193))
        self.btn_reset = Button((px + 175, 540, 160, 52), "Chơi lại", NAVY, outline=True)
        self.btn_menu = Button((px, 608, 160, 52), "Menu", NAVY, outline=True)
        self.btn_quit = Button((px + 175, 608, 160, 52), "Thoát", (150, 158, 170))

    def reset_episode(self):
        self.seed = random.randint(0, 10**6)
        self.obs, info = self.env.reset(seed=self.seed)
        self.pos, self.anchor = (info["row"], info["col"]), info["anchor"]
        self.trail = [self.pos]
        self.t, self.G = 0, 0.0
        self.playing, self.done, self.success = False, False, False
        self.last_action, self.last_wind = None, None
        self.log = ["Bấm “Bắt đầu” để agent bắt đầu đi."]
        self.anim = None  # (điểm đầu, điểm giữa, điểm cuối, thời điểm bắt đầu)
        self.next_step_at = 0

    def cell_center(self, rc):
        r, c = rc
        return (self.gx + c * self.cell + self.cell / 2, self.gy + r * self.cell + self.cell / 2)

    def do_step(self, now):
        env = self.env
        a = int(np.argmax(self.Q[self.obs]))
        r0, c0 = self.pos
        dr, dc = MOVES[a % 4]
        mid = (r0 + dr, c0 + dc) if env.is_free(r0 + dr, c0 + dc) else (r0, c0)
        self.obs, rew, term, trunc, info = env.step(a)
        new = (info["row"], info["col"])
        self.anim = (self.pos, mid, new, now)
        self.pos, self.anchor = new, info["anchor"]
        self.trail.append(new)
        self.t, self.G = info["t"], self.G + rew
        self.last_action, self.last_wind = a, info["wind"]
        msg = f"Bước {self.t}: {MOVE_VI[a % 4]}"
        if a >= 4:
            msg += " + dùng neo"
        if info["used_anchor"]:
            msg += "  → bỏ qua gió!"
        elif info["wind"] > 0:
            msg += f"  → gió đẩy {info['wind']} ô"
        if info["picked_anchor"]:
            msg += "  → nhặt được neo"
        self.log.append(msg)
        self.log = self.log[-7:]
        if term or trunc:
            self.done, self.playing, self.success = True, False, term
            self.btn_play.text = "Bắt đầu"
            self.log.append("Tới đích!" if term else "Hết số bước cho phép.")

    def agent_pixel(self, now):
        if self.anim is None:
            return self.cell_center(self.pos)
        p0, mid, p1, t0 = self.anim
        dur = STEP_MS * 0.8
        k = min(1.0, (now - t0) / dur)
        a, b, e = self.cell_center(p0), self.cell_center(mid), self.cell_center(p1)
        if mid == p1:  # không bị gió đẩy
            k = k * k * (3 - 2 * k)
            return (a[0] + (e[0] - a[0]) * k, a[1] + (e[1] - a[1]) * k)
        if k < 0.5:  # nửa đầu: bước đi
            u = k / 0.5
            u = u * u * (3 - 2 * u)
            return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
        u = (k - 0.5) / 0.5  # nửa sau: gió đẩy
        u = u * u * (3 - 2 * u)
        return (b[0] + (e[0] - b[0]) * u, b[1] + (e[1] - b[1]) * u)

    def draw_board(self, now):
        s, env, cs = self.screen, self.env, self.cell
        board = pygame.Rect(self.gx - 16, self.gy - 16, self.gw + 32, self.gh + 56)
        pygame.draw.rect(s, CARD, board, border_radius=18)
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                rect = pygame.Rect(self.gx + c * cs + 2, self.gy + r * cs + 2, cs - 4, cs - 4)
                if env.walls[r, c]:
                    color = WALL
                elif (r, c) in env.goals:
                    color = GOAL
                elif (r, c) == env.start:
                    color = START
                else:
                    color = WIND_TINT[min(3, int(env.wind[c]))]
                pygame.draw.rect(s, color, rect, border_radius=max(4, cs // 8))
        # gió chạy lên trong các cột có gió
        for c in range(env.n_cols):
            w = int(env.wind[c])
            if w <= 0:
                continue
            x = self.gx + c * cs + cs / 2
            for k in range(2 * w):
                y = self.gy + self.gh - ((now * 0.06 * w + k * self.gh / (2 * w)) % self.gh)
                rr, cc = int((y - self.gy) // cs), c
                if not (0 <= rr < env.n_rows) or env.walls[rr, cc] or (rr, cc) in env.goals or (rr, cc) == env.start:
                    continue
                pts = [(x - cs * 0.12, y + cs * 0.06), (x, y - cs * 0.06), (x + cs * 0.12, y + cs * 0.06)]
                pygame.draw.lines(s, (150, 190, 236), False, pts, 2)
        # ô đặc biệt
        for (r, c) in env.goals:
            draw_text(s, "G", self.f_cell, (255, 255, 255), self.cell_center((r, c)), "center")
        draw_text(s, "S", self.f_cell, (255, 255, 255), self.cell_center(env.start), "center")
        if env.anchor_mode != "none" and self.anchor == NO_ANCHOR:
            for rc in env.anchor_cells:
                cx, cy = self.cell_center(rc)
                pygame.draw.circle(s, GOLD, (int(cx), int(cy)), int(cs * 0.32))
                draw_text(s, "A", self.f_cell, (90, 60, 0), (cx, cy), "center")
        # vết đường đi
        for i, rc in enumerate(self.trail[:-1]):
            if rc == env.start:
                continue
            cx, cy = self.cell_center(rc)
            pygame.draw.circle(s, mix((255, 255, 255), AGENT, 0.35 + 0.4 * i / max(1, len(self.trail))), (int(cx), int(cy)), max(3, cs // 10))
        # agent
        ax, ay = self.agent_pixel(now)
        rad = int(cs * 0.34)
        pygame.draw.circle(s, (200, 205, 215), (int(ax) + 2, int(ay) + 3), rad)
        pygame.draw.circle(s, AGENT, (int(ax), int(ay)), rad)
        if self.anchor == HOLDING:
            pygame.draw.circle(s, GOLD, (int(ax), int(ay)), rad + 4, 4)
        pygame.draw.circle(s, (255, 255, 255), (int(ax - rad * 0.3), int(ay - rad * 0.3)), max(2, rad // 4))
        # cường độ gió dưới mỗi cột
        for c in range(env.n_cols):
            w = int(env.wind[c])
            cx = self.gx + c * cs + cs / 2
            draw_text(s, str(w), self.f_small, (52, 110, 190) if w else MUTED, (cx, self.gy + self.gh + 20), "center")

    def draw_panel(self, mouse):
        s, lv = self.screen, LEVELS[self.level]
        panel = pygame.Rect(820, 100, 355, 420)
        pygame.draw.rect(s, CARD, panel, border_radius=18)
        x = panel.x + 22
        pill = pygame.Rect(x, panel.y + 20, 110, 30)
        pygame.draw.rect(s, lv["color"], pill, border_radius=15)
        draw_text(s, lv["name"], self.f_small, (255, 255, 255), pill.center, "center")
        draw_text(s, ALGOS[self.algo][1], self.f_body, MUTED, (pill.right + 12, pill.y + 4))
        draw_text(s, lv["title"], self.f_h2, TEXT, (x, panel.y + 60))
        rows = [
            ("Số bước", str(self.t)),
            ("Return", f"{self.G:.0f}"),
            ("Hành động", "–" if self.last_action is None else MOVE_VI[self.last_action % 4] + (" + neo" if self.last_action >= 4 else "")),
            ("Gió vừa thổi", "–" if self.last_wind is None else f"{self.last_wind} ô"),
        ]
        for i, (k, v) in enumerate(rows):
            y = panel.y + 100 + i * 30
            draw_text(s, k, self.f_body, MUTED, (x, y))
            draw_text(s, v, self.f_body, TEXT, (panel.right - 22, y), "topright")
        if self.env.anchor_mode != "none":
            y = panel.y + 220
            draw_text(s, "Neo", self.f_body, MUTED, (x, y))
            col = ANCHOR_COLOR[self.anchor]
            tag = draw_text(s, ANCHOR_TEXT[self.anchor], self.f_body, col, (panel.right - 22, y), "topright")
            pygame.draw.circle(s, col, (tag.x - 12, tag.centery), 6)
        pygame.draw.line(s, LINE, (x, panel.y + 255), (panel.right - 22, panel.y + 255), 1)
        for i, line in enumerate(self.log[-6:]):
            draw_text(s, line, self.f_small, TEXT if i == len(self.log[-6:]) - 1 else MUTED, (x, panel.y + 268 + i * 23))
        self.btn_play.text = "Tạm dừng" if self.playing else ("Chơi tiếp" if self.t > 0 and not self.done else "Bắt đầu")
        self.btn_play.draw(s, self.f_h2, mouse)
        self.btn_reset.draw(s, self.f_h2, mouse)
        self.btn_menu.draw(s, self.f_h2, mouse)
        self.btn_quit.draw(s, self.f_h2, mouse)

    def draw_game(self, now, mouse):
        s = self.screen
        s.fill(BG)
        draw_text(s, "WINDY GRID WORLD", self.f_h1, NAVY, (30, 30))
        draw_text(s, "Ô xanh đậm = gió mạnh · số dưới mỗi cột = cường độ gió · vòng vàng = đang giữ neo",
                  self.f_small, MUTED, (30, 66))
        self.draw_board(now)
        self.draw_panel(mouse)
        if self.done and (self.anim is None or now - self.anim[3] > STEP_MS * 0.8):
            msg = f"Tới đích sau {self.t} bước!" if self.success else "Hết số bước cho phép"
            col = GOAL if self.success else AGENT
            box = pygame.Rect(0, 0, 420, 70)
            box.center = (self.gx + self.gw // 2, self.gy + self.gh // 2)
            pygame.draw.rect(s, CARD, box, border_radius=16)
            pygame.draw.rect(s, col, box, width=3, border_radius=16)
            draw_text(s, msg, self.f_h1, col, box.center, "center")

    def click_game(self, pos, now):
        if self.btn_play.hit(pos):
            if self.done:
                self.reset_episode()
                self.playing = True
            else:
                self.playing = not self.playing
            self.next_step_at = now
        elif self.btn_reset.hit(pos):
            self.reset_episode()
        elif self.btn_menu.hit(pos):
            self.mode = "menu"
        elif self.btn_quit.hit(pos):
            self.running_app = False

    # ---------------------------------------------------------------- vòng lặp chính
    def run(self):
        while self.running_app:
            now = pygame.time.get_ticks()
            mouse = pygame.mouse.get_pos()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running_app = False
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self.mode == "menu":
                        self.click_menu(ev.pos)
                    else:
                        self.click_game(ev.pos, now)
            if not self.running_app:
                break
            if self.mode == "game":
                if self.playing and not self.done and now >= self.next_step_at:
                    self.do_step(now)
                    self.next_step_at = now + STEP_MS
                self.draw_game(now, mouse)
            else:
                self.draw_menu(mouse)
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == "__main__":
    App().run()
    sys.exit()