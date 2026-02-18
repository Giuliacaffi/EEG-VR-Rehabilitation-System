# -*- coding: utf-8 -*-
"""
Gamified EEG Level 1 (pygame) — NEW UI/Flow
- Binary task: Up Arm (code=1) vs No Movement (code=6)
- Screens:
  1) Welcome Back (effects, colored)
  2) Mood
  3) Background mode (App default vs Choose now)
  4) Background gallery from assets/backgrounds (if choose now)
- Rest/ITI: CALM bubble larger + waves
- Preparation: ONLY semaphore dots (red->yellow->green), no confusing text
- Movement: HUD + avatar + optional feedback
- LSL markers:
    Session start: 900 IMAGINED / 901 EXECUTED
    Rest: 100
    Prep: 200
    Movement: class code only (1 or 6)
    Session end: 902
- Predictions: PRED_MODE = "FAKE" | "KEYS" | "LSL"

VIDEO AVATAR (moviepy):
- Uses assets/video/uparm.mp4 and assets/video/nomove.mp4
- Renders frames via clip.get_frame(t) into pygame surfaces
"""

import os
import csv
import time
import json
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import pygame
import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

# --- VIDEO (moviepy) ---
from moviepy.editor import VideoFileClip


# =====================
# PATHS
# =====================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

def rel(*parts) -> str:
    return str(SCRIPT_DIR.joinpath(*parts))

ASSETS_DIR = Path(rel("assets"))
BG_DIR = ASSETS_DIR / "backgrounds"
SFX_DIR = ASSETS_DIR / "sfx"
VIDEO_DIR = ASSETS_DIR / "video"
STATE_PATH = Path(rel("player_state.json"))


# =====================
# CONFIG
# =====================
@dataclass
class Config:
    # session
    SESSION_MODE: str = "EXECUTED"  # EXECUTED | IMAGINED
    SESSION_NUM: int = 1
    SUBJECT: str = "S01"
    EXPERIMENTER: str = "Giulia"

    # player/game
    PLAYER_NAME: str = "Rebecca"
    LEVEL: int = 1

    # timing
    REST_SEC: float = 3.0
    PREP_SEC: float = 3.0
    MOVE_SEC: float = 3.0
    ITI_SEC: float = 3.0
    ITI_JITTER_MAX: float = 2.0

    # trials
    N_NOFEEDBACK_PER_CLASS: int = 2
    N_FEEDBACK_PER_CLASS: int = 2

    # visuals
    FULLSCREEN: bool = True
    WINDOW_SIZE: Tuple[int, int] = (1400, 900)
    FONT_NAME: str = "Arial"

    # lsl
    MARK_REST: int = 100
    MARK_PREP: int = 200
    MARK_SESSION_IMAGINED: int = 900
    MARK_SESSION_EXECUTED: int = 901
    MARK_SESSION_END: int = 902

    CLASS_CODES: Dict[str, int] = None  # filled below

    # output
    OUTPUT_DIR: str = "./logs"

    # prediction mode: "FAKE" | "KEYS" | "LSL"
    PRED_MODE: str = "FAKE"
    LSL_PRED_STREAM_NAME: str = "Predictions"  # expected int32: 1 or 0
    LSL_PRED_TIMEOUT: float = 0.001

    # --- VIDEO AVATAR (moviepy) ---
    VIDEO_UPARM: str = "uparm.mp4"
    VIDEO_NOMOVE: str = "nomove.mp4"
    VIDEO_RENDER_SIZE: Tuple[int, int] = (520, 520)


CFG = Config()
CFG.CLASS_CODES = {"Up Arm": 1, "No Movement": 6}


# =====================
# THEME
# =====================
class Theme:
    # background gradient
    GRAD_TOP   = (120, 210, 255)
    GRAD_BOT   = (255, 150, 210)

    # neutrals
    INK        = (16, 16, 18)
    INK_SOFT   = (60, 60, 65)
    WHITE      = (255, 255, 255)
    CARD       = (255, 255, 255)
    CARD_TINT  = (245, 250, 255)
    OUTLINE    = (20, 20, 25)

    # accents
    ACCENT     = (80, 140, 255)
    ACCENT_2   = (255, 160, 90)

    GOOD       = (70, 210, 140)
    BAD        = (245, 90, 90)

    # overlays
    DIM_ALPHA  = 110
    PAUSE_ALPHA= 175

    # radii
    R_CARD     = 28
    R_BTN      = 22
    R_PILL     = 999

    # semaphore (prep)
    RED        = (240, 80, 80)
    YELLOW     = (255, 205, 70)
    GREEN      = (70, 210, 140)


# =====================
# UTIL
# =====================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def now_ts() -> float:
    return time.perf_counter()

def clamp(x, a, b):
    return max(a, min(b, x))

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_col(c1, c2, t):
    return (int(lerp(c1[0], c2[0], t)),
            int(lerp(c1[1], c2[1], t)),
            int(lerp(c1[2], c2[2], t)))

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"level": 1, "best_acc": None, "bg": None, "mood": None, "last_acc": None}

def save_state(state: dict):
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass

def list_backgrounds() -> List[Path]:
    if not BG_DIR.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    items = [p for p in BG_DIR.iterdir() if p.suffix.lower() in exts]
    return sorted(items)

def draw_rounded_rect(screen, rect, color, radius=18, border=0, border_color=None):
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    if border > 0 and border_color is not None:
        pygame.draw.rect(screen, border_color, rect, width=border, border_radius=radius)

def draw_shadow_rect(screen, rect, radius, shadow_alpha=60, dy=10, spread=6):
    w, h = rect.w + spread*2, rect.h + spread*2
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for i in range(5):
        a = int(shadow_alpha * (1 - i/5))
        rr = pygame.Rect(spread-i, spread-i, rect.w + 2*i, rect.h + 2*i)
        pygame.draw.rect(s, (0, 0, 0, a), rr, border_radius=radius+6)
    screen.blit(s, (rect.x - spread, rect.y - spread + dy))

def glow_rect(screen, rect: pygame.Rect, color=(80, 140, 255), strength=90, steps=6, radius=26):
    """Soft outer glow around a rounded rect."""
    g = pygame.Surface((rect.w + 140, rect.h + 140), pygame.SRCALPHA)
    base = pygame.Rect(70, 70, rect.w, rect.h)
    for i in range(steps, 0, -1):
        a = int(strength * (i/steps)**2)
        pygame.draw.rect(g, (color[0], color[1], color[2], a), base.inflate(18*i, 18*i), border_radius=radius+10)
    screen.blit(g, (rect.x-70, rect.y-70))

def draw_center(screen, surf, y):
    r = surf.get_rect(center=(screen.get_width()//2, y))
    screen.blit(surf, r)

def draw_shadow_text(screen, font, text, y, fg, shadow=(0,0,0), dy=2):
    s1 = font.render(text, True, shadow)
    s2 = font.render(text, True, fg)
    draw_center(screen, s1, y+dy)
    draw_center(screen, s2, y)

def gradient_bg(screen, top=Theme.GRAD_TOP, bottom=Theme.GRAD_BOT):
    w, h = screen.get_size()
    step = 4
    for i in range(0, h, step):
        t = i / max(1, h-1)
        c = lerp_col(top, bottom, t)
        pygame.draw.rect(screen, c, pygame.Rect(0, i, w, step))

def dim_overlay(screen, alpha=Theme.DIM_ALPHA, color=(255,255,255)):
    ov = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    ov.fill((color[0], color[1], color[2], alpha))
    screen.blit(ov, (0, 0))

def fade_transition(screen, clock, draw_under_fn, duration=0.28, fade_to_white=True):
    t0 = now_ts()
    while True:
        dt = now_ts() - t0
        p = clamp(dt / max(0.001, duration), 0.0, 1.0)
        draw_under_fn()
        a = int(255 * (1.0 - p))
        col = (255, 255, 255) if fade_to_white else (0, 0, 0)
        ov = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        ov.fill((col[0], col[1], col[2], a))
        screen.blit(ov, (0, 0))
        pygame.display.flip()
        if p >= 1.0:
            break
        clock.tick(60)

def star_rating(acc: float) -> int:
    if acc >= 0.90: return 3
    if acc >= 0.80: return 2
    if acc >= 0.70: return 1
    return 0

# --- NEW: text wrap (serve per far "contenere" i testi nei box)
def wrap_lines(text: str, font: pygame.font.Font, max_w: int) -> List[str]:
    """
    Word-wrap semplice per pygame.
    Ritorna lista di righe che stanno dentro max_w.
    """
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# =====================
# SFX (optional)
# =====================
class SFX:
    def __init__(self):
        self.enabled = False
        self.sounds = {}
        try:
            pygame.mixer.init()
            self.enabled = True
        except Exception:
            self.enabled = False

    def _load(self, name, filename):
        if not self.enabled:
            return
        path = SFX_DIR / filename
        if path.exists():
            try:
                self.sounds[name] = pygame.mixer.Sound(str(path))
            except Exception:
                pass

    def load_default_pack(self):
        self._load("start", "start.wav")
        self._load("good", "good.wav")
        self._load("bad", "bad.wav")
        self._load("select", "select.wav")

    def play(self, name):
        if not self.enabled:
            return
        s = self.sounds.get(name)
        if s:
            try:
                s.play()
            except Exception:
                pass


# =====================
# Ambient particles + sparkles
# =====================
class AmbientParticle:
    def __init__(self, W, H):
        self.reset(W, H)

    def reset(self, W, H):
        self.x = random.uniform(0, W)
        self.y = random.uniform(0, H)
        self.r = random.uniform(2.5, 7.0)
        self.vy = random.uniform(-18, -7)
        self.vx = random.uniform(-8, 8)
        self.a = random.uniform(18, 40)

    def update(self, dt, W, H):
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.y < -20 or self.x < -40 or self.x > W + 40:
            self.reset(W, H)
            self.y = H + 20

    def draw(self, screen):
        s = pygame.Surface((int(self.r*2+2), int(self.r*2+2)), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 255, 255, int(self.a)), (s.get_width()//2, s.get_height()//2), int(self.r))
        screen.blit(s, (self.x - s.get_width()//2, self.y - s.get_height()//2))


class Sparkle:
    def __init__(self, x, y, good=True):
        self.x, self.y = x, y
        self.life = 0.35
        self.t = 0.0
        self.good = good
        self.vx = random.uniform(-90, 90)
        self.vy = random.uniform(-120, -40)
        self.size = random.uniform(6, 12)

    def update(self, dt):
        self.t += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 260 * dt

    def alive(self):
        return self.t < self.life

    def draw(self, screen):
        p = clamp(1 - self.t / self.life, 0, 1)
        a = int(200 * p)
        col = Theme.GOOD if self.good else Theme.BAD
        s = pygame.Surface((64, 64), pygame.SRCALPHA)
        cx, cy = 32, 32
        for k in range(4):
            ang = k * (np.pi/2)
            dx = int(np.cos(ang) * self.size)
            dy = int(np.sin(ang) * self.size)
            pygame.draw.line(s, (col[0], col[1], col[2], a), (cx-dx, cy-dy), (cx+dx, cy+dy), width=3)
        screen.blit(s, (self.x-32, self.y-32))


# =====================
# Button
# =====================
class Button:
    def __init__(self, rect: pygame.Rect, label: str, font: pygame.font.Font, kind="primary"):
        self.rect = rect
        self.label = label
        self.font = font
        self.enabled = True
        self.kind = kind
        self.pressed_t = 0.0

    def is_hover(self, mx, my):
        return self.rect.collidepoint(mx, my)

    def draw(self, screen, mx, my):
        hov = self.is_hover(mx, my) and self.enabled
        press = self.pressed_t > 0.0

        if self.kind == "primary":
            fill = Theme.ACCENT
            fill2 = lerp_col(Theme.ACCENT, (255,255,255), 0.18)
            txtc = Theme.WHITE
            outline = Theme.OUTLINE
        elif self.kind == "secondary":
            fill = Theme.CARD
            fill2 = Theme.CARD_TINT
            txtc = Theme.INK
            outline = Theme.OUTLINE
        else:
            fill = Theme.CARD_TINT
            fill2 = Theme.CARD_TINT
            txtc = Theme.INK
            outline = Theme.OUTLINE

        r = self.rect.copy()
        if press:
            r.y += 2

        draw_shadow_rect(screen, r, radius=Theme.R_BTN, shadow_alpha=80 if hov else 60, dy=10 if hov else 9)
        draw_rounded_rect(screen, r, fill, radius=Theme.R_BTN, border=0)
        top_h = int(r.h * 0.45)
        top_rect = pygame.Rect(r.x, r.y, r.w, top_h)
        draw_rounded_rect(screen, top_rect, fill2, radius=Theme.R_BTN, border=0)
        pygame.draw.rect(screen, outline, r, width=2, border_radius=Theme.R_BTN)

        text = self.font.render(self.label, True, txtc)
        screen.blit(text, text.get_rect(center=r.center))

    def click(self, mx, my) -> bool:
        if self.enabled and self.rect.collidepoint(mx, my):
            self.pressed_t = 0.10
            return True
        return False

    def tick(self, dt):
        if self.pressed_t > 0:
            self.pressed_t = max(0.0, self.pressed_t - dt)


def pill(screen, rect, text, font, fill, outline=None, text_col=Theme.INK):
    draw_rounded_rect(screen, rect, fill, radius=Theme.R_PILL, border=2 if outline else 0, border_color=outline)
    t = font.render(text, True, text_col)
    screen.blit(t, t.get_rect(center=rect.center))

def progress_bar(screen, rect, p, fg=Theme.ACCENT, bg=(230,235,245), outline=Theme.OUTLINE):
    draw_rounded_rect(screen, rect, bg, radius=Theme.R_PILL, border=2, border_color=outline)
    inner = rect.inflate(-6, -6)
    fillw = int(inner.w * clamp(p, 0, 1))
    if fillw > 0:
        fr = pygame.Rect(inner.x, inner.y, fillw, inner.h)
        draw_rounded_rect(screen, fr, fg, radius=Theme.R_PILL, border=0)


# =====================
# LSL Markers
# =====================
def create_marker_outlet():
    info = StreamInfo(
        name="Markers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="int32",
        source_id="eeg_game_level1_markers",
    )
    desc = info.desc()
    desc.append_child_value("manufacturer", "GiuliaLab")
    classes = desc.append_child("classes")
    for label, code in CFG.CLASS_CODES.items():
        n = classes.append_child("class")
        n.append_child_value("label", label)
        n.append_child_value("code", str(code))
    return StreamOutlet(info)

def emit_marker(outlet: StreamOutlet, writer: csv.writer, event_idx: int,
                code: int, trial_idx: int, phase: str, class_label: str, scheduled_dur: Optional[float]):
    t_lsl = local_clock()
    outlet.push_sample([int(code)], timestamp=t_lsl)
    t_mono = now_ts()
    t_wall = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    writer.writerow([
        event_idx, CFG.SUBJECT, CFG.SESSION_NUM, trial_idx, phase, class_label,
        int(code), f"{t_lsl:.6f}", f"{t_mono:.6f}", t_wall, CFG.SESSION_MODE,
        CFG.EXPERIMENTER, f"{scheduled_dur:.3f}" if scheduled_dur is not None else ""
    ])

def resolve_pred_inlet() -> Optional[StreamInlet]:
    try:
        streams = resolve_byprop("name", CFG.LSL_PRED_STREAM_NAME, timeout=1.0)
        if not streams:
            return None
        return StreamInlet(streams[0])
    except Exception:
        return None


# =====================
# Prediction helper
# =====================
class Predictor:
    """
    Returns predicted label for current movement window. For Level 1:
    - Up Arm -> 1
    - No Movement -> 0
    """
    def __init__(self, mode: str):
        self.mode = mode.upper()
        self.inlet = resolve_pred_inlet() if self.mode == "LSL" else None
        self._last_key = None

    def get_pred(self) -> Optional[int]:
        if self.mode == "FAKE":
            return int(random.random() > 0.5)
        if self.mode == "KEYS":
            return None
        if self.mode == "LSL":
            if self.inlet is None:
                self.inlet = resolve_pred_inlet()
                if self.inlet is None:
                    return None
            sample, _ = self.inlet.pull_sample(timeout=CFG.LSL_PRED_TIMEOUT)
            if sample is None:
                return None
            try:
                return int(sample[0])
            except Exception:
                return None
        return None


# =====================
# Avatar VIDEO (moviepy)
# =====================
class MovieClipPlayer:
    """
    Minimal video player based on moviepy:
    - frame = clip.get_frame(t) (RGB numpy)
    - pygame.image.frombuffer(...) -> Surface
    - loop via (t % duration)
    """
    def __init__(self, path: str):
        self.path = path
        self.clip = VideoFileClip(path, audio=False)
        self.duration = float(self.clip.duration) if self.clip.duration else 0.0
        self.size = tuple(self.clip.size)  # (w,h)

    def get_surface(self, t_active: float, out_size: Tuple[int, int]) -> Optional[pygame.Surface]:
        if self.duration <= 0:
            return None
        tt = (t_active % self.duration)
        frame = self.clip.get_frame(tt)  # numpy (h,w,3) RGB
        surf = pygame.image.frombuffer(frame.tobytes(), self.size, "RGB")
        if out_size is not None:
            surf = pygame.transform.smoothscale(surf, out_size)
        return surf

    def close(self):
        try:
            self.clip.close()
        except Exception:
            pass

    def get_surface_fullscreen(self, t_active: float, screen_size: Tuple[int, int]) -> Optional[pygame.Surface]:
        if self.clip is None or self.duration <= 0:
            return None

        sw, sh = screen_size
        vw, vh = self.size
        if vw <= 0 or vh <= 0 or sw <= 0 or sh <= 0:
            return None

        tt = (t_active % self.duration)
        frame = self.clip.get_frame(tt)

        surf = pygame.image.frombuffer(frame.tobytes(), (vw, vh), "RGB")

        # scale "cover"
        scale = max(sw / vw, sh / vh)
        new_w = int(np.ceil(vw * scale))
        new_h = int(np.ceil(vh * scale))

        new_w = max(new_w, sw)
        new_h = max(new_h, sh)

        surf = pygame.transform.smoothscale(surf, (new_w, new_h))

        canvas = pygame.Surface((sw, sh))
        x = (sw - new_w) // 2
        y = (sh - new_h) // 2
        canvas.blit(surf, (x, y))
        return canvas


class VideoAvatarMoviePy:
    def __init__(self, up_path: str, nomove_path: str, render_size=(520, 520)):
        self.render_size = render_size
        self.up = MovieClipPlayer(up_path)
        self.no = MovieClipPlayer(nomove_path)
        self.t = 0.0
        self.use_up = True

    def update(self, dt, target_up: bool, should_move: bool, animate: bool = True):
        self.use_up = bool(target_up)
        self.t += max(0.0, dt)

    def draw(self, screen, anchor: Tuple[int, int], good: Optional[bool] = None, fullscreen: bool = False):
        x, y = anchor

        if fullscreen:
            surf = (self.up.get_surface_fullscreen(self.t, screen.get_size())
                    if self.use_up else
                    self.no.get_surface_fullscreen(self.t, screen.get_size()))
            if surf is None:
                surf = pygame.Surface(screen.get_size())
                surf.fill((240, 240, 240))
            screen.blit(surf, (0, 0))
        else:
            surf = (self.up.get_surface(self.t, self.render_size)
                    if self.use_up else
                    self.no.get_surface(self.t, self.render_size))

            if surf is None:
                ph = pygame.Surface(self.render_size, pygame.SRCALPHA)
                draw_rounded_rect(ph, ph.get_rect(), (255, 255, 255, 220), radius=28, border=2, border_color=Theme.OUTLINE)
                txt = pygame.font.SysFont(CFG.FONT_NAME, 24, bold=True).render("VIDEO NOT AVAILABLE", True, Theme.INK)
                ph.blit(txt, txt.get_rect(center=(self.render_size[0]//2, self.render_size[1]//2)))
                surf = ph

            rect = surf.get_rect(center=(x, y - 40))
            screen.blit(surf, rect)

    def close(self):
        self.up.close()
        self.no.close()


# =====================
# Rest / ITI visuals
# =====================
def draw_soft_waves(screen, t, base_y, amp=12, spacing=20):
    W, H = screen.get_size()
    for layer in range(3):
        pts = []
        yy = base_y + layer*spacing
        for x in range(0, W+1, 18):
            y = yy + int(np.sin((x*0.012) + (t*1.4) + layer*0.8) * (amp - layer*2))
            pts.append((x, y))
        pygame.draw.lines(screen, (0,0,0), False, pts, width=2)

def draw_calm_bubble_big(screen, font_big, font_small, t, label="CALM"):
    W, H = screen.get_size()
    cx, cy = W//2, H//2 + 20

    r = 200 + int(np.sin(t*2.0)*7)

    bub = pygame.Surface((2*r+120, 2*r+120), pygame.SRCALPHA)
    for k in range(9, 0, -1):
        a = int(16 + k*5)
        pygame.draw.circle(bub, (255,255,255,a), (bub.get_width()//2, bub.get_height()//2), r+10*k)
    pygame.draw.circle(bub, (255,255,255,220), (bub.get_width()//2, bub.get_height()//2), r)
    pygame.draw.circle(bub, (0,0,0,45), (bub.get_width()//2, bub.get_height()//2), r, width=3)
    screen.blit(bub, (cx - bub.get_width()//2, cy - bub.get_height()//2))

    txt = font_big.render(label, True, Theme.INK)
    screen.blit(txt, txt.get_rect(center=(cx, cy-18)))

    tip = font_small.render("Keep still", True, Theme.INK_SOFT)
    screen.blit(tip, tip.get_rect(center=(cx, cy+62)))


# =====================
# Preparation visuals
# =====================
def draw_semaphore(screen, center_xy, active_idx: int, r=20, gap=18):
    """
    active_idx: 0=red, 1=yellow, 2=green
    """
    cx, cy = center_xy
    cols = [Theme.RED, Theme.YELLOW, Theme.GREEN]
    for i in range(3):
        col = cols[i]
        alpha = 255 if i == active_idx else 85
        glow = pygame.Surface((r*6, r*6), pygame.SRCALPHA)
        pygame.draw.circle(glow, (col[0], col[1], col[2], int(0.35*alpha)), (glow.get_width()//2, glow.get_height()//2), r*2)
        screen.blit(glow, (cx + (i-1)*(2*r+gap) - glow.get_width()//2, cy - glow.get_height()//2))

        pygame.draw.circle(screen, col, (cx + (i-1)*(2*r+gap), cy), r)
        pygame.draw.circle(screen, Theme.OUTLINE, (cx + (i-1)*(2*r+gap), cy), r, width=2)

def prep_active_idx(elapsed: float, dur: float) -> int:
    """
    Map elapsed time to semaphore step.
    - first third: red
    - second third: yellow
    - last third: green
    """
    if dur <= 0:
        return 2
    p = clamp(elapsed / dur, 0, 1)
    if p < 1/3: return 0
    if p < 2/3: return 1
    return 2

# =====================
# run_game() — PARTE 2/4
# (setup + scaling helpers + cursor + background system + welcome/mood/bg screens + level intro)
# =====================
def run_game():
    pygame.init()
    pygame.mouse.set_visible(False)

    # --- window / screen ---
    flags = pygame.FULLSCREEN if CFG.FULLSCREEN else 0
    screen = pygame.display.set_mode(CFG.WINDOW_SIZE, flags)
    pygame.display.set_caption("EEG Game — Level 1")
    clock = pygame.time.Clock()

    # --- screen size ---
    W, H = screen.get_size()

    # ---------- UI SCALE ----------
    base_w, base_h = 1400, 900
    ui_scale = min(W / base_w, H / base_h)
    ui_scale = max(0.85, min(ui_scale, 1.6))

    def F(px: int) -> int:
        """Scala un font size in px."""
        return max(14, int(px * ui_scale))

    def S(px: int) -> int:
        """Scala dimensioni/padding (box, margini)."""
        return max(2, int(px * ui_scale))

    # ---------- CURSOR: custom big cursor ----------
    CUR_R = max(10, S(14))   # raggio cerchio
    CUR_O = max(2,  S(4))    # outline thickness

    # Fonts (scalati)
    font_h1   = pygame.font.SysFont(CFG.FONT_NAME, F(65), bold=True)
    font_h2   = pygame.font.SysFont(CFG.FONT_NAME, F(45), bold=True)
    font_h3   = pygame.font.SysFont(CFG.FONT_NAME, F(38), bold=True)
    font_p    = pygame.font.SysFont(CFG.FONT_NAME, F(32))
    font_btn  = pygame.font.SysFont(CFG.FONT_NAME, F(36), bold=True)
    font_pill = pygame.font.SysFont(CFG.FONT_NAME, F(28), bold=True)

    sfx = SFX()
    sfx.load_default_pack()

    ambient = [AmbientParticle(W, H) for _ in range(28)]
    sparkles: List[Sparkle] = []

    state = load_state()
    state["level"] = 1

    def draw_cursor():
        mx, my = pygame.mouse.get_pos()
        pygame.draw.circle(screen, Theme.OUTLINE, (mx, my), CUR_R, width=CUR_O)
        pygame.draw.circle(screen, Theme.WHITE, (mx, my), max(2, CUR_R // 4))

    # -----------------
    # Background system
    # -----------------
    bg_surface = None
    bg_choice_path = state.get("bg", None)

    def tick_ambient(dt):
        W_, H_ = screen.get_size()
        for pt in ambient:
            pt.update(dt, W_, H_)

    def draw_background():
        if bg_surface is None:
            gradient_bg(screen)
            for pt in ambient:
                pt.draw(screen)
        else:
            img = pygame.transform.smoothscale(bg_surface, screen.get_size())
            screen.blit(img, (0, 0))
            dim_overlay(screen, alpha=Theme.DIM_ALPHA)

    def pump_events():
        return pygame.event.get()

    def want_quit(events) -> bool:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            return True
        for ev in events:
            if ev.type == pygame.QUIT:
                return True
        return False

    def set_bg_from_path(path: Optional[str]):
        nonlocal bg_surface
        if not path:
            bg_surface = None
            return
        try:
            bg_surface = pygame.image.load(path).convert()
        except Exception:
            bg_surface = None

    # -----------------
    # VIDEO AVATAR init
    # -----------------
    def _video_path(name: str) -> str:
        return str(VIDEO_DIR / name)

    up_path = _video_path(CFG.VIDEO_UPARM)
    no_path = _video_path(CFG.VIDEO_NOMOVE)

    avatar = VideoAvatarMoviePy(up_path, no_path, render_size=CFG.VIDEO_RENDER_SIZE)

    def safe_close_avatar():
        try:
            avatar.close()
        except Exception:
            pass

    def draw_movie_in_rect(player: MovieClipPlayer, rect: pygame.Rect, t: float):
        surf = player.get_surface(t, out_size=(rect.w, rect.h))
        if surf is None:
            draw_rounded_rect(screen, rect, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)
            txt = pygame.font.SysFont(CFG.FONT_NAME, F(22), bold=True).render("VIDEO NOT AVAILABLE", True, Theme.INK)
            screen.blit(txt, txt.get_rect(center=rect.center))
            return
        screen.blit(surf, rect)

    # -----------------
    # Shared HUD render (responsive)
    # -----------------
    def draw_topbar(title_left: str, subtitle: str = ""):
        pad = S(40)
        bar_h = S(92)
        bar = pygame.Rect(pad, S(28), W - 2*pad, bar_h)

        draw_shadow_rect(screen, bar, radius=Theme.R_CARD, shadow_alpha=40, dy=S(8))
        draw_rounded_rect(screen, bar, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

        pill_w = S(160)
        pill_h = S(46)
        pill(screen, pygame.Rect(bar.x+S(16), bar.y+S(24), pill_w, pill_h), title_left, font_pill,
             fill=Theme.ACCENT, outline=None, text_col=Theme.WHITE)

        if subtitle:
            screen.blit(font_p.render(subtitle, True, Theme.INK_SOFT), (bar.x+S(190), bar.y+S(34)))

    def pause_overlay():
        draw_background()
        dim_overlay(screen, alpha=Theme.PAUSE_ALPHA)
        draw_shadow_text(screen, font_h1, "PAUSED", y=H//2 - S(60), fg=Theme.INK, shadow=(0,0,0))
        draw_center(screen, font_p.render("Press SPACE to resume • ESC to quit", True, Theme.INK), H//2 + S(20))
        draw_cursor()
        pygame.display.flip()

    paused = False

    def timed_phase(duration, render_fn):
        nonlocal paused
        elapsed = 0.0
        last = now_ts()
        while elapsed < duration:
            dt = clock.tick(60) / 1000.0
            events = pump_events()

            if want_quit(events):
                return False

            for ev in events:
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    paused = not paused

            if paused:
                pause_overlay()
                continue

            cur = now_ts()
            elapsed += max(0.0, cur - last)
            last = cur

            render_fn(dt, elapsed, duration)
        return True

    # -----------------
    # helpers: card sizing that "contains" texts
    # -----------------
    def text_block_height(lines: List[str], font: pygame.font.Font, line_gap: int) -> int:
        if not lines:
            return 0
        return len(lines) * font.get_linesize() + (len(lines)-1) * line_gap

    def draw_wrapped_paragraph(center_x: int, y: int, text: str, font: pygame.font.Font,
                              max_w: int, color, line_gap: int):
        lines = wrap_lines(text, font, max_w)
        yy = y
        for ln in lines:
            surf = font.render(ln, True, color)
            screen.blit(surf, surf.get_rect(center=(center_x, yy)))
            yy += font.get_linesize() + line_gap
        return yy  # end y

    # -----------------
    # WELCOME BACK screen
    # -----------------
    def screen_welcome_back() -> bool:
        t = 0.0
        proceed = False

        btn_w, btn_h = S(420), S(86)
        b_continue = Button(
            pygame.Rect(W//2 - btn_w//2, int(H*0.70), btn_w, btn_h),
            "CONTINUE", font_btn, kind="primary"
        )

        while not proceed:
            dt = clock.tick(60) / 1000.0
            t += dt
            events = pump_events()
            if want_quit(events):
                return False
            tick_ambient(dt)

            mx, my = pygame.mouse.get_pos()

            gradient_bg(
                screen,
                top=lerp_col(Theme.GRAD_TOP, Theme.GRAD_BOT, 0.15*np.sin(t*0.7)+0.15),
                bottom=lerp_col(Theme.GRAD_BOT, Theme.GRAD_TOP, 0.15*np.sin(t*0.5)+0.15)
            )
            for pt in ambient:
                pt.draw(screen)

            # --- responsive card
            pad_x = S(90)
            card_w = min(W - 2*pad_x, S(1040))
            card_x = W//2 - card_w//2
            card_y = S(160)
            card_h = S(470)
            card = pygame.Rect(card_x, card_y, card_w, card_h)

            glow_rect(screen, card, color=Theme.ACCENT, strength=110, steps=7, radius=Theme.R_CARD)
            draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=75, dy=S(12))
            draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

            name = CFG.PLAYER_NAME
            draw_shadow_text(screen, font_h1, f"Welcome back, {name}", y=card.y + S(115), fg=Theme.INK, shadow=(0,0,0))
            draw_center(screen, font_h3.render("Level 1 training • EEG + adaptive feedback", True, Theme.INK_SOFT),
                        card.y + S(195))

            cx, cy = W//2, card.y + S(290)
            pulse = S(70) + int(S(10)*np.sin(t*3.0))
            ring = pygame.Surface((S(320), S(320)), pygame.SRCALPHA)
            pygame.draw.circle(ring, (Theme.ACCENT[0], Theme.ACCENT[1], Theme.ACCENT[2], 80),
                               (ring.get_width()//2, ring.get_height()//2), pulse, width=S(10))
            pygame.draw.circle(ring, (255, 255, 255, 110),
                               (ring.get_width()//2, ring.get_height()//2), max(2, pulse-S(18)), width=S(6))
            screen.blit(ring, (cx-ring.get_width()//2, cy-ring.get_height()//2))

            b_continue.tick(dt)
            b_continue.draw(screen, mx, my)

            footer_h = S(54)
            footer = pygame.Rect(S(50), H - S(90), W - S(100), footer_h)
            draw_shadow_rect(screen, footer, radius=Theme.R_CARD, shadow_alpha=45, dy=S(8))
            draw_rounded_rect(screen, footer, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)
            screen.blit(font_p.render("ESC to quit • SPACE pauses during gameplay", True, Theme.INK_SOFT),
                        (footer.x+S(22), footer.y+S(12)))

            draw_cursor()
            pygame.display.flip()

            for ev in events:
                if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    proceed = True
                    sfx.play("select")
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if b_continue.click(mx, my):
                        proceed = True
                        sfx.play("select")

        return True

    # -----------------
    # MOOD screen (cards auto-size to text)
    # -----------------
    def screen_mood() -> Optional[str]:
        mood = None
        while mood is None:
            dt = clock.tick(60) / 1000.0
            events = pump_events()
            if want_quit(events):
                return None
            tick_ambient(dt)

            mx, my = pygame.mouse.get_pos()
            draw_background()
            draw_topbar("WELCOME", "How do you feel today?")

            pad_x = S(90)
            card_w = min(W - 2*pad_x, S(1040))
            card_x = W//2 - card_w//2

            # Compute heights based on wrapped text
            max_text_w = card_w - S(140)
            title_lines = wrap_lines("Choose your mood", font_h2, max_text_w)
            sub_lines = wrap_lines("This only changes break frequency + coaching tone.", font_p, max_text_w)

            gap = S(14)
            top_pad = S(38)
            block_h = (
                text_block_height(title_lines, font_h2, gap) +
                S(18) +
                text_block_height(sub_lines, font_p, S(6))
            )

            bw, bh = S(280), S(78)
            buttons_h = bh
            bottom_pad = S(46)

            card_h = top_pad + block_h + S(40) + buttons_h + bottom_pad
            card = pygame.Rect(card_x, S(170), card_w, card_h)

            draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=55, dy=S(10))
            draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

            y = card.y + top_pad
            # Title
            for ln in title_lines:
                draw_center(screen, font_h2.render(ln, True, Theme.INK), y)
                y += font_h2.get_linesize() + gap

            y += S(10)
            # Subtitle
            for ln in sub_lines:
                draw_center(screen, font_p.render(ln, True, Theme.INK_SOFT), y)
                y += font_p.get_linesize() + S(6)

            # Buttons row
            y_btn = card.bottom - bottom_pad - bh
            b1 = Button(pygame.Rect(W//2 - bw - S(170), y_btn, bw, bh), "tired", font_btn, kind="secondary")
            b2 = Button(pygame.Rect(W//2 - bw//2,     y_btn, bw, bh), "so-and-so", font_btn, kind="secondary")
            b3 = Button(pygame.Rect(W//2 + S(170),   y_btn, bw, bh), "strong", font_btn, kind="secondary")

            for b in (b1, b2, b3):
                b.tick(dt)
                b.draw(screen, mx, my)

            draw_cursor()
            pygame.display.flip()

            for ev in events:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if b1.click(mx, my): mood = "tired"; sfx.play("select")
                    if b2.click(mx, my): mood = "so-and-so"; sfx.play("select")
                    if b3.click(mx, my): mood = "strong"; sfx.play("select")

        return mood

    # -----------------
    # Background mode screen (responsive)
    # -----------------
    def screen_bg_mode() -> Optional[str]:
        choice = None
        while choice is None:
            dt = clock.tick(60) / 1000.0
            events = pump_events()
            if want_quit(events):
                return None
            tick_ambient(dt)

            mx, my = pygame.mouse.get_pos()
            draw_background()
            draw_topbar("BACKGROUND", "Use app default, or pick now?")

            pad_x = S(90)
            card_w = min(W - 2*pad_x, S(1040))
            card_x = W//2 - card_w//2

            max_text_w = card_w - S(140)
            title_lines = wrap_lines("Background settings", font_h2, max_text_w)

            info_1 = "App default = the background saved from previous sessions."
            info_2 = "Choose now = pick an image from assets/backgrounds."
            info_lines = wrap_lines(info_1, font_p, max_text_w) + [""] + wrap_lines(info_2, font_p, max_text_w)

            gap = S(10)
            top_pad = S(38)

            title_h = text_block_height(title_lines, font_h2, S(12))
            info_h = len(info_lines) * font_p.get_linesize() + (len(info_lines)-1) * S(6)

            bw, bh = S(360), S(84)
            buttons_h = bh
            bottom_pad = S(46)

            card_h = top_pad + title_h + S(18) + info_h + S(34) + buttons_h + bottom_pad
            card = pygame.Rect(card_x, S(170), card_w, card_h)

            draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=55, dy=S(10))
            draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

            y = card.y + top_pad
            for ln in title_lines:
                draw_center(screen, font_h2.render(ln, True, Theme.INK), y)
                y += font_h2.get_linesize() + S(12)

            y += S(6)
            for ln in info_lines:
                if ln == "":
                    y += S(10)
                    continue
                draw_center(screen, font_p.render(ln, True, Theme.INK_SOFT), y)
                y += font_p.get_linesize() + S(6)

            y_btn = card.bottom - bottom_pad - bh
            b_app = Button(pygame.Rect(W//2 - bw - S(40), y_btn, bw, bh), "USE APP DEFAULT", font_btn, kind="secondary")
            b_cho = Button(pygame.Rect(W//2 + S(40),      y_btn, bw, bh), "CHOOSE NOW", font_btn, kind="primary")
            for b in (b_app, b_cho):
                b.tick(dt)
                b.draw(screen, mx, my)

            draw_cursor()
            pygame.display.flip()

            for ev in events:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if b_app.click(mx, my):
                        choice = "APP"
                        sfx.play("select")
                    if b_cho.click(mx, my):
                        choice = "CHOOSE"
                        sfx.play("select")
        return choice

    # -----------------
    # Background gallery (layout responsive)
    # -----------------
    def screen_bg_gallery() -> Optional[str]:
        bg_paths = list_backgrounds()
        thumbs = []
        for p in bg_paths[:12]:
            try:
                img = pygame.image.load(str(p)).convert()
                thumbs.append((p, img))
            except Exception:
                pass

        while True:
            dt = clock.tick(60) / 1000.0
            events = pump_events()
            if want_quit(events):
                return None
            tick_ambient(dt)

            mx, my = pygame.mouse.get_pos()
            draw_background()
            draw_topbar("BACKGROUND", "Choose an image (from assets/backgrounds)")

            b_none = Button(pygame.Rect(S(60), H-S(120), S(260), S(78)), "NO IMAGE", font_btn, kind="secondary")
            b_conf = Button(pygame.Rect(W-S(320), H-S(120), S(260), S(78)), "CONFIRM", font_btn, kind="primary")
            b_none.tick(dt); b_conf.tick(dt)
            b_none.draw(screen, mx, my)
            b_conf.draw(screen, mx, my)

            area = pygame.Rect(S(60), S(150), W-S(120), H-S(300))
            draw_shadow_rect(screen, area, radius=Theme.R_CARD, shadow_alpha=35, dy=S(8))
            draw_rounded_rect(screen, area, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

            if not thumbs:
                draw_center(screen, font_p.render("No images found in assets/backgrounds.", True, Theme.INK), area.centery - S(12))
                draw_center(screen, font_p.render("Add .png/.jpg files and retry.", True, Theme.INK_SOFT), area.centery + S(26))
                draw_cursor()
                pygame.display.flip()

                for ev in events:
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        if b_none.click(mx, my):
                            return ""
                continue

            # responsive columns
            cols = 3 if W >= 1100 else 2
            gap = S(22)
            inner = area.inflate(-S(40), -S(40))
            card_w = (inner.w - (cols-1)*gap) // cols
            card_h = int(card_w * 0.58)

            cards = []
            for i, (p, img) in enumerate(thumbs):
                row, col = divmod(i, cols)
                x = inner.x + col*(card_w+gap)
                y = inner.y + row*(card_h+gap)
                rect = pygame.Rect(x, y, card_w, card_h)
                if rect.bottom > inner.bottom:
                    break
                cards.append((rect, p, img))

            for rect, p, img in cards:
                draw_shadow_rect(screen, rect, radius=Theme.R_CARD, shadow_alpha=35, dy=S(8))
                draw_rounded_rect(screen, rect, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)
                thumb = pygame.transform.smoothscale(img, (rect.w-S(10), rect.h-S(10)))
                screen.blit(thumb, (rect.x+S(5), rect.y+S(5)))

                tag = pygame.Rect(rect.x+S(14), rect.y+S(14), min(S(220), rect.w-S(28)), S(34))
                pill(screen, tag, p.stem[:18], font_pill, fill=Theme.CARD, outline=None, text_col=Theme.INK)

                if rect.collidepoint(mx, my):
                    pygame.draw.rect(screen, Theme.ACCENT, rect, width=S(4), border_radius=Theme.R_CARD)

            draw_cursor()
            pygame.display.flip()

            for ev in events:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if b_none.click(mx, my):
                        return ""
                    if b_conf.click(mx, my):
                        return str(bg_choice_path) if bg_choice_path else ""
                    for rect, p, _ in cards:
                        if rect.collidepoint(mx, my):
                            return str(p)

    # -----------------
    # Level intro (responsive + wrapped)
    # -----------------
    def screen_level_intro(mood: str, break_every: int) -> bool:
        proceed = False
        while not proceed:
            dt = clock.tick(60) / 1000.0
            events = pump_events()
            if want_quit(events):
                return False
            tick_ambient(dt)

            mx, my = pygame.mouse.get_pos()
            draw_background()
            draw_topbar("LEVEL 1-1", f"Mood: {mood}  •  Break every {break_every} trials")

            pad_x = S(90)
            card_w = min(W - 2*pad_x, S(1120))
            card_x = W//2 - card_w//2

            max_text_w = card_w - S(160)
            title_lines = wrap_lines("Up Arm vs No Movement", font_h1, max_text_w)

            body = [
                "Two tasks:",
                "• Up Arm: raise your right arm forward to 90° (or imagine it).",
                "• No Movement: stay completely still.",
                "",
                "Press START to watch a short tutorial.",
            ]

            body_lines = []
            for ln in body:
                if ln == "":
                    body_lines.append("")
                else:
                    body_lines += wrap_lines(ln, font_p, max_text_w)

            top_pad = S(40)
            gap_title = S(10)
            gap_body = S(6)

            title_h = text_block_height(title_lines, font_h1, gap_title)
            body_h = 0
            for ln in body_lines:
                if ln == "":
                    body_h += S(10)
                else:
                    body_h += font_p.get_linesize() + gap_body

            btn_w, btn_h = S(400), S(86)
            bottom_pad = S(46)

            card_h = top_pad + title_h + S(24) + body_h + S(36) + btn_h + bottom_pad
            card = pygame.Rect(card_x, S(150), card_w, card_h)

            draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=55, dy=S(10))
            draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

            y = card.y + top_pad
            for ln in title_lines:
                draw_center(screen, font_h1.render(ln, True, Theme.INK), y)
                y += font_h1.get_linesize() + gap_title

            y += S(10)
            for ln in body_lines:
                if ln == "":
                    y += S(10)
                    continue
                draw_center(screen, font_p.render(ln, True, Theme.INK), y)
                y += font_p.get_linesize() + gap_body

            b_start = Button(pygame.Rect(W//2 - btn_w//2, card.bottom - bottom_pad - btn_h, btn_w, btn_h),
                             "START", font_btn, kind="primary")
            b_start.tick(dt)
            b_start.draw(screen, mx, my)

            draw_cursor()
            pygame.display.flip()

            for ev in events:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if b_start.click(mx, my):
                        proceed = True
                        sfx.play("start")

        return True
    
# =====================
# run_game() — PARTE 3/4
# (tutorial video + render calm/prep/move + break screen + predictor + run_block)
# =====================

    # -----------------
    # Tutorial screen (VIDEO panels) — responsive
    # -----------------
    def screen_tutorial() -> bool:
        # carichiamo 2 player separati solo per il tutorial
        try:
            tut_up = MovieClipPlayer(up_path)
        except Exception:
            tut_up = None
        try:
            tut_no = MovieClipPlayer(no_path)
        except Exception:
            tut_no = None

        t = 0.0
        tut_done = False
        while not tut_done:
            dt = clock.tick(60) / 1000.0
            t += dt
            events = pump_events()
            if want_quit(events):
                if tut_up: tut_up.close()
                if tut_no: tut_no.close()
                return False
            tick_ambient(dt)

            for ev in events:
                if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    tut_done = True
                    sfx.play("select")

            draw_background()
            draw_topbar("TUTORIAL", "Press SPACE/ENTER when ready")

            # panels responsive
            panel_w = min(S(520), (W - S(140)) // 2)
            panel_h = panel_w  # quadrati
            top_y = S(190)

            left  = pygame.Rect(W//2 - panel_w - S(40), top_y, panel_w, panel_h)
            right = pygame.Rect(W//2 + S(40),          top_y, panel_w, panel_h)

            for r, title in [(left, "Up Arm"), (right, "No Movement")]:
                draw_shadow_rect(screen, r, radius=Theme.R_CARD, shadow_alpha=45, dy=S(10))
                draw_rounded_rect(screen, r, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)
                tt = font_h2.render(title, True, Theme.INK)
                screen.blit(tt, tt.get_rect(center=(r.centerx, r.y + S(70))))

            # video areas (inside panel, under title)
            vpad = S(110)
            vrectL = pygame.Rect(left.x+S(28),  left.y+vpad,  left.w-S(56),  left.h-vpad-S(28))
            vrectR = pygame.Rect(right.x+S(28), right.y+vpad, right.w-S(56), right.h-vpad-S(28))

            if tut_up:
                draw_movie_in_rect(tut_up, vrectL, t)
            else:
                draw_rounded_rect(screen, vrectL, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

            if tut_no:
                draw_movie_in_rect(tut_no, vrectR, t)
            else:
                draw_rounded_rect(screen, vrectR, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

            footer = pygame.Rect(S(90), H-S(120), W-S(180), S(70))
            draw_shadow_rect(screen, footer, radius=Theme.R_CARD, shadow_alpha=40, dy=S(8))
            draw_rounded_rect(screen, footer, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)
            screen.blit(font_p.render("Each repetition: Rest → Preparation → Movement → ITI", True, Theme.INK),
                        (footer.x+S(22), footer.y+S(22)))

            draw_cursor()
            pygame.display.flip()

        if tut_up: tut_up.close()
        if tut_no: tut_no.close()
        return True

    # -----------------
    # CALM renderer (Rest/ITI) — unchanged visuals, responsive paddings
    # -----------------
    calm_t = 0.0
    def render_calm(tag_text: str, dt):
        nonlocal calm_t
        calm_t += dt
        draw_background()
        dim_overlay(screen, alpha=65)
        draw_soft_waves(screen, calm_t, base_y=H//2 + S(170), amp=S(14), spacing=S(22))
        draw_calm_bubble_big(screen, font_h2, font_p, calm_t, label="CALM")
        draw_topbar("LEVEL 1-1", tag_text)

    # -----------------
    # PREP renderer (semaphore only) — FIXED layout for big fonts
    # -----------------
    def render_prep(block_name: str, trial_i: int, total: int, target: str,
                    el: float, dur: float, mode_badge: str, dt: float):

        # video sotto: sempre NO MOVE durante PREP
        avatar.update(dt, target_up=False, should_move=False, animate=False)
        avatar.draw(screen, (W//2, H//2), good=None, fullscreen=True)

        dim_overlay(screen, alpha=65)

        # --- top bar (responsive)
        pad = S(40)
        bar_h = S(92)
        bar = pygame.Rect(pad, S(28), W-2*pad, bar_h)
        draw_shadow_rect(screen, bar, radius=Theme.R_CARD, shadow_alpha=40, dy=S(8))
        draw_rounded_rect(screen, bar, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

        pill(screen, pygame.Rect(bar.x+S(16),  bar.y+S(24), S(140), S(46)), "LEVEL 1-1", font_pill,
             fill=Theme.ACCENT, outline=None, text_col=Theme.WHITE)
        pill(screen, pygame.Rect(bar.x+S(162), bar.y+S(24), S(210), S(46)), mode_badge, font_pill,
             fill=Theme.CARD, outline=Theme.OUTLINE, text_col=Theme.INK)

        # testo a destra: se stringa lunga, riduci area e lascia comunque leggibile
        right_x = bar.x + S(390)
        screen.blit(
            font_p.render(f"{block_name} • Trial {trial_i}/{total} • PREPARATION", True, Theme.INK),
            (right_x, bar.y+S(28))
        )
        pb = pygame.Rect(right_x, bar.y+S(60), min(S(520), bar.right-right_x-S(30)), S(22))
        progress_bar(screen, pb, (trial_i-1)/max(1, total), fg=Theme.ACCENT)

        # --- cue card: altezza auto in base al font_h2
        cue = "Up Arm soon" if target == "Up Arm" else "Stay still"

        card_w = W - S(160)
        card_x = W//2 - card_w//2
        card_y = bar.bottom + S(35)

        # altezza minima + padding coerente
        card_h = max(S(150), font_h2.get_linesize() + S(90))
        card = pygame.Rect(card_x, card_y, card_w, card_h)

        draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=55, dy=S(10))
        draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

        ct = font_h2.render(cue, True, Theme.INK)
        screen.blit(ct, ct.get_rect(center=card.center))

        # --- semaphore: posizionato sotto la card con spazio fisso, ma clamp per schermi bassi
        idx = prep_active_idx(el, dur)
        sem_y = min(H - S(220), card.bottom + S(120))
        draw_semaphore(screen, (W//2, sem_y), active_idx=idx, r=S(26), gap=S(22))

    # -----------------
    # MOVEMENT HUD renderer (VIDEO) — FIXED layout for big fonts
    # -----------------
    def render_move(block_name: str, trial_i: int, total: int, target_is_up: bool,
                    el: float, dur: float, mode_badge: str, phase_good: Optional[bool]):

        # video sotto
        avatar.draw(screen, (W//2, H//2), good=None, fullscreen=True)
        dim_overlay(screen, alpha=65)

        # --- top bar
        pad = S(40)
        bar_h = S(92)
        bar = pygame.Rect(pad, S(28), W-2*pad, bar_h)
        draw_shadow_rect(screen, bar, radius=Theme.R_CARD, shadow_alpha=40, dy=S(8))
        draw_rounded_rect(screen, bar, Theme.CARD_TINT, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

        pill(screen, pygame.Rect(bar.x+S(16),  bar.y+S(24), S(140), S(46)), "LEVEL 1-1", font_pill,
             fill=Theme.ACCENT, outline=None, text_col=Theme.WHITE)
        pill(screen, pygame.Rect(bar.x+S(162), bar.y+S(24), S(210), S(46)), mode_badge, font_pill,
             fill=Theme.CARD, outline=Theme.OUTLINE, text_col=Theme.INK)

        right_x = bar.x + S(390)
        screen.blit(
            font_p.render(f"{block_name} • Trial {trial_i}/{total} • MOVEMENT", True, Theme.INK),
            (right_x, bar.y+S(28))
        )
        pb = pygame.Rect(right_x, bar.y+S(60), min(S(520), bar.right-right_x-S(30)), S(22))
        progress_bar(screen, pb, (trial_i-1)/max(1, total), fg=Theme.ACCENT)

        # --- cue card autosize
        cue = "MOVE NOW" if target_is_up else "DON'T MOVE"
        card_w = W - S(160)
        card_x = W//2 - card_w//2
        card_y = bar.bottom + S(35)
        card_h = max(S(150), font_h2.get_linesize() + S(90))
        card = pygame.Rect(card_x, card_y, card_w, card_h)

        draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=55, dy=S(10))
        draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

        ct = font_h2.render(cue, True, Theme.INK)
        screen.blit(ct, ct.get_rect(center=card.center))

        # phase progress bar sotto testo, sempre dentro card
        phase_bar = pygame.Rect(card.x+S(30), card.bottom - S(30), card.w-S(60), S(14))
        progress_bar(screen, phase_bar, el/max(1e-6, dur), fg=Theme.ACCENT_2)


    # -----------------
    # Break screen (responsive)
    # -----------------
    def do_break_screen(done, total, mood: str) -> bool:
        nonlocal paused
        paused = True
        while paused:
            dt = clock.tick(60) / 1000.0
            events = pump_events()
            if want_quit(events):
                return False

            for ev in events:
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    paused = False

            draw_background()
            dim_overlay(screen, alpha=Theme.PAUSE_ALPHA)

            card = pygame.Rect(W//2-S(420), S(220), S(840), S(360))
            draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=70, dy=S(12))
            draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

            draw_shadow_text(screen, font_h2, "Take a short break", y=card.y+S(65), fg=Theme.INK, shadow=(0,0,0))
            draw_center(screen, font_p.render(f"Progress: {done}/{total}", True, Theme.INK), card.y+S(130))

            if mood == "tired":
                tip = "Breathe. Relax shoulders. Keep gaze steady."
            elif mood == "so-and-so":
                tip = "Nice pace. Focus on consistency."
            else:
                tip = "Strong energy. Keep movement minimal and clean."
            draw_center(screen, font_p.render(tip, True, Theme.INK_SOFT), card.y+S(175))

            pb = pygame.Rect(card.x+S(70), card.y+S(250), card.w-S(140), S(28))
            progress_bar(screen, pb, done/max(1, total), fg=Theme.ACCENT)

            draw_center(screen, font_p.render("Press SPACE when ready", True, Theme.INK), card.bottom - S(45))
            draw_cursor()
            pygame.display.flip()
        return True

    # -----------------
    # Predictor helpers
    # -----------------
    predictor = Predictor(CFG.PRED_MODE)

    def get_pred_label():
        if CFG.PRED_MODE.upper() == "KEYS":
            return predictor._last_key
        return predictor.get_pred()

    # -----------------
    # EXPERIMENT BLOCKS
    # -----------------
    def run_block(block_name: str, trial_list: List[str], with_feedback: bool, break_every: int, mood: str,
                  outlet: StreamOutlet, writer: csv.writer,
                  event_idx_ref: Dict[str, int],
                  y_true: List[int], y_pred: List[int],
                  fb_stats: Dict[str, int]) -> bool:

        total = len(trial_list)
        mode_badge = "NO FEEDBACK" if not with_feedback else "FEEDBACK ON"

        for i, target in enumerate(trial_list, start=1):

            if i > 1 and (i-1) % break_every == 0:
                if not do_break_screen(done=i-1, total=total, mood=mood):
                    return False

            # -------- REST marker
            event_idx_ref["v"] += 1
            emit_marker(outlet, writer, event_idx_ref["v"], CFG.MARK_REST, i, "REST", target, CFG.REST_SEC)

            def r_rest(dt, el, dur):
                render_calm(f"{block_name} • REST", dt)
                draw_cursor()
                pygame.display.flip()

            if not timed_phase(CFG.REST_SEC, r_rest):
                return False

            # -------- PREP marker
            event_idx_ref["v"] += 1
            emit_marker(outlet, writer, event_idx_ref["v"], CFG.MARK_PREP, i, "PREPARATION", target, CFG.PREP_SEC)

            def r_prep(dt, el, dur):
                # allow KEYS simulation
                if CFG.PRED_MODE.upper() == "KEYS":
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_1] or keys[pygame.K_KP1]:
                        predictor._last_key = 1
                    if keys[pygame.K_0] or keys[pygame.K_KP0]:
                        predictor._last_key = 0

                render_prep(block_name, i, total, target, el, dur, mode_badge, dt)
                draw_cursor()
                pygame.display.flip()

            if not timed_phase(CFG.PREP_SEC, r_prep):
                return False

            # -------- MOVE marker (class code only)
            class_code = CFG.CLASS_CODES[target]
            event_idx_ref["v"] += 1
            emit_marker(outlet, writer, event_idx_ref["v"], class_code, i, "MOVEMENT", target, CFG.MOVE_SEC)

            target_is_up = (target == "Up Arm")
            true_bin = 1 if target_is_up else 0

            phase_pred = None
            phase_good = None
            sparkle_cooldown = 0.0

            def r_move(dt, el, dur):
                nonlocal sparkle_cooldown, phase_pred, phase_good

                for sp in list(sparkles):
                    sp.update(dt)
                    if not sp.alive():
                        sparkles.remove(sp)

                # KEYS sim
                if CFG.PRED_MODE.upper() == "KEYS":
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_1] or keys[pygame.K_KP1]:
                        predictor._last_key = 1
                    if keys[pygame.K_0] or keys[pygame.K_KP0]:
                        predictor._last_key = 0

                p = get_pred_label()
                if p is not None:
                    phase_pred = int(p)

                if with_feedback and phase_pred is not None:
                    phase_good = (phase_pred == true_bin)
                else:
                    phase_good = None


                # update video time + pick clip
                avatar.update(dt, target_up=target_is_up, should_move=target_is_up, animate=True)

                render_move(block_name, i, total, target_is_up, el, dur, mode_badge, phase_good)
                draw_cursor()
                pygame.display.flip()

            if not timed_phase(CFG.MOVE_SEC, r_move):
                return False

            if with_feedback:
                fb_stats["total"] += 1
                if phase_pred is None:
                    y_true.append(true_bin); y_pred.append(-1)
                else:
                    y_true.append(true_bin); y_pred.append(int(phase_pred))
                    if int(phase_pred) == true_bin:
                        fb_stats["correct"] += 1

            iti = CFG.ITI_SEC + random.uniform(0.0, CFG.ITI_JITTER_MAX)

            def r_iti(dt, el, dur):
                render_calm("Inter-trial", dt)
                draw_cursor()
                pygame.display.flip()

            if not timed_phase(iti, r_iti):
                return False

        return True
    
# =====================
# run_game() — PARTE 4/4
# (flow start + block transition + results + run blocks + cleanup)
# =====================

    # -----------------
    # FLOW START
    # -----------------
    try:
        if not screen_welcome_back():
            safe_close_avatar()
            pygame.quit()
            return

        fade_transition(
            screen, clock,
            lambda: (tick_ambient(0), draw_background()),
            duration=0.20, fade_to_white=True
        )

        mood = screen_mood()
        if mood is None:
            safe_close_avatar()
            pygame.quit()
            return
        state["mood"] = mood

        break_every = 10 if mood == "tired" else (20 if mood == "so-and-so" else 30)

        fade_transition(screen, clock, draw_background, duration=0.20, fade_to_white=True)

        bg_mode = screen_bg_mode()
        if bg_mode is None:
            safe_close_avatar()
            pygame.quit()
            return

        if bg_mode == "APP":
            pass
        else:
            chosen = screen_bg_gallery()
            if chosen is None:
                safe_close_avatar()
                pygame.quit()
                return
            if chosen == "":
                state["bg"] = None
                set_bg_from_path(None)
            else:
                state["bg"] = chosen
                set_bg_from_path(chosen)

        save_state(state)
        fade_transition(screen, clock, draw_background, duration=0.20, fade_to_white=True)

        if not screen_level_intro(mood, break_every):
            safe_close_avatar()
            pygame.quit()
            return

        fade_transition(screen, clock, draw_background, duration=0.18, fade_to_white=True)

        if not screen_tutorial():
            safe_close_avatar()
            pygame.quit()
            return

        fade_transition(screen, clock, draw_background, duration=0.18, fade_to_white=True)

        # -----------------
        # Experiment setup
        # -----------------
        nofb_trials = (["Up Arm"] * CFG.N_NOFEEDBACK_PER_CLASS + ["No Movement"] * CFG.N_NOFEEDBACK_PER_CLASS)
        fb_trials   = (["Up Arm"] * CFG.N_FEEDBACK_PER_CLASS   + ["No Movement"] * CFG.N_FEEDBACK_PER_CLASS)
        random.shuffle(nofb_trials)
        random.shuffle(fb_trials)

        ensure_dir(CFG.OUTPUT_DIR)
        base = f"{CFG.SUBJECT}_sess{CFG.SESSION_NUM}_{CFG.SESSION_MODE.lower()}_{time.strftime('%Y%m%d_%H%M%S')}_level1"
        csv_path = os.path.join(CFG.OUTPUT_DIR, base + ".csv")
        csvfile = open(csv_path, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow([
            "event_idx","subject","session_num",
            "trial_idx","phase","class_label","marker_code",
            "lsl_time","mono_time","wall_time",
            "session_mode","experimenter","scheduled_dur"
        ])

        outlet = create_marker_outlet()
        event_idx_ref = {"v": 0}

        # SESSION START marker
        event_idx_ref["v"] += 1
        start_code = CFG.MARK_SESSION_EXECUTED if CFG.SESSION_MODE.upper() == "EXECUTED" else CFG.MARK_SESSION_IMAGINED
        emit_marker(outlet, writer, event_idx_ref["v"], start_code, 0, "SESSION_START", "NA", None)

        # -----------------
        # Helper screens: Block transition + Results
        # -----------------
        def screen_block_transition(title: str, subtitle: str, button_text: str = "START") -> bool:
            t = 0.0
            btn_w, btn_h = S(360), S(86)
            b_go = Button(
                pygame.Rect(W//2 - btn_w//2, int(H*0.72), btn_w, btn_h),
                button_text, font_btn, kind="primary"
            )

            proceed = False
            while not proceed:
                dt = clock.tick(60) / 1000.0
                t += dt
                events = pump_events()
                if want_quit(events):
                    return False
                tick_ambient(dt)

                mx, my = pygame.mouse.get_pos()
                draw_background()

                card = pygame.Rect(W//2-S(520), S(180), S(1040), S(460))
                glow_rect(screen, card, color=Theme.ACCENT_2, strength=95, steps=7, radius=Theme.R_CARD)
                draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=70, dy=S(12))
                draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

                draw_shadow_text(screen, font_h1, title, y=card.y+S(105), fg=Theme.INK, shadow=(0,0,0))
                draw_center(screen, font_h3.render(subtitle, True, Theme.INK_SOFT), card.y+S(180))

                cx, cy = W//2, card.y+S(290)
                pulse = S(78) + int(S(10)*np.sin(t*3.2))
                ring = pygame.Surface((S(360), S(360)), pygame.SRCALPHA)
                pygame.draw.circle(
                    ring, (Theme.ACCENT_2[0], Theme.ACCENT_2[1], Theme.ACCENT_2[2], 75),
                    (ring.get_width()//2, ring.get_height()//2), pulse, width=S(12)
                )
                pygame.draw.circle(
                    ring, (255, 255, 255, 120),
                    (ring.get_width()//2, ring.get_height()//2), max(1, pulse-S(20)), width=S(7)
                )
                screen.blit(ring, (cx-ring.get_width()//2, cy-ring.get_height()//2))

                b_go.tick(dt)
                b_go.draw(screen, mx, my)

                draw_cursor()
                pygame.display.flip()

                for ev in events:
                    if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        proceed = True
                        sfx.play("start")
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        if b_go.click(mx, my):
                            proceed = True
                            sfx.play("start")
            return True

        def screen_results(acc: float, has_feedback: bool, csv_path_str: str) -> str:
            t = 0.0
            stars = star_rating(acc)

            btn_w, btn_h = S(360), S(86)
            b_replay = Button(
                pygame.Rect(W//2 - btn_w - S(20), int(H*0.78), btn_w, btn_h),
                "REPLAY", font_btn, kind="secondary"
            )
            b_exit = Button(
                pygame.Rect(W//2 + S(20), int(H*0.78), btn_w, btn_h),
                "EXIT", font_btn, kind="primary"
            )

            while True:
                dt = clock.tick(60) / 1000.0
                t += dt
                events = pump_events()
                if want_quit(events):
                    return "EXIT"
                tick_ambient(dt)

                mx, my = pygame.mouse.get_pos()
                draw_background()

                card = pygame.Rect(W//2-S(560), S(150), S(1120), S(560))
                glow_rect(
                    screen, card,
                    color=Theme.GOOD if acc >= 0.80 else Theme.ACCENT,
                    strength=120, steps=7, radius=Theme.R_CARD
                )
                draw_shadow_rect(screen, card, radius=Theme.R_CARD, shadow_alpha=75, dy=S(12))
                draw_rounded_rect(screen, card, Theme.CARD, radius=Theme.R_CARD, border=2, border_color=Theme.OUTLINE)

                draw_shadow_text(screen, font_h1, "Session Complete!", y=card.y+S(85), fg=Theme.INK, shadow=(0,0,0))

                acc_txt = f"Accuracy: {acc*100:.1f}%" if has_feedback else "Accuracy: (not computed — feedback off)"
                draw_center(screen, font_h3.render(acc_txt, True, Theme.INK_SOFT), card.y+S(155))

                sx, sy = W//2, card.y+S(260)
                for i in range(3):
                    a = 1.0 if i < stars else 0.25
                    bob = int(np.sin(t*3.0 + i)*S(6))
                    s = pygame.Surface((S(120), S(120)), pygame.SRCALPHA)
                    r1, r2 = S(46), S(20)
                    pts = []
                    for k in range(10):
                        ang = -np.pi/2 + k*(np.pi/5)
                        rr = r1 if k % 2 == 0 else r2
                        pts.append((s.get_width()//2 + int(np.cos(ang)*rr),
                                    s.get_height()//2 + int(np.sin(ang)*rr)))
                    col = Theme.ACCENT if i < stars else Theme.OUTLINE
                    pygame.draw.polygon(s, (col[0], col[1], col[2], int(230*a)), pts)
                    pygame.draw.polygon(s, (0,0,0, int(60*a)), pts, width=max(1, S(3)))
                    screen.blit(s, (sx - S(180) + i*S(180) - s.get_width()//2,
                                    sy - s.get_height()//2 + bob))

                pill_rect = pygame.Rect(W//2 - S(360), card.y+S(370), S(720), S(46))
                pill(screen, pill_rect, f"Saved log: {os.path.basename(csv_path_str)}", font_pill,
                     fill=Theme.CARD_TINT, outline=Theme.OUTLINE, text_col=Theme.INK)

                b_replay.tick(dt); b_exit.tick(dt)
                b_replay.draw(screen, mx, my)
                b_exit.draw(screen, mx, my)

                draw_cursor()
                pygame.display.flip()

                for ev in events:
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_r:
                            return "REPLAY"
                        if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                            return "EXIT"
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        if b_replay.click(mx, my):
                            return "REPLAY"
                        if b_exit.click(mx, my):
                            return "EXIT"

        # -----------------
        # Run blocks
        # -----------------
        y_true: List[int] = []
        y_pred: List[int] = []
        fb_stats = {"correct": 0, "total": 0}

        if not screen_block_transition("BLOCK A", "Calibration • No feedback", button_text="START BLOCK A"):
            csvfile.close()
            safe_close_avatar()
            pygame.quit()
            return
        fade_transition(screen, clock, draw_background, duration=0.18, fade_to_white=True)

        ok = run_block(
            block_name="Block A",
            trial_list=nofb_trials,
            with_feedback=False,
            break_every=break_every,
            mood=mood,
            outlet=outlet,
            writer=writer,
            event_idx_ref=event_idx_ref,
            y_true=y_true,
            y_pred=y_pred,
            fb_stats=fb_stats
        )
        if not ok:
            csvfile.close()
            safe_close_avatar()
            pygame.quit()
            return

        if not screen_block_transition("BLOCK B", "Adaptive feedback ON", button_text="START BLOCK B"):
            csvfile.close()
            safe_close_avatar()
            pygame.quit()
            return
        fade_transition(screen, clock, draw_background, duration=0.18, fade_to_white=True)

        ok = run_block(
            block_name="Block B",
            trial_list=fb_trials,
            with_feedback=True,
            break_every=break_every,
            mood=mood,
            outlet=outlet,
            writer=writer,
            event_idx_ref=event_idx_ref,
            y_true=y_true,
            y_pred=y_pred,
            fb_stats=fb_stats
        )
        if not ok:
            csvfile.close()
            safe_close_avatar()
            pygame.quit()
            return

        # SESSION END marker
        event_idx_ref["v"] += 1
        emit_marker(outlet, writer, event_idx_ref["v"], CFG.MARK_SESSION_END, 0, "SESSION_END", "NA", None)
        csvfile.close()

        # compute accuracy on valid predictions only
        valid = [(t, p) for (t, p) in zip(y_true, y_pred) if p in (0, 1)]
        if len(valid) > 0:
            n_ok = sum(int(t == p) for (t, p) in valid)
            acc = n_ok / len(valid)
        else:
            acc = 0.0

        state["last_acc"] = acc
        if state.get("best_acc") is None or (isinstance(state.get("best_acc"), (int, float)) and acc > state["best_acc"]):
            state["best_acc"] = acc
        save_state(state)

        fade_transition(screen, clock, draw_background, duration=0.18, fade_to_white=True)

        action = screen_results(acc, has_feedback=True, csv_path_str=csv_path)
        if action == "REPLAY":
            safe_close_avatar()
            pygame.quit()
            run_game()
            return

        safe_close_avatar()
        pygame.quit()
        return

    except Exception:
        safe_close_avatar()
        pygame.quit()
        raise
    finally:
        safe_close_avatar()
        try:
            pygame.quit()
        except Exception:
            pass


# =====================
# ENTRY POINT
# =====================
if __name__ == "__main__":
    run_game()