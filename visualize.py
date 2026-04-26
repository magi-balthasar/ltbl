#!/usr/bin/env python3
"""
Real-time pygame visualizer for Phase 1 single-island simulation.

Usage:
    python visualize.py                     # default settings
    python visualize.py --width 120 --height 120 --agents 60
    python visualize.py --speed 3           # ticks per frame

Controls:
    SPACE  — pause / resume
    +/-    — speed up / slow down
    Q/ESC  — quit
"""

import sys
import os
import argparse
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Layout constants ──────────────────────────────────────────────────────────

SIM_W, SIM_H = 600, 600      # simulation canvas (left)
PANEL_W       = 320           # right stats panel
WIN_W         = SIM_W + PANEL_W
WIN_H         = SIM_H

FPS           = 30
HISTORY_LEN   = 200           # steps kept in sparkline history

# Dark theme colours
BG            = (14,  17,  23)
PANEL_BG      = (20,  24,  36)
TEXT_COL      = (200, 205, 220)
DIM_COL       = (100, 110, 130)
GREEN         = ( 80, 200, 120)
RED           = (220,  80,  80)
YELLOW        = (240, 200,  60)
BLUE          = ( 80, 140, 220)
ORANGE        = (220, 140,  60)


def energy_color(energy: float, threshold: float) -> tuple:
    """Map agent energy to RGB: red (dying) → yellow → green (thriving)."""
    ratio = min(1.0, max(0.0, energy / max(threshold, 0.1)))
    if ratio < 0.5:
        t = ratio * 2
        return (int(220 - 140 * t), int(80 + 120 * t), 80)
    else:
        t = (ratio - 0.5) * 2
        return (int(80 - 0 * t), int(200 - 0 * t), int(80 + 40 * t))


def draw_text(surf, text, x, y, font, color=TEXT_COL, right=False, center=False):
    img = font.render(text, True, color)
    rect = img.get_rect()
    if right:
        rect.right = x
        rect.top = y
    elif center:
        rect.centerx = x
        rect.top = y
    else:
        rect.topleft = (x, y)
    surf.blit(img, rect)
    return rect.bottom


def draw_sparkline(surf, history, x, y, w, h, color, vmin=None, vmax=None):
    if len(history) < 2:
        return
    vals = list(history)
    lo = vmin if vmin is not None else min(vals)
    hi = vmax if vmax is not None else max(vals)
    rang = max(hi - lo, 1e-6)
    pts = []
    for i, v in enumerate(vals):
        px = x + int(i / (len(vals) - 1) * w)
        py = y + h - int((v - lo) / rang * h)
        pts.append((px, py))
    import pygame
    if len(pts) > 1:
        pygame.draw.lines(surf, color, False, pts, 2)


def draw_gauge(surf, label, value, x, y, w, h, color, font_sm, font_lg,
               vmin=0.0, vmax=1.0):
    import pygame
    ratio = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-6)))
    pygame.draw.rect(surf, (40, 44, 60), (x, y, w, h), border_radius=4)
    fill_w = int(w * ratio)
    if fill_w > 0:
        pygame.draw.rect(surf, color, (x, y, fill_w, h), border_radius=4)
    draw_text(surf, label,       x + 6,  y + 2,  font_sm, DIM_COL)
    draw_text(surf, f'{value:.3f}', x + w - 6, y + 2, font_sm, TEXT_COL, right=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('LTBL — Let There Be Light  |  Phase 1')
    clock = pygame.time.Clock()

    font_lg = pygame.font.SysFont('monospace', 15, bold=True)
    font_md = pygame.font.SysFont('monospace', 13)
    font_sm = pygame.font.SysFont('monospace', 11)

    # ── Simulation setup ─────────────────────────────────────────────────────
    from world.primordial_sea import PrimordialSea, SeaConfig
    from agents.membrane import CellAgent
    from genetics.genome import Genome
    from genetics.replication import ReplicationEngine
    from consciousness.level_monitor import ConsciousnessMonitor

    sea_cfg = SeaConfig(width=args.width, height=args.height,
                        vent_count=args.vents, tidal_period=80.0)
    sea = PrimordialSea(sea_cfg)
    replicator = ReplicationEngine(base_mutation_rate=args.mutation)
    monitor = ConsciousnessMonitor()

    agents = [
        CellAgent(
            float(np.random.uniform(5, args.width - 5)),
            float(np.random.uniform(5, args.height - 5)),
            Genome(), mode='asexual',
        )
        for _ in range(args.agents)
    ]

    # Scale: simulation cell → screen pixel
    scale_x = SIM_W / args.width
    scale_y = SIM_H / args.height

    # History buffers for sparklines
    from collections import deque
    hist_pop  = deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN)
    hist_c1   = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
    hist_c2   = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
    hist_e    = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
    hist_gen  = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

    # Simulation state
    step       = 0
    paused     = False
    speed      = args.speed       # ticks per frame
    max_agents = args.max_agents
    metrics    = {'C0': 0.0, 'C1': 0.0, 'C2': 0.0}
    c_level    = 0

    # Nutrient surface (reused each frame)
    nutrient_surf = pygame.Surface((args.width, args.height))
    toxin_surf    = pygame.Surface((args.width, args.height))

    running = True
    while running:
        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    speed = min(speed + 1, 20)
                elif event.key == pygame.K_MINUS:
                    speed = max(speed - 1, 1)

        # ── Simulation ticks ─────────────────────────────────────────────────
        if not paused:
            for _ in range(speed):
                sea.step()
                next_gen = []
                for a in agents:
                    a.step(sea, (args.width, args.height))
                    if not a.alive:
                        continue
                    if a.can_replicate() and len(next_gen) < max_agents:
                        a.pay_replication_cost()
                        child = CellAgent(
                            (a.x + np.random.uniform(-1, 1)) % args.width,
                            (a.y + np.random.uniform(-1, 1)) % args.height,
                            replicator.asexual(a.genome),
                        )
                        next_gen.append(child)
                    next_gen.append(a)
                agents = next_gen
                step += 1

            # Sample metrics every frame (not every tick — cheap enough)
            metrics = monitor.measure(agents)
            c_level = monitor.consciousness_level(metrics)
            pop = len(agents)
            avg_e = float(np.mean([a.state.energy for a in agents])) if agents else 0.0
            avg_g = float(np.mean([a.genome.generation for a in agents])) if agents else 0.0

            hist_pop.append(pop)
            hist_c1.append(metrics['C1'])
            hist_c2.append(metrics['C2'])
            hist_e.append(avg_e)
            hist_gen.append(avg_g)

        # ── Draw simulation canvas (left) ────────────────────────────────────
        screen.fill(BG)

        # Render chemical grids as coloured surfaces
        # Nutrient: green channel  |  Toxin: red channel
        n_arr = sea.nutrient          # (H, W)
        t_arr = sea.toxin

        n_norm = np.clip(n_arr / 10.0, 0, 1)
        t_norm = np.clip(t_arr / 5.0,  0, 1)

        # Build RGB array: nutrient=green, toxin=red, blend
        rgb = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        rgb[:, :, 0] = (t_norm * 180).astype(np.uint8)
        rgb[:, :, 1] = (n_norm * 160).astype(np.uint8)
        rgb[:, :, 2] = ((n_norm * 0.3 + t_norm * 0.1) * 60).astype(np.uint8)
        # Add ambient glow near vents
        for vx, vy, vs in sea.vents:
            y0 = max(0, vy - 3)
            y1 = min(args.height, vy + 4)
            x0 = max(0, vx - 3)
            x1 = min(args.width, vx + 4)
            rgb[y0:y1, x0:x1, 2] = np.clip(
                rgb[y0:y1, x0:x1, 2].astype(int) + 60, 0, 255
            ).astype(np.uint8)

        # pygame.surfarray expects (W, H, 3)
        chem_surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        chem_scaled = pygame.transform.scale(chem_surf, (SIM_W, SIM_H))
        screen.blit(chem_scaled, (0, 0))

        # Draw vent markers
        for vx, vy, vs in sea.vents:
            px = int(vx * scale_x)
            py = int(vy * scale_y)
            pygame.draw.circle(screen, (100, 160, 255), (px, py), 5, 2)

        # Draw agents
        for a in agents:
            px = int(a.x * scale_x)
            py = int(a.y * scale_y)
            col = energy_color(a.state.energy, a.genome.replication_threshold)
            pygame.draw.circle(screen, col, (px, py), 3)

        # Sim canvas border
        pygame.draw.rect(screen, (50, 55, 75), (0, 0, SIM_W, SIM_H), 1)

        # ── Draw stats panel (right) ──────────────────────────────────────────
        panel_x = SIM_W + 10
        panel_rect = pygame.Rect(SIM_W, 0, PANEL_W, WIN_H)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)

        y = 14

        # Title
        y = draw_text(screen, 'LTBL — Phase 1', panel_x, y, font_lg, YELLOW) + 4
        y = draw_text(screen, f'Step {step:>6d}   Speed ×{speed}',
                      panel_x, y, font_sm, DIM_COL) + 12

        # Consciousness level
        c_col = [DIM_COL, GREEN, YELLOW, ORANGE, RED][min(c_level, 4)]
        y = draw_text(screen, f'Consciousness  C{c_level}', panel_x, y, font_md, c_col) + 4

        # C gauges
        bar_w = PANEL_W - 20
        for label, key, col in (('C0  state complexity', 'C0', BLUE),
                                 ('C1  internal drive',   'C1', GREEN),
                                 ('C2  temporal depth',   'C2', ORANGE)):
            val = metrics.get(key, 0.0)
            vmax = 5.0 if key == 'C0' else 1.0
            draw_gauge(screen, label, val, panel_x, y, bar_w, 18, col,
                       font_sm, font_md, vmax=vmax)
            y += 22
        y += 6

        # Population & generation
        pop = int(hist_pop[-1]) if hist_pop else 0
        avg_g_now = float(hist_gen[-1]) if hist_gen else 0.0
        avg_e_now = float(hist_e[-1]) if hist_e else 0.0

        for label, val, unit in (
            ('Population', pop,        ''),
            ('Avg generation', avg_g_now, ''),
            ('Avg energy',     avg_e_now, ''),
        ):
            y = draw_text(screen, f'{label:<16} {val:>7.1f}{unit}',
                          panel_x, y, font_sm, TEXT_COL) + 3
        y += 10

        # Sparklines
        spk_h = 36
        spk_w = bar_w

        sparklines = [
            ('Population',  hist_pop,  BLUE,   None, None),
            ('C1',          hist_c1,   GREEN,  0.0,  1.0),
            ('Avg Energy',  hist_e,    YELLOW, 0.0,  None),
            ('Avg Gen',     hist_gen,  ORANGE, 0.0,  None),
        ]
        for slabel, history, scol, svmin, svmax in sparklines:
            pygame.draw.rect(screen, (28, 32, 48),
                             (panel_x, y, spk_w, spk_h), border_radius=3)
            draw_text(screen, slabel, panel_x + 4, y + 2, font_sm, DIM_COL)
            draw_sparkline(screen, history, panel_x + 2, y + 2,
                           spk_w - 4, spk_h - 4, scol, svmin, svmax)
            y += spk_h + 4
        y += 6

        # Paused indicator
        if paused:
            draw_text(screen, '⏸  PAUSED', panel_x, y, font_md, YELLOW)
            y += 20

        # Controls hint
        hints = ['SPACE pause   +/- speed', 'Q / ESC  quit']
        for h in hints:
            y = draw_text(screen, h, panel_x, y, font_sm, DIM_COL) + 2

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def main():
    p = argparse.ArgumentParser(description='LTBL real-time visualizer')
    p.add_argument('--width',      type=int,   default=100)
    p.add_argument('--height',     type=int,   default=100)
    p.add_argument('--agents',     type=int,   default=60)
    p.add_argument('--max-agents', type=int,   default=400, dest='max_agents')
    p.add_argument('--vents',      type=int,   default=3)
    p.add_argument('--mutation',   type=float, default=0.02)
    p.add_argument('--speed',      type=int,   default=2,
                   help='simulation ticks per rendered frame')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
