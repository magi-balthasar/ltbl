#!/usr/bin/env python3
"""
Post-simulation analysis: reads logs/ltbl.db and generates matplotlib figures.

Usage:
    python analysis.py                   # uses logs/ltbl.db, saves analysis.png
    python analysis.py --db my.db        # custom db path
    python analysis.py --show            # display interactively instead of saving
"""

import sys
import os
import argparse
import sqlite3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PALETTE = {
    'asexual':    '#4C9BE8',
    'sexual':     '#E8874C',
    'lamarckian': '#6DBE6D',
    'control':    '#B06DBE',
}

MODE_LABEL = {
    'asexual':    'Asexual',
    'sexual':     'Sexual',
    'lamarckian': 'Lamarckian',
}


def load(db_path: str):
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        print("Run a simulation first:  python main.py --steps 200")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        'SELECT step, island_id, replication_mode, mutation_rate, '
        'population, c_level, c0, c1, c2, avg_generation, max_generation, avg_energy '
        'FROM observations ORDER BY step, island_id'
    ).fetchall()
    events = conn.execute(
        'SELECT step, island_id, event_type FROM emergence_events ORDER BY step'
    ).fetchall()
    conn.close()

    cols = ['step', 'island_id', 'mode', 'mu', 'pop',
            'c_level', 'c0', 'c1', 'c2', 'avg_gen', 'max_gen', 'energy']
    data = {c: [] for c in cols}
    for row in rows:
        for c, v in zip(cols, row):
            data[c].append(v)
    for k in cols:
        data[k] = np.array(data[k]) if k not in ('mode',) else data[k]
    data['mode'] = np.array(data['mode'])
    return data, events


def island_series(data, island_id):
    mask = data['island_id'] == island_id
    return {k: v[mask] if hasattr(v, '__len__') else v
            for k, v in data.items()}


def mode_color(mode: str, mu: float = None) -> str:
    base = PALETTE.get(mode, '#999999')
    return base


def draw(data, events, out_path, show):
    island_ids = sorted(set(data['island_id'].tolist()))
    n_islands = len(island_ids)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0e1117')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_c1   = fig.add_subplot(gs[0, :2])   # C1 over time (wide)
    ax_pop  = fig.add_subplot(gs[1, :2])   # Population dynamics
    ax_gen  = fig.add_subplot(gs[2, :2])   # Max generation over time
    ax_mu   = fig.add_subplot(gs[0, 2])    # Mutation rate vs avg C1
    ax_heat = fig.add_subplot(gs[1, 2])    # C1 heatmap: mode × island
    ax_box  = fig.add_subplot(gs[2, 2])    # Energy distribution

    axes = [ax_c1, ax_pop, ax_gen, ax_mu, ax_heat, ax_box]
    for ax in axes:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')

    # ── Panel 1: C1 over time per island ─────────────────────────────────────
    ax_c1.set_title('C1 — Internally-Driven Behavior Ratio', color='white', fontsize=11)
    ax_c1.set_xlabel('Step', color='#aaaaaa', fontsize=9)
    ax_c1.set_ylabel('C1', color='#aaaaaa', fontsize=9)
    ax_c1.set_ylim(0, 1.05)
    ax_c1.axhline(0.5, color='#ff6b6b', lw=0.8, ls='--', alpha=0.6, label='C1=0.5 threshold')

    mode_seen = set()
    for iid in island_ids:
        s = island_series(data, iid)
        mode = s['mode'][0]
        mu   = float(s['mu'][0])
        color = mode_color(mode)
        alpha = 0.9 if mu == min(s['mu']) else 0.55
        lw = 1.8 if alpha > 0.7 else 1.0
        label = MODE_LABEL.get(mode, mode) if mode not in mode_seen else None
        mode_seen.add(mode)
        ax_c1.plot(s['step'], s['c1'], color=color, lw=lw, alpha=alpha, label=label)

    # Mark emergence events
    for step, iid, etype in events:
        if etype in ('LEVEL_UP', 'C1_THRESHOLD'):
            ax_c1.axvline(step, color='#ffdd57', lw=0.7, alpha=0.5)

    ax_c1.legend(fontsize=8, facecolor='#1a1d27', labelcolor='white',
                 framealpha=0.7, loc='upper left')

    # ── Panel 2: Population dynamics ─────────────────────────────────────────
    ax_pop.set_title('Population Dynamics', color='white', fontsize=11)
    ax_pop.set_xlabel('Step', color='#aaaaaa', fontsize=9)
    ax_pop.set_ylabel('Agents', color='#aaaaaa', fontsize=9)

    for iid in island_ids:
        s = island_series(data, iid)
        mode = s['mode'][0]
        ax_pop.plot(s['step'], s['pop'], color=mode_color(mode), lw=1.2, alpha=0.7)

    # ── Panel 3: Max generation over time ─────────────────────────────────────
    ax_gen.set_title('Max Generation (Evolution Progress)', color='white', fontsize=11)
    ax_gen.set_xlabel('Step', color='#aaaaaa', fontsize=9)
    ax_gen.set_ylabel('Generation', color='#aaaaaa', fontsize=9)

    for iid in island_ids:
        s = island_series(data, iid)
        mode = s['mode'][0]
        ax_gen.plot(s['step'], s['max_gen'], color=mode_color(mode), lw=1.4, alpha=0.8)

    # ── Panel 4: Mutation rate vs avg C1 scatter ───────────────────────────────
    ax_mu.set_title('Mutation Rate vs C1', color='white', fontsize=11)
    ax_mu.set_xlabel('Mutation Rate (μ)', color='#aaaaaa', fontsize=9)
    ax_mu.set_ylabel('Avg C1 (final 20%)', color='#aaaaaa', fontsize=9)

    for iid in island_ids:
        s = island_series(data, iid)
        mode = s['mode'][0]
        mu = float(s['mu'][0])
        tail = int(max(1, len(s['c1']) * 0.2))
        avg_c1 = float(np.mean(s['c1'][-tail:]))
        ax_mu.scatter(mu, avg_c1, color=mode_color(mode), s=90, zorder=5,
                      edgecolors='white', linewidths=0.5)
        ax_mu.annotate(f"{iid}", (mu, avg_c1), textcoords='offset points',
                       xytext=(4, 3), fontsize=7, color='#cccccc')

    # ── Panel 5: C1 heatmap (island × time buckets) ───────────────────────────
    ax_heat.set_title('C1 Heatmap (Island × Time)', color='white', fontsize=11)
    steps_all = sorted(set(data['step'].tolist()))
    n_buckets = min(20, len(steps_all))
    bucket_edges = np.linspace(min(steps_all), max(steps_all), n_buckets + 1)
    heat = np.zeros((n_islands, n_buckets))

    for i, iid in enumerate(island_ids):
        s = island_series(data, iid)
        for b in range(n_buckets):
            lo, hi = bucket_edges[b], bucket_edges[b + 1]
            mask = (s['step'] >= lo) & (s['step'] < hi)
            heat[i, b] = float(np.mean(s['c1'][mask])) if mask.any() else 0.0

    im = ax_heat.imshow(heat, aspect='auto', cmap='plasma', vmin=0, vmax=1,
                        origin='lower')
    ax_heat.set_yticks(range(n_islands))
    ax_heat.set_yticklabels([str(iid) for iid in island_ids], fontsize=7)
    ax_heat.set_xlabel('Time bucket', color='#aaaaaa', fontsize=9)
    ax_heat.set_ylabel('Island ID', color='#aaaaaa', fontsize=9)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04).ax.tick_params(
        colors='#aaaaaa', labelsize=7)

    # ── Panel 6: Final energy distribution ────────────────────────────────────
    ax_box.set_title('Avg Energy by Mode (final 10%)', color='white', fontsize=11)
    ax_box.set_ylabel('Avg Energy', color='#aaaaaa', fontsize=9)

    mode_energies = defaultdict(list)
    for iid in island_ids:
        s = island_series(data, iid)
        mode = s['mode'][0]
        tail = max(1, int(len(s['energy']) * 0.1))
        mode_energies[mode].append(float(np.mean(s['energy'][-tail:])))

    modes_present = list(mode_energies.keys())
    vals = [mode_energies[m] for m in modes_present]
    bp = ax_box.boxplot(vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', lw=2))
    for patch, m in zip(bp['boxes'], modes_present):
        patch.set_facecolor(mode_color(m))
        patch.set_alpha(0.8)
    for element in ('whiskers', 'caps', 'fliers'):
        for line in bp[element]:
            line.set_color('#aaaaaa')
    ax_box.set_xticks(range(1, len(modes_present) + 1))
    ax_box.set_xticklabels([MODE_LABEL.get(m, m) for m in modes_present],
                           fontsize=8, color='#cccccc')

    # ── Legend (mode colors) ──────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=PALETTE[m], lw=2, label=MODE_LABEL[m])
        for m in ('asexual', 'sexual', 'lamarckian') if m in PALETTE
    ]
    legend_handles.append(
        Line2D([0], [0], color='#ffdd57', lw=1.2, ls='--', label='Emergence event')
    )
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               facecolor='#1a1d27', labelcolor='white', fontsize=9,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle('LTBL — Phase 1 Simulation Analysis', color='white',
                 fontsize=15, fontweight='bold', y=0.98)

    if show:
        plt.show()
    else:
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db',   default='logs/ltbl.db')
    p.add_argument('--out',  default='logs/analysis.png')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()

    data, events = load(args.db)
    print(f"Loaded {len(data['step'])} observations, {len(events)} emergence events")
    print(f"Islands: {sorted(set(data['island_id'].tolist()))}")
    print(f"Steps:   {int(data['step'].min())} – {int(data['step'].max())}")
    draw(data, events, args.out, args.show)


if __name__ == '__main__':
    main()
