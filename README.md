# LTBL — Let There Be Light

> *"Declare the conditions. Let the result emerge."*

**[한국어](README.ko.md) | English**

---

## Two Core Questions

**Q1.** When and under what conditions does consciousness emerge on its own?  
**Q2.** Once emerged, how is consciousness transmitted and developed across generations?

These are not separate problems.  
Consciousness that cannot be transmitted is evolutionarily meaningless.  
The transmission mechanism itself determines the direction in which consciousness evolves.

---

## Core Philosophy

- **Body first, senses second, algorithm last** — intelligence arises from physical feedback loops
- **Nature = the longest simulation** — reproduce evolution in biological order
- **No reward function** — design only pressure and conditions
- **Do not try to create consciousness** — create the conditions under which it can appear
- **Consciousness that cannot be passed on is the same as no consciousness**

---

## 7-Phase Roadmap

| Phase | Model | Consciousness Level | Genetic Mechanism |
|-------|-------|--------------------|--------------------|
| **1** | Single cell (Primordial Sea) | C0–C2 | Self-replication, mutation, selection |
| 2 | Distributed neural net (Hydra, Jellyfish) | C2–C3 | Information stabilization (DNA equivalent) |
| 3 | Arthropod (CPG locomotion) | C3–C4 | Recombination (sexual reproduction equivalent) |
| 4 | Fish (spinal cord–brainstem) | C4 | Structural complexity transfer |
| 5 | Quadruped | C4–C5 | Behavioral pattern + learned transfer |
| 6 | Social learning | C5–C6 | Cultural transmission (beyond genes) |
| 7 | Abstraction | C6 | Information compression via language |

---

## Phase 1: Primordial Sea (Current)

### Environment

- **Fluid**: 2D chemical concentration grid, diffusion-equation based
- **Energy source**: Hydrothermal vents — random position and strength, modulated by tidal effects
- **Hazard**: Locally distributed toxin patches
- **Periodicity**: Tidal cycles, vent pulsation, concentration diffusion

### Agent (CellAgent)

```
[Environmental chemical concentrations]
          │
    8 sensors (weights evolvable)
          │
    Internal state (energy, toxin, history)
          │
    Homeostatic pressure → behavioral urgency
          │
    Actuator (movement direction / speed)
          │
[Consumption / toxin absorption / energy cost]
```

No external reward function — only internal energy homeostatic pressure drives behavior.

### Genetic System

Three replication strategies run in parallel:

| Strategy | Mechanism | Characteristic |
|----------|-----------|---------------|
| **Asexual** | Copy + random mutation | Baseline |
| **Sexual** | Two-parent crossover + mutation | Increased diversity |
| **Lamarckian** | Compress lifetime experience into genome | Inheritance of acquired traits |

The optimal mutation rate is discovered experimentally — too accurate means no evolution; too inaccurate means information collapse.

### Consciousness Level Metrics

| Metric | Description | Phase |
|--------|-------------|-------|
| **C0** | Count and activity of internal state variables | 1-A |
| **C1** | Fraction of behavior driven by internal state | 1-A |
| **C2** | Temporal comparison depth (energy history analysis) | 1-A |
| C3 | Prediction accuracy × prediction time horizon | 1-C |
| C4 | Self-model accuracy | 1-C+ |
| C5 | Other-model accuracy | 1-D |
| C6 | Presence of recursive self-reference | Phase 7 |

---

## Structure

```
ltbl/
├── world/
│   └── primordial_sea.py      # 2D chemical environment, vents, diffusion, tides
├── agents/
│   └── membrane.py            # CellAgent: sense → decide → act, homeostasis
├── genetics/
│   ├── genome.py              # Genome (float vector, 17-dim)
│   ├── replication.py         # Asexual / sexual / Lamarckian replication
│   └── lineage_tracker.py     # Generational lineage tracking
├── consciousness/
│   ├── level_monitor.py       # C0–C6 measurement, discrete level classifier
│   └── emergence_signals.py   # Threshold-crossing event detection
├── evolution/
│   ├── island_model.py        # Island configs (3 modes × mutation rates)
│   └── parallel_engine.py     # Ray-based parallel IslandActor + migration
├── god/
│   ├── observer.py            # SQLite recording + live reports
│   └── intervener.py          # Resource redistribution, cross-pollination
└── main.py
```

---

## Running

### Install

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

### Single island (debug)

```bash
python main.py --single --steps 200 --report 20
```

### Parallel experiment (8 islands via Ray)

```bash
python main.py --steps 500 --report 10 --intervene 50
```

```
┌── Step    10 ─────────────────────────────────────────────────┐
│ [0] asexual      μ=0.005 pop= 312 gen=  4.2/ 7  E= 2.31  C=1 ...
│ [3] sexual       μ=0.010 pop= 289 gen=  3.8/ 6  E= 2.18  C=1 ...
│ [5] lamarckian   μ=0.010 pop= 301 gen=  3.1/ 5  E= 2.44  C=1 ...
└───────────────────────────────────────────────────────────────┘
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 200 | Number of macro steps |
| `--report` | 10 | Report interval |
| `--intervene` | 50 | God intervention interval |
| `--db` | `logs/ltbl.db` | SQLite output path |
| `--single` | off | Single-island debug mode |

All observation data accumulates in `logs/ltbl.db` and can be queried directly:

```sql
SELECT replication_mode, AVG(c1), MAX(max_generation)
FROM observations
GROUP BY replication_mode;
```

---

## Parallel Experiment Design

```
Island 0: asexual    μ=0.005  ─┐
Island 1: asexual    μ=0.020   │
Island 2: asexual    μ=0.080   ├─ Isolated → diversity
Island 3: sexual     μ=0.010   │  Periodic migration → convergence
Island 4: sexual     μ=0.040   │
Island 5: lamarckian μ=0.010   │
Island 6: lamarckian μ=0.040   │
Island 7: high-stress μ=0.030 ─┘  (small world, 1 vent)
```

**God intervention cycle (every N steps):**
1. Scan all consciousness levels
2. Rescue dying islands via resource redistribution
3. Cross-pollinate the two highest-consciousness islands
4. Record full history to SQLite

---

## Phase Transition Criteria

Phase 1 → Phase 2 transition requires:
- Energy homeostasis maintenance rate > 80%
- C1 (internally-driven behavior) > 0.5, sustained
- Consciousness level baseline rising across generations via genetic transfer
- Phase 2 signal detected (collective response patterns)

---

## Tech Stack

| Role | Library |
|------|---------|
| Numerical computation | NumPy, SciPy |
| Parallelism | Ray |
| Data recording | SQLite |
| Visualization (planned) | Pygame, Matplotlib |
| Hardware integration (planned) | ROS2 |

---

*Claude Code is the god of this world. It does not design results. It designs only pressure and conditions. Every failure is data.*
