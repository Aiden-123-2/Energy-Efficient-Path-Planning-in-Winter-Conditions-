# Energy-Efficient Path Planning in Winter Conditions

**A Comparative Study of Traditional Baseline, Q-learning, and Proximal Policy Optimization in a Grid World Environment**

Published in *Convergence Journal 2026*.

---

## Overview

This repository contains the full source code for the experiments described in the paper. We compare three routing strategies for energy-efficient navigation in a simulated 50×50 winter grid:

- **BFS** — traditional shortest-path baseline (ignores snow costs)
- **Q-learning** — tabular value-based reinforcement learning
- **PPO (with curriculum learning)** — policy-gradient reinforcement learning

All methods are evaluated under both **deterministic (non-slip)** and **stochastic (slip)** winter conditions across 10 independent random seeds.

---

## Repository Structure

```
├── main.py                        # Entry point — runs full multi-seed experiment
├── requirements.txt               # Python dependencies
│
├── environment/
│   ├── grid_env.py                # 50×50 FrozenLake grid, START/GOAL positions, utilities
│   └── ev_battery_wrapper.py      # Snow cost wrapper and energy-aware reward function
│
├── algorithms/
│   ├── bfs.py                     # BFS shortest-path baseline
│   ├── q_learning.py              # Tabular Q-learning with ε-greedy exploration
│   └── ppo_curriculum.py          # PPO with two-phase curriculum learning
│
├── evaluation/
│   └── evaluator.py               # Episode rollout and multi-episode evaluation functions
│
└── utils/
    ├── plotting.py                # All visualisation functions
    └── statistics.py              # mean_std_ci, Welch's t-test
```

---

## Requirements

- Python 3.10 or above
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Experiment

```bash
python main.py
```

By default this runs 10 seeds under both slip and non-slip conditions.

**Optional environment variables:**

| Variable | Default | Description |
|---|---|---|
| `N_SEEDS` | `10` | Number of independent random seeds |
| `SLIP_MODE` | `both` | `both` / `true` / `false` |

Example — quick test with 1 seed:

```bash
N_SEEDS=1 python main.py
```

---

## Key Outputs

All outputs are saved under `submission_outputs_10seeds/`:

| File | Description |
|---|---|
| `summary_10seeds.csv` | Mean ± std + 95% CI for all methods and conditions |
| `welch_ttest_results.csv` | Welch's t-test: Q-learning vs PPO (slip=TRUE) |
| `learning_curves_10seeds_slipTRUE.png` | Cross-seed learning curves with ±std band |
| `sensitivity/` | Hyperparameter sensitivity bar charts |
| `seed*/` | Per-seed trajectory plots, heatmaps, and ablation charts |

---

## Environment Design

| Parameter | Value |
|---|---|
| Grid size | 50 × 50 |
| Start position | (47, 2) |
| Goal position | (20, 45) |
| Step penalty | −1.5 |
| Near snow penalty | −0.2 |
| Edge snow penalty | −0.5 |
| Core snow penalty | −2.0 |
| Goal reward | +700 |
| Max steps per episode | 1000 |
| Slip probability (stochastic) | 2/3 random direction |

---

## Hyperparameters

### Q-learning

| Parameter | Value |
|---|---|
| Learning rate (α) | 0.10 |
| Discount factor (γ) | 0.99 |
| Initial ε | 1.0 |
| Minimum ε | 0.05 |
| ε decay (per episode) | 0.9995 |
| Training episodes | 15,000 |

### PPO (Curriculum)

| Parameter | Value |
|---|---|
| Learning rate | 3 × 10⁻⁴ |
| Discount factor (γ) | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Rollout buffer | 2048 steps |
| Batch size | 256 |
| Optimisation epochs | 10 |
| Network architecture | [256, 256] ReLU |
| Phase 1 timesteps | 300,000 |
| Phase 2 timesteps | 900,000 |

---

## Citation

If you use this code in your research, please cite:

> Yu, S.-J. (2025). Energy-Efficient Path Planning in Winter Conditions: A Comparative Study of Traditional Baseline, Q-learning, and Proximal Policy Optimization in a Grid World Environment. *Convergence Journal*.
