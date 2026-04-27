# environment/grid_env.py
# Sections 1-3 from winter_rl_10seed_final_v3.py (unchanged)

import numpy as np
import gymnasium as gym
from collections import deque

# =========================================================
# 1) Fixed START / GOAL
# =========================================================
START_POS = (47, 2)
GOAL_POS  = (20, 45)
grid_size = 50

# =========================================================
# 2) Build 50x50 FrozenLake map with fixed S/G
# =========================================================
custom_map_50x50 = ["F" * grid_size for _ in range(grid_size)]

def _set_char(row_str, idx, ch):
    row = list(row_str)
    row[idx] = ch
    return "".join(row)

custom_map_50x50[START_POS[0]] = _set_char(custom_map_50x50[START_POS[0]], START_POS[1], "S")
custom_map_50x50[GOAL_POS[0]]  = _set_char(custom_map_50x50[GOAL_POS[0]],  GOAL_POS[1],  "G")

# =========================================================
# 3) Utilities
# =========================================================
ACTIONS = [0, 1, 2, 3]  # LEFT, DOWN, RIGHT, UP
MOVE = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

def in_bounds(r, c):
    return 0 <= r < grid_size and 0 <= c < grid_size

def state_to_pos(s):
    s = int(s)
    return s // grid_size, s % grid_size

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# =========================================================
# HARD FIX for TimeLimit (Double Guard)
# =========================================================
def force_unwrap_time_limit(env):
    changed = True
    while changed:
        changed = False
        if env.__class__.__name__.lower().endswith("timelimit") and hasattr(env, "env"):
            env = env.env
            changed = True
            continue
        if hasattr(env, "env"):
            inner = env.env
            if env.__class__.__name__.lower().endswith("timelimit"):
                env = inner
                changed = True
                continue
    return env

def force_set_max_steps(env, max_steps=2000):
    e = env
    visited = set()
    while True:
        if id(e) in visited:
            break
        visited.add(id(e))
        if hasattr(e, "_max_episode_steps"):
            try:
                e._max_episode_steps = int(max_steps)
            except Exception:
                pass
        if hasattr(e, "env"):
            e = e.env
        else:
            break
    return env

def make_base_env(seed=0, is_slippery=True, max_steps_guard=2000):
    env = gym.make("FrozenLake-v1", desc=custom_map_50x50, is_slippery=is_slippery)
    env = force_unwrap_time_limit(env)
    env = force_set_max_steps(env, max_steps=max_steps_guard)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
