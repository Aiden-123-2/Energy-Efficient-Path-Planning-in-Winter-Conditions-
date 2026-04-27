# environment/ev_battery_wrapper.py
# Sections 4-5 from winter_rl_10seed_final_v3.py (unchanged)

import numpy as np
from collections import deque
from gymnasium import Wrapper

from environment.grid_env import (
    grid_size, START_POS, GOAL_POS,
    ACTIONS, MOVE, in_bounds, state_to_pos, manhattan
)

# =========================================================
# 4) Compute snow metrics
# =========================================================
def compute_dist_to_snow(winter_zones):
    INF = 10**9
    dist = [[INF]*grid_size for _ in range(grid_size)]
    q = deque()
    for (r, c) in winter_zones:
        dist[r][c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for a in ACTIONS:
            dr, dc = MOVE[a]
            rr, cc = r+dr, c+dc
            if in_bounds(rr, cc) and dist[rr][cc] > dist[r][c] + 1:
                dist[rr][cc] = dist[r][c] + 1
                q.append((rr, cc))
    return dist

def compute_snow_depth_from_boundary(winter_zones):
    depth = [[-1]*grid_size for _ in range(grid_size)]
    q = deque()
    for (r, c) in winter_zones:
        for a in ACTIONS:
            dr, dc = MOVE[a]
            rr, cc = r+dr, c+dc
            if (not in_bounds(rr, cc)) or ((rr, cc) not in winter_zones):
                depth[r][c] = 0
                q.append((r, c))
                break
    while q:
        r, c = q.popleft()
        for a in ACTIONS:
            dr, dc = MOVE[a]
            rr, cc = r+dr, c+dc
            if in_bounds(rr, cc) and (rr, cc) in winter_zones and depth[rr][cc] == -1:
                depth[rr][cc] = depth[r][c] + 1
                q.append((rr, cc))
    return depth

# =========================================================
# 5) Wrapper: Balanced winter + continuous snow cost
# =========================================================
class EVBatteryWrapper(Wrapper):
    def __init__(
        self,
        env,
        step_penalty=1.5,
        charge_reward=700.0,
        shaping_reward=0.0,
        core_cost=2.0,
        edge_cost=0.5,
        outer_cost=0.2,
        seed_for_snow=42
    ):
        super().__init__(env)
        self.step_penalty = float(step_penalty)
        self.charge_reward = float(charge_reward)
        self.shaping_reward = float(shaping_reward)
        self.core_cost = float(core_cost)
        self.edge_cost = float(edge_cost)
        self.outer_cost = float(outer_cost)
        self.prev_distance = None
        rng = np.random.default_rng(seed_for_snow)
        self.winter_zones = set()
        for r in range(12, 22):
            for c in range(12, 22):
                self.winter_zones.add((r, c))
        for r in range(15, 26):
            for c in range(28, 39):
                self.winter_zones.add((r, c))
        for r in range(28, 39):
            for c in range(14, 25):
                self.winter_zones.add((r, c))
        for r in range(24, 36):
            for c in range(24, 36):
                self.winter_zones.add((r, c))
        for r in range(40, 50):
            for c in range(0, 14):
                if rng.random() < 0.28:
                    self.winter_zones.add((r, c))
        for r in range(4, 18):
            for c in range(36, 49):
                if rng.random() < 0.40:
                    self.winter_zones.add((r, c))
        for r in range(34, 48):
            for c in range(34, 48):
                if rng.random() < 0.65:
                    self.winter_zones.add((r, c))

        def add_cluster(cr, cc, radius=2, p=0.75):
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    rr, cc2 = cr+dr, cc+dc
                    if not in_bounds(rr, cc2):
                        continue
                    dist = abs(dr) + abs(dc)
                    if dist == 0:
                        self.winter_zones.add((rr, cc2))
                    else:
                        prob = p * (0.80 ** max(0, dist - 1))
                        if rng.random() < prob:
                            self.winter_zones.add((rr, cc2))

        def seed_prob(r, c):
            east = c / (grid_size - 1)
            north = 1 - (r / (grid_size - 1))
            return 0.03 + 0.08 * east + 0.05 * north

        placed, tries = 0, 0
        num_seeds = 65
        while placed < num_seeds and tries < num_seeds * 80:
            tries += 1
            r = int(rng.integers(0, grid_size))
            c = int(rng.integers(0, grid_size))
            if manhattan((r, c), START_POS) <= 2:
                continue
            if manhattan((r, c), GOAL_POS) <= 1:
                continue
            if rng.random() < seed_prob(r, c):
                east = c / (grid_size - 1)
                north = 1 - (r / (grid_size - 1))
                radius = 2 if rng.random() < (0.45*east + 0.25*north) else 1
                p = 0.65 + 0.20*east + 0.10*north
                add_cluster(r, c, radius=radius, p=min(p, 0.92))
                placed += 1

        self.dist_to_snow = compute_dist_to_snow(self.winter_zones)
        self.depth_in_snow = compute_snow_depth_from_boundary(self.winter_zones)
        print(f"Balanced Winter Grid loaded: {len(self.winter_zones)} snow cells")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_distance = manhattan(START_POS, GOAL_POS)
        return obs, info

    def _snow_cost(self, r, c):
        if (r, c) in self.winter_zones:
            return self.core_cost if self.depth_in_snow[r][c] >= 2 else self.edge_cost
        return self.outer_cost if self.dist_to_snow[r][c] <= 1 else 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r, c = state_to_pos(obs)
        reward -= self.step_penalty
        reward -= self._snow_cost(r, c)
        if terminated:
            reward += self.charge_reward
        if self.shaping_reward > 0:
            d = manhattan((r, c), GOAL_POS)
            if self.prev_distance is None:
                self.prev_distance = d
            else:
                reward += self.shaping_reward * (self.prev_distance - d)
                self.prev_distance = d
        info["winter_visits"] = 1 if (r, c) in self.winter_zones else 0
        info["snow_visits"] = info["winter_visits"]
        info["is_success"] = bool(terminated) and (not bool(truncated))
        return obs, reward, terminated, truncated, info
