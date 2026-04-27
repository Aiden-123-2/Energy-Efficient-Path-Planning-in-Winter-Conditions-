# algorithms/bfs.py
# Section 6 from winter_rl_10seed_final_v3.py (unchanged)

from collections import deque
from environment.grid_env import (
    START_POS, GOAL_POS, ACTIONS, MOVE, in_bounds, state_to_pos
)

# =========================================================
# 6) BFS shortest baseline
# =========================================================
def bfs_shortest_actions():
    start, goal = START_POS, GOAL_POS
    q = deque([start])
    parent = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for a in ACTIONS:
            dr, dc = MOVE[a]
            rr, cc = r+dr, c+dc
            if not in_bounds(rr, cc):
                continue
            nxt = (rr, cc)
            if nxt not in parent:
                parent[nxt] = (r, c)
                q.append(nxt)
    if goal not in parent:
        return None
    nodes = []
    cur = goal
    while cur is not None:
        nodes.append(cur)
        cur = parent[cur]
    nodes.reverse()
    policy = {}
    for i in range(len(nodes)-1):
        cur = nodes[i]
        nxt = nodes[i+1]
        dr = nxt[0] - cur[0]
        dc = nxt[1] - cur[1]
        for a in ACTIONS:
            if MOVE[a] == (dr, dc):
                policy[cur] = a
                break
    return policy

BFS_POLICY = bfs_shortest_actions()

def bfs_action_from_obs(obs):
    r, c = state_to_pos(obs)
    return BFS_POLICY.get((r, c), 0)
