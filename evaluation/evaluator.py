# evaluation/evaluator.py
# Section 12 from winter_rl_10seed_final_v3.py (unchanged)

import numpy as np
from environment.grid_env import grid_size, GOAL_POS, state_to_pos
from algorithms.bfs import bfs_action_from_obs

# =========================================================
# 12) Rollout helpers
# =========================================================
def run_episode_trajectory(env, policy, mode, max_steps=1000, deterministic=True, eps_q=0.0):
    obs, _ = env.reset()
    traj = [state_to_pos(obs)]
    done = False
    ep_reward = 0.0
    ep_winter = 0
    ep_steps = 0
    obs_ppo = np.array([int(obs)], dtype=np.int64)

    while (not done) and ep_steps < max_steps:
        if mode == "bfs":
            a = bfs_action_from_obs(obs)
        elif mode == "q":
            s = int(obs)
            if np.random.rand() < eps_q:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(policy[s]))
        elif mode == "ppo":
            act, _ = policy.predict(obs_ppo, deterministic=deterministic)
            a = int(act[0])
        else:
            raise ValueError("Unknown mode")

        obs, r, terminated, truncated, info = env.step(a)
        done = bool(terminated) or bool(truncated)
        traj.append(state_to_pos(obs))
        ep_reward += float(r)
        ep_winter += int(info.get("snow_visits", info.get("winter_visits", 0)))
        ep_steps += 1
        if mode == "ppo":
            obs_ppo = np.array([int(obs)], dtype=np.int64)

    success = (traj[-1] == GOAL_POS)
    return ep_reward, ep_steps, ep_winter, success, traj

def evaluate(env, policy, mode, eval_episodes=200, num_traj=5, max_steps_eval=1000,
             deterministic=True, eps_q=0.0, return_episode_data=False):
    rewards, steps, winters, succs = [], [], [], []
    trajs = []
    visits = np.zeros((grid_size, grid_size), dtype=np.float32)

    for ep in range(int(eval_episodes)):
        r, s, w, success, t = run_episode_trajectory(
            env, policy, mode,
            max_steps=max_steps_eval,
            deterministic=deterministic,
            eps_q=eps_q
        )
        rewards.append(r); steps.append(s); winters.append(w); succs.append(1 if success else 0)
        for (rr, cc) in t:
            if 0 <= rr < grid_size and 0 <= cc < grid_size:
                visits[rr, cc] += 1.0
        if ep < num_traj:
            trajs.append(t)

    v_max = float(visits.max())
    visits_norm = (visits / v_max) if v_max > 0 else visits

    if return_episode_data:
        return (float(np.mean(rewards)), float(np.mean(steps)), float(np.mean(winters)),
                float(np.mean(succs)), trajs, visits_norm, rewards, steps, winters, succs)
    return float(np.mean(rewards)), float(np.mean(steps)), float(np.mean(winters)), float(np.mean(succs)), trajs, visits_norm

def evaluate_ppo_vec(model, vec_env, eval_episodes=200, max_steps_eval=1000,
                     deterministic=True, return_episode_data=False):
    if hasattr(vec_env, "training"):
        vec_env.training = False
    if hasattr(vec_env, "norm_reward"):
        vec_env.norm_reward = False

    rewards, steps, snows, successes = [], [], [], []
    trajs = []
    visits = np.zeros((grid_size, grid_size), dtype=np.float32)

    def _obs_to_state(o):
        try:
            if isinstance(o, (list, tuple)):
                o0 = o[0]
            else:
                o0 = o
            if hasattr(o0, '__len__') and not isinstance(o0, (str, bytes)):
                o0 = o0[0]
            return int(o0)
        except Exception:
            return None

    for ep in range(int(eval_episodes)):
        obs = vec_env.reset()
        ep_r = 0.0
        ep_s = 0
        ep_snow = 0
        traj = []
        s0 = _obs_to_state(obs)
        if s0 is not None:
            traj.append(state_to_pos(s0))
            rr, cc = traj[-1]
            if 0 <= rr < grid_size and 0 <= cc < grid_size:
                visits[rr, cc] += 1.0
        success = False
        done = False

        while (not done) and ep_s < int(max_steps_eval):
            act, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done_arr, infos = vec_env.step(act)
            done = bool(done_arr[0])
            ep_r += float(r[0])
            ep_s += 1
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            if isinstance(info0, dict):
                ep_snow += int(info0.get("snow_visits", info0.get("winter_visits", 0)))
                if bool(info0.get("is_success", False)):
                    success = True
            if done and isinstance(info0, dict) and ("terminal_observation" in info0):
                term_obs = info0.get("terminal_observation")
                try:
                    term_s = int(term_obs) if np.isscalar(term_obs) else None
                except Exception:
                    term_s = None
                if term_s is not None:
                    traj.append(state_to_pos(term_s))
                    rr, cc = traj[-1]
                    if 0 <= rr < grid_size and 0 <= cc < grid_size:
                        visits[rr, cc] += 1.0
                    if state_to_pos(term_s) == GOAL_POS:
                        success = True
                else:
                    st = _obs_to_state(obs)
                    if st is not None:
                        traj.append(state_to_pos(st))
                    if traj:
                        rr, cc = traj[-1]
                        if 0 <= rr < grid_size and 0 <= cc < grid_size:
                            visits[rr, cc] += 1.0
            else:
                st = _obs_to_state(obs)
                if st is not None:
                    traj.append(state_to_pos(st))
                    rr, cc = traj[-1]
                    if 0 <= rr < grid_size and 0 <= cc < grid_size:
                        visits[rr, cc] += 1.0
                    if state_to_pos(st) == GOAL_POS:
                        success = True

        rewards.append(ep_r); steps.append(ep_s); snows.append(ep_snow); successes.append(1 if success else 0)
        if ep < 1:
            trajs.append(traj)

    v_max = float(visits.max())
    visits_norm = (visits / v_max) if v_max > 0 else visits

    if return_episode_data:
        return (float(np.mean(rewards)), float(np.mean(steps)), float(np.mean(snows)),
                float(np.mean(successes)), trajs, visits_norm, rewards, steps, snows, successes)
    return float(np.mean(rewards)), float(np.mean(steps)), float(np.mean(snows)), float(np.mean(successes)), trajs, visits_norm
