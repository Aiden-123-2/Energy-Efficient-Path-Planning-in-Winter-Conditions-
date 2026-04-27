# algorithms/q_learning.py
# Section 7 from winter_rl_10seed_final_v3.py (unchanged)

import numpy as np
from collections import deque

# =========================================================
# 7) Q-learning with learning curve logging
# =========================================================
def train_q_learning(env, episodes=15000, max_steps_train=1000, log_every=2000,
                     alpha=0.1, gamma=0.99, eps_decay=0.9995):
    """
    Returns: (Q_table, lc_episodes, lc_rewards)
    lc_episodes / lc_rewards are lists for plotting the learning curve.
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    eps, eps_min = 1.0, 0.05
    recent_steps = deque(maxlen=500)
    recent_success = deque(maxlen=500)

    # Learning curve accumulators (log every `log_every` episodes)
    lc_episodes = []
    lc_rewards  = []
    window_rewards = deque(maxlen=log_every)

    print(f"Q-Learning training started ({episodes} episodes, alpha={alpha}, eps_decay={eps_decay})...")
    for ep in range(1, episodes+1):
        obs, _ = env.reset()
        steps = 0
        terminated = False
        done = False
        ep_reward = 0.0
        while (not done) and steps < max_steps_train:
            s = int(obs)
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s]))
            obs2, r, term, trunc, _ = env.step(a)
            terminated = term
            done = term or trunc
            s2 = int(obs2)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
            obs = obs2
            steps += 1
            ep_reward += r
        recent_steps.append(steps)
        recent_success.append(1 if terminated else 0)
        window_rewards.append(ep_reward)
        eps = max(eps_min, eps * eps_decay)
        if ep % log_every == 0:
            lc_episodes.append(ep)
            lc_rewards.append(float(np.mean(window_rewards)))
            print(f"  [Q] ep={ep:6d} | epsilon={eps:0.3f} | avg_steps={np.mean(recent_steps):0.1f} | success={100*np.mean(recent_success):0.1f}%")
    print("Q-Learning training completed.")
    return Q, lc_episodes, lc_rewards
