# utils/plotting.py
# Section 13 from winter_rl_10seed_final_v3.py (unchanged)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.grid_env import grid_size, START_POS, GOAL_POS
from evaluation.evaluator import run_episode_trajectory

# =========================================================
# 13) Plotting helpers
# =========================================================
def preview_winter_map(winter_zones, title="Balanced Winter Grid Preview"):
    winter_array = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in winter_zones:
        winter_array[r, c] = 1
    plt.figure(figsize=(7, 7))
    plt.imshow(winter_array.T, origin="lower", cmap="Blues", alpha=0.85)
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.scatter([START_POS[1]], [START_POS[0]], marker="s", s=90, label="Start (S)")
    plt.scatter([GOAL_POS[1]],  [GOAL_POS[0]],  marker="*", s=140, label="Goal (G)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("winter_map_preview.png", dpi=350, bbox_inches="tight")
    plt.close()

def plot_learning_curves(q_eps, q_rews, ppo_ts, ppo_rews, outfile="learning_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Learning Curves", fontsize=16)
    ax = axes[0]
    ax.plot(q_eps, q_rews, color="steelblue", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Episode Reward (window)")
    ax.set_title("Q-Learning")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    if ppo_ts:
        ax.plot(ppo_ts, ppo_rews, color="darkorange", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("PPO (Curriculum)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves -> {outfile}")

def plot_learning_curves_with_seeds(q_all_eps, q_all_rews, ppo_all_ts, ppo_all_rews,
                                     outfile="learning_curves_seeds.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Learning Curves (Mean ± Std across Seeds)", fontsize=15)

    def _plot_band(ax, xs_list, ys_list, color, title, xlabel):
        min_len = min(len(y) for y in ys_list)
        xs = xs_list[0][:min_len]
        ys = np.array([y[:min_len] for y in ys_list])
        mean = ys.mean(axis=0)
        std  = ys.std(axis=0, ddof=1)
        ax.plot(xs, mean, color=color, linewidth=2, label="Mean")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.25, color=color, label="±1 Std")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_band(axes[0], q_all_eps, q_all_rews, "steelblue", "Q-Learning", "Episode")
    _plot_band(axes[1], ppo_all_ts, ppo_all_rews, "darkorange", "PPO (Curriculum)", "Timestep")
    plt.tight_layout()
    plt.savefig(outfile, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved seed-averaged learning curves -> {outfile}")

def plot_hyperparameter_sensitivity(results_dict, param_name, metric="reward",
                                    outfile="hyperparam_sensitivity.png"):
    params = sorted(results_dict.keys())
    values = [results_dict[p] for p in params]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(p) for p in params], values, color="steelblue", edgecolor="black")
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Hyperparameter Sensitivity: {param_name}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved sensitivity plot -> {outfile}")

def plot_curriculum_ablation(with_cur, without_cur, metrics=("reward", "success_rate"),
                             outfile="curriculum_ablation.png"):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        vals = [with_cur.get(m, 0), without_cur.get(m, 0)]
        labels = ["With Curriculum", "Without Curriculum"]
        bars = ax.bar(labels, vals, color=["steelblue", "salmon"], edgecolor="black")
        ax.set_title(m.replace("_", " ").title())
        ax.set_ylabel(m.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*abs(bar.get_height()),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Curriculum Ablation Study (PPO)", fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved curriculum ablation plot -> {outfile}")

def plot_curriculum_ablation_both_conditions(
    with_cur_false, without_cur_false,
    with_cur_true,  without_cur_true,
    outfile="curriculum_ablation_both.png"
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Curriculum Ablation Study (PPO) — Both Conditions", fontsize=15)
    conditions = [
        ("slipFALSE (Deterministic)", with_cur_false, without_cur_false),
        ("slipTRUE (Stochastic)",     with_cur_true,  without_cur_true),
    ]
    metrics = ["reward", "success_rate"]
    ylabels = ["Cumulative Reward", "Success Rate"]
    for col, (cond_name, with_cur, without_cur) in enumerate(conditions):
        for row, (m, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[row][col]
            vals = [with_cur.get(m, 0), without_cur.get(m, 0)]
            labels = ["With\nCurriculum", "Without\nCurriculum"]
            colors = ["steelblue", "salmon"]
            bars = ax.bar(labels, vals, color=colors, edgecolor="black")
            ax.set_title(f"{cond_name}" if row == 0 else "", fontsize=11)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)
            for bar, v in zip(bars, vals):
                offset = 0.01 * abs(v) if v != 0 else 0.5
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + offset,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved both-condition ablation plot -> {outfile}")

def plot_4way_trajectories(winter_zones, bfs_traj, q_traj, ppo_cur_trajs, ppo_base_traj,
                            outfile="four_way_trajectory.png"):
    winter_array = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in winter_zones:
        winter_array[r, c] = 1
    fig, axes = plt.subplots(1, 4, figsize=(36, 9))
    titles = ["BFS Shortest\n(ignore snow)", "Q-Learning", "PPO Curriculum", "PPO Baseline\n(no curriculum)"]
    trajs  = [
        bfs_traj,
        q_traj,
        ppo_cur_trajs[0] if ppo_cur_trajs else [],
        ppo_base_traj[0] if ppo_base_traj else [],
    ]
    for ax, title, traj in zip(axes, titles, trajs):
        ax.imshow(winter_array.T, origin="lower", cmap="Blues", alpha=0.6)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)
        ax.grid(True, alpha=0.3)
        ax.plot(START_POS[1], START_POS[0], "gs", markersize=10, label="Start (S)")
        ax.plot(GOAL_POS[1],  GOAL_POS[0],  "r*", markersize=14, label="Goal (G)")
        ax.legend(fontsize=9)
        t = np.array(traj) if traj else np.array([])
        if t.size > 0 and len(t.shape) == 2 and t.shape[0] >= 2:
            ax.plot(t[:, 1], t[:, 0], marker="o", linewidth=2, markersize=3, alpha=0.8)
        else:
            ax.text(0.5, 0.5, "No path found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="red")
    fig.suptitle("4-Way Trajectory Comparison (Balanced Winter Grid)", fontsize=18, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outfile, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved 4-way trajectory -> {outfile}")

def plot_state_visitation_heatmaps(winter_zones, q_visits, ppo_visits, slip_flag, outfile):
    winter = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in winter_zones:
        winter[r, c] = 1.0
    winter_plot = np.flipud(winter)
    q_plot = np.flipud(q_visits) if q_visits is not None else None
    ppo_plot = np.flipud(ppo_visits) if ppo_visits is not None else None
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"State Visitation Heatmap | slip={'TRUE' if slip_flag else 'FALSE'}", fontsize=16)

    def _panel(ax, hm, title):
        ax.imshow(winter_plot, cmap="Greys", interpolation="nearest", alpha=0.55, vmin=0, vmax=1)
        im = None
        if hm is not None:
            im = ax.imshow(hm, cmap="hot", interpolation="nearest", alpha=0.90, vmin=0, vmax=1)
        ax.scatter(START_POS[1], grid_size - 1 - START_POS[0], c="cyan", s=90, marker="s")
        ax.scatter(GOAL_POS[1],  grid_size - 1 - GOAL_POS[0],  c="orange", s=140, marker="*")
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        return im

    im1 = _panel(axes[0], q_plot, "Q-learning (Late, greedy)")
    im2 = _panel(axes[1], ppo_plot, "PPO (Late, deterministic)")
    if im1 is not None:
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    if im2 is not None:
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.savefig(outfile, dpi=350, bbox_inches="tight")
    plt.close()

def plot_3way_trajectories(winter_zones, bfs_traj, q_traj, ppo_cur_trajs):
    winter_array = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in winter_zones:
        winter_array[r, c] = 1
    fig, ax = plt.subplots(1, 3, figsize=(27, 9))
    titles = ["BFS Shortest (ignore snow)", "Q-Learning", "PPO Curriculum"]

    def draw_panel(a, title):
        a.imshow(winter_array.T, origin="lower", cmap="Blues", alpha=0.6)
        a.set_title(title, fontsize=16)
        a.set_xlim(0, grid_size); a.set_ylim(0, grid_size)
        a.grid(True, alpha=0.3)
        a.plot(START_POS[1], START_POS[0], "gs", markersize=10, label="Start (S)")
        a.plot(GOAL_POS[1],  GOAL_POS[0],  "r*", markersize=14, label="Goal (G)")
        a.legend()

    draw_panel(ax[0], titles[0])
    t = np.array(bfs_traj)
    if t.size > 0 and len(t.shape) == 2 and t.shape[0] >= 2:
        ax[0].plot(t[:, 1], t[:, 0], marker="o", linewidth=2, markersize=3)

    draw_panel(ax[1], titles[1])
    t = np.array(q_traj)
    if t.size > 0 and len(t.shape) == 2 and t.shape[0] >= 2:
        ax[1].plot(t[:, 1], t[:, 0], marker="o", linewidth=2, markersize=3)

    draw_panel(ax[2], titles[2])
    if ppo_cur_trajs and len(ppo_cur_trajs) > 0:
        t = np.array(ppo_cur_trajs[0])
        if t.size > 0 and len(t.shape) == 2 and t.shape[0] >= 2:
            ax[2].plot(t[:, 1], t[:, 0], marker="o", linewidth=2, markersize=3, alpha=0.8)

    fig.suptitle("3-Way Trajectory Comparison (Balanced Winter Grid)", fontsize=18, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("three_way_trajectory.png", dpi=350, bbox_inches="tight")
    plt.close()

def plot_ppo_pseudo_early_vs_late(eval_env, model, n_traj=5, max_steps=1000):
    winter_zones = eval_env.winter_zones
    winter_array = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in winter_zones:
        winter_array[r, c] = 1
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    titles = ["PPO Curriculum (Stochastic / Early)", "PPO Curriculum (Deterministic / Late)"]
    for ax, det, title in zip(axes, [False, True], titles):
        ax.imshow(winter_array.T, origin="lower", cmap="Blues", alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=16)
        ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)
        ax.plot(START_POS[1], START_POS[0], "gs", markersize=10, label="Start (S)")
        ax.plot(GOAL_POS[1],  GOAL_POS[0],  "r*", markersize=14, label="Goal (G)")
        ax.legend()
        for _ in range(n_traj):
            _, _, _, _, traj = run_episode_trajectory(eval_env, model, mode="ppo", max_steps=max_steps, deterministic=det)
            t = np.array(traj)
            if t.size == 0 or len(t.shape) != 2 or t.shape[0] < 2:
                ax.text(0.5, 0.5, "No trajectory", transform=ax.transAxes, ha="center", va="center")
            else:
                ax.plot(t[:, 1], t[:, 0], marker="o", linewidth=2, markersize=3, alpha=0.7)
    plt.suptitle("PPO Curriculum Trajectories: Pseudo Early vs Late", fontsize=18)
    plt.tight_layout()
    plt.savefig("ppo_pseudo_early_vs_late.png", dpi=350, bbox_inches="tight")
    plt.close()

def plot_q_early_vs_late(eval_env, Q, eps_early=0.5, n_traj=5, max_steps=1000):
    winter_zones = eval_env.winter_zones
    winter_array = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in winter_zones:
        winter_array[r, c] = 1
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    titles = [f"Q (epsilon-greedy / Early, eps={eps_early})", "Q (Greedy / Late, eps=0)"]
    for ax, eps_q, title in zip(axes, [eps_early, 0.0], titles):
        ax.imshow(winter_array.T, origin="lower", cmap="Blues", alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=16)
        ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)
        ax.plot(START_POS[1], START_POS[0], "gs", markersize=10, label="Start (S)")
        ax.plot(GOAL_POS[1],  GOAL_POS[0],  "r*", markersize=14, label="Goal (G)")
        ax.legend()
        for _ in range(n_traj):
            _, _, _, _, traj = run_episode_trajectory(eval_env, Q, mode="q", max_steps=max_steps, eps_q=eps_q)
            t = np.array(traj)
            if t.size == 0 or len(t.shape) != 2 or t.shape[0] < 2:
                ax.text(0.5, 0.5, "No trajectory", transform=ax.transAxes, ha="center", va="center")
            else:
                ax.plot(t[:, 1], t[:, 0], marker="o", linewidth=2, markersize=3, alpha=0.7)
    plt.suptitle("Q-Learning Trajectories: Early vs Late", fontsize=18)
    plt.tight_layout()
    plt.savefig("q_learning_early_vs_late.png", dpi=350, bbox_inches="tight")
    plt.close()
