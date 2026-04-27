# main.py
# Sections 14, 15, 16 from winter_rl_10seed_final_v3.py (unchanged)
#
# Usage:
#   python main.py                    — run 10 seeds, both conditions
#   N_SEEDS=1 python main.py          — quick test with 1 seed
#   SLIP_MODE=true python main.py     — slip condition only

import os
import csv
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment.grid_env import make_base_env
from environment.ev_battery_wrapper import EVBatteryWrapper
from algorithms.bfs import bfs_action_from_obs
from algorithms.q_learning import train_q_learning
from algorithms.ppo_curriculum import (
    train_ppo_curriculum,
    train_ppo_baseline,
    train_ppo_curriculum_lr,
    make_ppo_eval_vecenv_from_trained,
)
from evaluation.evaluator import evaluate, evaluate_ppo_vec
from utils.plotting import (
    preview_winter_map,
    plot_learning_curves,
    plot_learning_curves_with_seeds,
    plot_hyperparameter_sensitivity,
    plot_curriculum_ablation,
    plot_curriculum_ablation_both_conditions,
    plot_3way_trajectories,
    plot_4way_trajectories,
    plot_ppo_pseudo_early_vs_late,
    plot_q_early_vs_late,
    plot_state_visitation_heatmaps,
)
from utils.statistics import mean_std_ci, welch_t_test

# =========================================================
# 14) Appendix CSV helpers
# =========================================================
def _ensure_appendix_dir(dir_name="Appendix"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def _save_episode_csv(rows, out_csv):
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def _save_matrix_csv(matrix, out_csv):
    np.savetxt(out_csv, matrix, delimiter=",", fmt="%.6f")

# =========================================================
# 15) Per-condition experiment runner
# =========================================================
def run_submission_condition(is_slippery: bool, out_dir: str,
                             run_baseline_ppo: bool = True,
                             seed_override: int = 888):
    os.makedirs(out_dir, exist_ok=True)
    _cwd = os.getcwd()
    try:
        os.chdir(out_dir)

        is_slippery = bool(is_slippery)
        PPO_STEPS   = 1_200_000
        PPO_PHASE1  = 300_000
        PPO_PHASE2  = PPO_STEPS - PPO_PHASE1
        Q_EPISODES  = 15_000
        EVAL_EPISODES     = 200
        HEATMAP_AVG_RUNS  = 10
        MAX_STEPS_EVAL    = 1000
        SEED_TRAIN = int(seed_override)
        SEED_EVAL  = int(seed_override) + 1000
        EXPORT_APPENDIX = True

        # Fix NumPy randomness for full reproducibility (affects Q-learning epsilon-greedy)
        np.random.seed(SEED_TRAIN)

        print(f"Starting | is_slippery={is_slippery} | seed={seed_override}")

        train_base = make_base_env(seed=SEED_TRAIN, is_slippery=is_slippery, max_steps_guard=2000)
        eval_base  = make_base_env(seed=SEED_EVAL,  is_slippery=is_slippery, max_steps_guard=2000)
        train_env  = EVBatteryWrapper(train_base, shaping_reward=0.8, seed_for_snow=42)
        eval_env   = EVBatteryWrapper(eval_base,  shaping_reward=0.0, seed_for_snow=42)

        preview_winter_map(eval_env.winter_zones, "Balanced Winter Grid (Fixed S/G)")

        # --- BFS ---
        bfs_reward, bfs_steps, bfs_winter, bfs_succ, bfs_trajs, bfs_vis, \
            bfs_ep_rewards, bfs_ep_steps, bfs_ep_winters, bfs_ep_succs = evaluate(
                eval_env, policy=None, mode="bfs",
                eval_episodes=1, num_traj=1, max_steps_eval=MAX_STEPS_EVAL,
                return_episode_data=True
            )
        print(f"BFS: R={bfs_reward:.2f} S={bfs_steps:.2f} Snow={bfs_winter:.2f} Succ={100*bfs_succ:.1f}%")

        # --- Q-learning ---
        Q, q_lc_eps, q_lc_rews = train_q_learning(train_env, episodes=Q_EPISODES, max_steps_train=2000)
        q_reward, q_steps, q_winter, q_succ, q_trajs, q_vis, \
            q_ep_rewards, q_ep_steps, q_ep_winters, q_ep_succs = evaluate(
                eval_env, policy=Q, mode="q",
                eval_episodes=EVAL_EPISODES, num_traj=1, max_steps_eval=MAX_STEPS_EVAL,
                return_episode_data=True
            )
        print(f"Q-Learning: R={q_reward:.2f} S={q_steps:.2f} Snow={q_winter:.2f} Succ={100*q_succ:.1f}%")

        # --- PPO Curriculum ---
        model, ppo_vec_env, ppo_lc_ts, ppo_lc_rews = train_ppo_curriculum(
            is_slippery=is_slippery,
            phase1_steps=PPO_PHASE1,
            phase2_steps=PPO_PHASE2,
            seed_train=SEED_TRAIN,
            checkpoint_dir="ppo_checkpoints"
        )
        cur_eval_vec = make_ppo_eval_vecenv_from_trained(ppo_vec_env, seed_eval=SEED_EVAL, is_slippery=is_slippery)
        ppo_cur_reward, ppo_cur_steps, ppo_cur_winter, ppo_cur_succ, ppo_cur_traj, ppo_cur_vis, \
            ppo_cur_ep_rewards, ppo_cur_ep_steps, ppo_cur_ep_snows, ppo_cur_ep_succs = evaluate_ppo_vec(
                model, cur_eval_vec, eval_episodes=EVAL_EPISODES, max_steps_eval=MAX_STEPS_EVAL,
                return_episode_data=True
            )
        print(f"PPO Cur: R={ppo_cur_reward:.2f} S={ppo_cur_steps:.2f} Snow={ppo_cur_winter:.2f} Succ={100*ppo_cur_succ:.1f}%")

        # --- PPO Baseline (no curriculum) --- for ablation
        ppo_base_reward = ppo_base_steps = ppo_base_winter = ppo_base_succ = None
        ppo_base_lc_ts = ppo_base_lc_rews = []
        ppo_base_traj = []
        if run_baseline_ppo:
            base_model, base_vec_env, ppo_base_lc_ts, ppo_base_lc_rews = train_ppo_baseline(
                is_slippery=is_slippery,
                total_steps=PPO_STEPS,
                seed_train=SEED_TRAIN,
                checkpoint_dir="ppo_baseline_checkpoints"
            )
            base_eval_vec = make_ppo_eval_vecenv_from_trained(base_vec_env, seed_eval=SEED_EVAL, is_slippery=is_slippery)
            ppo_base_reward, ppo_base_steps, ppo_base_winter, ppo_base_succ, ppo_base_traj, _, \
                ppo_base_ep_rewards, ppo_base_ep_steps, ppo_base_ep_snows, ppo_base_ep_succs = evaluate_ppo_vec(
                    base_model, base_eval_vec, eval_episodes=EVAL_EPISODES, max_steps_eval=MAX_STEPS_EVAL,
                    return_episode_data=True
                )
            print(f"PPO Base: R={ppo_base_reward:.2f} S={ppo_base_steps:.2f} Snow={ppo_base_winter:.2f} Succ={100*ppo_base_succ:.1f}%")

        # --- Learning curves ---
        try:
            plot_learning_curves(q_lc_eps, q_lc_rews, ppo_lc_ts, ppo_lc_rews,
                                 outfile="learning_curves.png")
        except Exception as e:
            print(f"[WARN] Learning curve plot failed: {e}")

        # --- Curriculum ablation plot ---
        if run_baseline_ppo and ppo_base_reward is not None:
            try:
                plot_curriculum_ablation(
                    with_cur={"reward": ppo_cur_reward, "success_rate": ppo_cur_succ},
                    without_cur={"reward": ppo_base_reward, "success_rate": ppo_base_succ},
                    outfile="curriculum_ablation.png"
                )
            except Exception as e:
                print(f"[WARN] Curriculum ablation plot failed: {e}")

        # --- Trajectory plots ---
        try:
            bfs_path = bfs_trajs[0] if bfs_trajs else []
            q_path   = q_trajs[0]   if q_trajs   else []
            plot_3way_trajectories(eval_env.winter_zones, bfs_path, q_path, ppo_cur_traj)
            if run_baseline_ppo and ppo_base_reward is not None:
                ppo_base_path = ppo_base_traj[0] if ppo_base_traj and len(ppo_base_traj) > 0 else []
                plot_4way_trajectories(
                    eval_env.winter_zones,
                    bfs_path, q_path,
                    ppo_cur_traj,
                    [ppo_base_path],
                    outfile="four_way_trajectory.png"
                )
            plot_ppo_pseudo_early_vs_late(eval_env, model, n_traj=5, max_steps=MAX_STEPS_EVAL)
            plot_q_early_vs_late(eval_env, Q, eps_early=0.5, n_traj=5, max_steps=MAX_STEPS_EVAL)
        except Exception as e:
            print(f"[WARN] Trajectory plots failed: {e}")

        # --- Appendix export ---
        if EXPORT_APPENDIX:
            try:
                app_dir = _ensure_appendix_dir("Appendix")
                condition = f"slip{'TRUE' if is_slippery else 'FALSE'}"
                rows = []

                def _append_rows(method_name, ep_rewards, ep_steps, ep_snows, ep_succs):
                    if ep_rewards is None:
                        return
                    for i in range(len(ep_rewards)):
                        rows.append({
                            "condition": condition,
                            "method": method_name,
                            "episode": i,
                            "reward": float(ep_rewards[i]),
                            "steps": float(ep_steps[i]),
                            "snow_visits": float(ep_snows[i]),
                            "success": int(ep_succs[i]),
                        })

                _append_rows("BFS", bfs_ep_rewards, bfs_ep_steps, bfs_ep_winters, bfs_ep_succs)
                _append_rows("Q-learning", q_ep_rewards, q_ep_steps, q_ep_winters, q_ep_succs)
                if run_baseline_ppo and ppo_base_reward is not None:
                    _append_rows("PPO Baseline", ppo_base_ep_rewards, ppo_base_ep_steps,
                                 ppo_base_ep_snows, ppo_base_ep_succs)
                _append_rows("PPO Curriculum", ppo_cur_ep_rewards, ppo_cur_ep_steps,
                             ppo_cur_ep_snows, ppo_cur_ep_succs)

                _save_episode_csv(rows, os.path.join(app_dir, f"Appendix_A_EpisodeMetrics_{condition}.csv"))

                from environment.grid_env import grid_size
                winter_mask = np.zeros((grid_size, grid_size), dtype=int)
                for r, c in eval_env.winter_zones:
                    winter_mask[r, c] = 1
                _save_matrix_csv(winter_mask, os.path.join(app_dir, f"Appendix_B_WinterMask_{condition}.csv"))
                print(f"[Appendix] Saved for {condition}")
            except Exception as e:
                print(f"[WARN] Appendix export failed: {e}")

        # --- Heatmaps ---
        try:
            from environment.grid_env import grid_size
            q_vis_acc   = np.zeros((grid_size, grid_size), dtype=np.float32)
            ppo_vis_acc = np.zeros((grid_size, grid_size), dtype=np.float32)
            for run_i in range(HEATMAP_AVG_RUNS):
                np.random.seed(int(SEED_EVAL) + run_i)
                _, _, _, _, _, q_vis_i = evaluate(
                    eval_env, policy=Q, mode="q",
                    eval_episodes=EVAL_EPISODES, num_traj=0,
                    max_steps_eval=MAX_STEPS_EVAL, deterministic=True
                )
                q_vis_acc += q_vis_i
                cur_eval_vec_i = make_ppo_eval_vecenv_from_trained(
                    ppo_vec_env, seed_eval=int(SEED_EVAL)+run_i, is_slippery=is_slippery)
                _, _, _, _, _, ppo_vis_i = evaluate_ppo_vec(
                    model, cur_eval_vec_i, eval_episodes=EVAL_EPISODES, max_steps_eval=MAX_STEPS_EVAL, deterministic=True)
                ppo_vis_acc += ppo_vis_i
            q_vis_avg   = q_vis_acc / HEATMAP_AVG_RUNS
            ppo_vis_avg = ppo_vis_acc / HEATMAP_AVG_RUNS
            q_vis_avg   = q_vis_avg   / q_vis_avg.max()   if q_vis_avg.max()   > 0 else q_vis_avg
            ppo_vis_avg = ppo_vis_avg / ppo_vis_avg.max() if ppo_vis_avg.max() > 0 else ppo_vis_avg
            plot_state_visitation_heatmaps(
                eval_env.winter_zones, q_vis_avg, ppo_vis_avg,
                slip_flag=is_slippery,
                outfile=f"state_visitation_Q_vs_PPO_slip{'TRUE' if is_slippery else 'FALSE'}.png"
            )
        except Exception as e:
            print(f"[WARN] Heatmap failed: {e}")

        print("\n=== Final Comparison Table ===")
        print(f"{'Method':<16}{'Reward':>12}{'Steps':>10}{'Snow Visits':>12}{'Success':>10}")
        print(f"{'BFS Shortest':<16}{bfs_reward:12.2f}{bfs_steps:10.2f}{bfs_winter:12.2f}{100*bfs_succ:10.1f}%")
        print(f"{'Q-Learning':<16}{q_reward:12.2f}{q_steps:10.2f}{q_winter:12.2f}{100*q_succ:10.1f}%")
        if run_baseline_ppo and ppo_base_reward is not None:
            print(f"{'PPO Baseline':<16}{ppo_base_reward:12.2f}{ppo_base_steps:10.2f}{ppo_base_winter:12.2f}{100*ppo_base_succ:10.1f}%")
        print(f"{'PPO Curriculum':<16}{ppo_cur_reward:12.2f}{ppo_cur_steps:10.2f}{ppo_cur_winter:12.2f}{100*ppo_cur_succ:10.1f}%")

        return {
            "bfs":     (float(bfs_reward), float(bfs_steps), float(bfs_winter), float(bfs_succ)),
            "q":       (float(q_reward),   float(q_steps),   float(q_winter),   float(q_succ)),
            "ppo_cur": (float(ppo_cur_reward), float(ppo_cur_steps), float(ppo_cur_winter), float(ppo_cur_succ)),
            "ppo_base": None if ppo_base_reward is None else
                        (float(ppo_base_reward), float(ppo_base_steps), float(ppo_base_winter), float(ppo_base_succ)),
            "q_lc_eps":       q_lc_eps,
            "q_lc_rews":      q_lc_rews,
            "ppo_lc_ts":      ppo_lc_ts,
            "ppo_lc_rews":    ppo_lc_rews,
            "q_ep_succs":       q_ep_succs,
            "ppo_cur_ep_succs": ppo_cur_ep_succs,
            "q_ep_rewards":       q_ep_rewards,
            "ppo_cur_ep_rewards": ppo_cur_ep_rewards,
        }
    finally:
        os.chdir(_cwd)

# =========================================================
# 16) MAIN: multi-seed loop + statistical summary
# =========================================================
if __name__ == "__main__":

    OUTPUT_ROOT = "submission_outputs_10seeds"

    try:
        N_SEEDS = int(os.environ.get("N_SEEDS", "10"))
    except Exception:
        N_SEEDS = 10

    slip_mode = os.environ.get("SLIP_MODE", "both").strip().lower()
    if slip_mode in ("false", "0", "no", "n"):
        conditions = [(False, os.path.join(OUTPUT_ROOT, "slipFALSE"))]
    elif slip_mode in ("true", "1", "yes", "y"):
        conditions = [(True, os.path.join(OUTPUT_ROOT, "slipTRUE"))]
    else:
        conditions = [
            (False, os.path.join(OUTPUT_ROOT, "slipFALSE")),
            (True,  os.path.join(OUTPUT_ROOT, "slipTRUE")),
        ]

    all_results = []

    for seed in range(N_SEEDS):
        print(f"\n{'='*50}")
        print(f" RUNNING SEED {seed} / {N_SEEDS - 1}")
        print(f"{'='*50}\n")
        for slip, out_dir in conditions:
            metrics = run_submission_condition(
                is_slippery=slip,
                out_dir=os.path.join(out_dir, f"seed{seed}"),
                run_baseline_ppo=True,
                seed_override=seed,
            )
            all_results.append({"seed": int(seed), "slip": bool(slip), **metrics})

    # ── Cross-seed aggregate ──────────────────────────────────────────────────
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_rows = []

    print(f"\n{'='*60}")
    print(f" FINAL RESULTS ({N_SEEDS} seeds, Mean ± Std [95% CI])")
    print(f"{'='*60}")

    for slip in sorted(set(r["slip"] for r in all_results)):
        subset = [r for r in all_results if r["slip"] == slip]
        cond_name = "slipTRUE" if slip else "slipFALSE"
        print(f"\n=== {cond_name} ===")

        for method_key, label in [("bfs", "BFS Shortest"), ("q", "Q-Learning"),
                                   ("ppo_cur", "PPO Curriculum"), ("ppo_base", "PPO No-Curriculum")]:
            vals = [r[method_key] for r in subset if r.get(method_key) is not None]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)

            rew_arr  = arr[:, 0]
            step_arr = arr[:, 1]
            snow_arr = arr[:, 2]
            succ_arr = arr[:, 3]

            r_m,  r_s,  r_lo,  r_hi  = mean_std_ci(rew_arr)
            st_m, st_s, _,     _     = mean_std_ci(step_arr)
            sn_m, sn_s, _,     _     = mean_std_ci(snow_arr)
            s_m,  s_s,  s_lo,  s_hi  = mean_std_ci(succ_arr * 100)

            print(
                f"{label:<16}"
                f"  Reward: {r_m:8.2f}±{r_s:6.2f} [{r_lo:7.2f},{r_hi:7.2f}]"
                f"  Steps: {st_m:7.2f}±{st_s:5.2f}"
                f"  Snow: {sn_m:7.2f}±{sn_s:5.2f}"
                f"  Success: {s_m:6.1f}%±{s_s:4.1f}% [{s_lo:6.1f},{s_hi:6.1f}]"
            )

            summary_rows.append({
                "condition":      cond_name,
                "method":         label,
                "n_seeds":        N_SEEDS,
                "reward_mean":    float(r_m),
                "reward_std":     float(r_s),
                "reward_ci_lo":   float(r_lo),
                "reward_ci_hi":   float(r_hi),
                "steps_mean":     float(st_m),
                "steps_std":      float(st_s),
                "snow_mean":      float(sn_m),
                "snow_std":       float(sn_s),
                "success_mean":   float(s_m / 100),
                "success_std":    float(s_s / 100),
                "success_ci_lo":  float(s_lo / 100),
                "success_ci_hi":  float(s_hi / 100),
            })

        # --- Welch's t-test (slip condition) ---
        if slip:
            print(f"\n--- Welch's t-test (Q-learning vs PPO Curriculum, slip=TRUE) ---")
            q_succs   = np.array([r["q"][3]       for r in subset], dtype=float)
            ppo_succs = np.array([r["ppo_cur"][3] for r in subset], dtype=float)
            q_rews    = np.array([r["q"][0]       for r in subset], dtype=float)
            ppo_rews  = np.array([r["ppo_cur"][0] for r in subset], dtype=float)

            t_succ, p_succ = welch_t_test(q_succs, ppo_succs)
            t_rew,  p_rew  = welch_t_test(q_rews,  ppo_rews)

            print(f"  Success rate : t={t_succ:.4f}, p={p_succ:.4f} {'*** p<0.01' if p_succ < 0.01 else '** p<0.05' if p_succ < 0.05 else 'n.s.'}")
            print(f"  Reward       : t={t_rew:.4f},  p={p_rew:.4f}  {'*** p<0.01' if p_rew  < 0.01 else '** p<0.05' if p_rew  < 0.05 else 'n.s.'}")

            ttest_path = os.path.join(OUTPUT_ROOT, "welch_ttest_results.csv")
            with open(ttest_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["metric", "t_stat", "p_value", "significance",
                                                   "Q_mean", "PPO_mean", "Q_std", "PPO_std"])
                w.writeheader()
                for metric, t, p, qa, pa, qs, ps in [
                    ("reward",       t_rew,  p_rew,  q_rews.mean(),  ppo_rews.mean(),
                     q_rews.std(ddof=1),  ppo_rews.std(ddof=1)),
                    ("success_rate", t_succ, p_succ, q_succs.mean(), ppo_succs.mean(),
                     q_succs.std(ddof=1), ppo_succs.std(ddof=1)),
                ]:
                    w.writerow({
                        "metric": metric, "t_stat": round(t, 6), "p_value": round(p, 6),
                        "significance": "p<0.01" if p < 0.01 else "p<0.05" if p < 0.05 else "n.s.",
                        "Q_mean": round(qa, 6), "PPO_mean": round(pa, 6),
                        "Q_std":  round(qs, 6), "PPO_std":  round(ps, 6),
                    })
            print(f"  Saved t-test results -> {ttest_path}")

    # --- Cross-seed learning curve plot ---
    try:
        slip_subset  = [r for r in all_results if r["slip"] is True]
        q_all_eps    = [r["q_lc_eps"]   for r in slip_subset if r.get("q_lc_eps")]
        q_all_rews   = [r["q_lc_rews"]  for r in slip_subset if r.get("q_lc_rews")]
        ppo_all_ts   = [r["ppo_lc_ts"]  for r in slip_subset if r.get("ppo_lc_ts")]
        ppo_all_rews = [r["ppo_lc_rews"] for r in slip_subset if r.get("ppo_lc_rews")]
        if q_all_rews and ppo_all_rews:
            plot_learning_curves_with_seeds(
                q_all_eps, q_all_rews, ppo_all_ts, ppo_all_rews,
                outfile=os.path.join(OUTPUT_ROOT, "learning_curves_10seeds_slipTRUE.png")
            )
    except Exception as e:
        print(f"[WARN] Cross-seed learning curve failed: {e}")

    # --- Both-condition curriculum ablation plot ---
    try:
        def _mean_metric(results, slip_val, method_key, metric_idx):
            arr = [r[method_key][metric_idx] for r in results
                   if r["slip"] == slip_val and r.get(method_key) is not None]
            return float(np.mean(arr)) if arr else 0.0

        plot_curriculum_ablation_both_conditions(
            with_cur_false={"reward": _mean_metric(all_results, False, "ppo_cur", 0),
                            "success_rate": _mean_metric(all_results, False, "ppo_cur", 3)},
            without_cur_false={"reward": _mean_metric(all_results, False, "ppo_base", 0),
                               "success_rate": _mean_metric(all_results, False, "ppo_base", 3)},
            with_cur_true={"reward": _mean_metric(all_results, True, "ppo_cur", 0),
                           "success_rate": _mean_metric(all_results, True, "ppo_cur", 3)},
            without_cur_true={"reward": _mean_metric(all_results, True, "ppo_base", 0),
                              "success_rate": _mean_metric(all_results, True, "ppo_base", 3)},
            outfile=os.path.join(OUTPUT_ROOT, "curriculum_ablation_both_conditions.png")
        )
    except Exception as e:
        print(f"[WARN] Both-condition ablation plot failed: {e}")

    # --- Save final summary CSV ---
    summary_csv = os.path.join(OUTPUT_ROOT, "summary_10seeds.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved summary CSV -> {summary_csv}")

    # ── Hyperparameter sensitivity ────────────────────────────────────────────
    print("\n=== Hyperparameter Sensitivity Analysis ===")

    Q_EPISODES  = 15_000
    PPO_STEPS   = 1_200_000
    PPO_PHASE1  = 300_000
    PPO_PHASE2  = PPO_STEPS - PPO_PHASE1

    _sens_slip    = True
    _sens_seed    = 0
    _sens_eval_ep = 100

    sens_dir = os.path.join(OUTPUT_ROOT, "sensitivity")
    os.makedirs(sens_dir, exist_ok=True)

    print("--- Q-learning: Learning Rate (alpha) ---")
    q_sens_lr = {}
    for lr in [0.05, 0.10, 0.15, 0.20, 0.30]:
        _base = make_base_env(seed=_sens_seed, is_slippery=_sens_slip, max_steps_guard=2000)
        _env  = EVBatteryWrapper(_base, shaping_reward=0.0, seed_for_snow=42)
        Q_s, _, _ = train_q_learning(_env, episodes=Q_EPISODES, max_steps_train=2000, alpha=lr)
        _eval_base = make_base_env(seed=_sens_seed+500, is_slippery=_sens_slip, max_steps_guard=2000)
        _eval_env  = EVBatteryWrapper(_eval_base, shaping_reward=0.0, seed_for_snow=42)
        rew, _, _, succ, _, _ = evaluate(_eval_env, Q_s, mode="q", eval_episodes=_sens_eval_ep, num_traj=0)
        q_sens_lr[lr] = {"success_rate": succ, "reward": rew}
        print(f"  alpha={lr:.2f}:  success={100*succ:.1f}%   reward={rew:.1f}")

    print("--- Q-learning: Epsilon Decay ---")
    q_sens_decay = {}
    for decay in [0.9990, 0.9995, 0.9998, 0.9999]:
        _base = make_base_env(seed=_sens_seed, is_slippery=_sens_slip, max_steps_guard=2000)
        _env  = EVBatteryWrapper(_base, shaping_reward=0.0, seed_for_snow=42)
        Q_s, _, _ = train_q_learning(_env, episodes=Q_EPISODES, max_steps_train=2000, eps_decay=decay)
        _eval_base = make_base_env(seed=_sens_seed+500, is_slippery=_sens_slip, max_steps_guard=2000)
        _eval_env  = EVBatteryWrapper(_eval_base, shaping_reward=0.0, seed_for_snow=42)
        rew, _, _, succ, _, _ = evaluate(_eval_env, Q_s, mode="q", eval_episodes=_sens_eval_ep, num_traj=0)
        q_sens_decay[decay] = {"success_rate": succ, "reward": rew}
        print(f"  eps_decay={decay}:  success={100*succ:.1f}%   reward={rew:.1f}")

    def _eval_ppo_raw(model, seed_eval=_sens_seed+500, n_ep=_sens_eval_ep):
        def _make():
            base = make_base_env(seed=seed_eval, is_slippery=_sens_slip, max_steps_guard=2000)
            return Monitor(EVBatteryWrapper(base, shaping_reward=0.0, seed_for_snow=42))
        raw_vec = DummyVecEnv([_make])
        rews, succs = [], []
        for _ in range(n_ep):
            obs = raw_vec.reset()
            ep_r, ep_s, done, succ = 0.0, 0, False, False
            while not done and ep_s < 1000:
                act, _ = model.predict(obs, deterministic=True)
                obs, r, done_arr, infos = raw_vec.step(act)
                done = bool(done_arr[0])
                ep_r += float(r[0])
                ep_s += 1
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                if isinstance(info0, dict) and bool(info0.get("is_success", False)):
                    succ = True
            rews.append(ep_r); succs.append(1 if succ else 0)
        return float(np.mean(rews)), float(np.mean(succs))

    print("--- PPO: Clip Range ---")
    ppo_sens_clip = {}
    for clip in [0.1, 0.2, 0.3]:
        _m, _ve, _, _ = train_ppo_curriculum(
            is_slippery=_sens_slip, phase1_steps=PPO_PHASE1, phase2_steps=PPO_PHASE2,
            seed_train=_sens_seed, checkpoint_dir=f"ppo_sens_clip{clip}", clip_range=clip,
        )
        rew, succ = _eval_ppo_raw(_m)
        ppo_sens_clip[clip] = {"success_rate": succ, "reward": rew}
        print(f"  clip_range={clip}:  success={100*succ:.1f}%   reward={rew:.1f}")

    print("--- PPO: Learning Rate ---")
    ppo_sens_lrate = {}
    for lr in [1e-4, 3e-4, 1e-3]:
        _m, _ve, _, _ = train_ppo_curriculum_lr(
            is_slippery=_sens_slip, phase1_steps=PPO_PHASE1, phase2_steps=PPO_PHASE2,
            seed_train=_sens_seed, checkpoint_dir=f"ppo_sens_lr{lr:.0e}", learning_rate=lr,
        )
        rew, succ = _eval_ppo_raw(_m)
        ppo_sens_lrate[lr] = {"success_rate": succ, "reward": rew}
        print(f"  learning_rate={lr:.0e}:  success={100*succ:.1f}%   reward={rew:.1f}")

    plot_hyperparameter_sensitivity(
        {k: v["reward"] for k, v in q_sens_lr.items()},
        "Q-learning Learning Rate (alpha)", metric="reward",
        outfile=os.path.join(sens_dir, "sensitivity_Q_lr.png"))
    plot_hyperparameter_sensitivity(
        {k: v["reward"] for k, v in q_sens_decay.items()},
        "Q-learning Epsilon Decay", metric="reward",
        outfile=os.path.join(sens_dir, "sensitivity_Q_decay.png"))
    plot_hyperparameter_sensitivity(
        {k: v["reward"] for k, v in ppo_sens_clip.items()},
        "PPO Clip Range", metric="reward",
        outfile=os.path.join(sens_dir, "sensitivity_PPO_clip.png"))
    plot_hyperparameter_sensitivity(
        {k: v["reward"] for k, v in ppo_sens_lrate.items()},
        "PPO Learning Rate", metric="reward",
        outfile=os.path.join(sens_dir, "sensitivity_PPO_lr.png"))

    print(f"\nDONE: {N_SEEDS}-seed experiment completed.")
    print("Key outputs:")
    print("  summary_10seeds.csv                  — mean ± std + 95% CI per method/condition")
    print("  welch_ttest_results.csv              — Welch t-test Q vs PPO (slip=TRUE)")
    print("  learning_curves_10seeds_slipTRUE.png — cross-seed learning curves with ±std band")
    print("  sensitivity/                         — hyperparameter sensitivity bar charts")
    print("  seed*/learning_curves.png            — per-seed learning curves")
    print("  seed*/curriculum_ablation.png        — PPO with vs without curriculum")
