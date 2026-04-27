# algorithms/ppo_curriculum.py
# Sections 8-11 and 14b from winter_rl_10seed_final_v3.py (unchanged)

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment.grid_env import make_base_env
from environment.ev_battery_wrapper import EVBatteryWrapper

# =========================================================
# 8) PPO callbacks
# =========================================================
class PPOLearningCurveCallback(BaseCallback):
    """Records mean episode reward every `log_freq` steps for learning curve plotting."""
    def __init__(self, log_freq=50_000):
        super().__init__()
        self.log_freq = int(log_freq)
        self.next_log = int(log_freq)
        self.timesteps = []
        self.mean_rewards = []
        self._ep_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_rewards.append(float(info["episode"]["r"]))
        cur = int(self.num_timesteps)
        if cur >= self.next_log:
            mean_r = float(np.mean(self._ep_rewards)) if self._ep_rewards else float("nan")
            self.timesteps.append(cur)
            self.mean_rewards.append(mean_r)
            self._ep_rewards = []
            self.next_log += self.log_freq
        return True

class PPOBarHeartbeatCallback(BaseCallback):
    def __init__(self, total_timesteps: int, heartbeat_every: int = 200_000):
        super().__init__()
        self.total = int(total_timesteps)
        self.heartbeat_every = int(heartbeat_every)
        self.next_heartbeat = int(heartbeat_every)

    def _on_step(self) -> bool:
        cur = int(self.num_timesteps)
        if cur >= self.next_heartbeat:
            print(f"[PPO] heartbeat: {cur}/{self.total} timesteps...")
            self.next_heartbeat += self.heartbeat_every
        return True

# =========================================================
# 9) PPO Curriculum training
# =========================================================
def train_ppo_curriculum(
    is_slippery: bool,
    phase1_steps: int = 300_000,
    phase2_steps: int = 900_000,
    seed_train: int = 888,
    checkpoint_dir: str = "ppo_checkpoints",
    clip_range: float = 0.2,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env_phase1():
        base = make_base_env(seed=seed_train, is_slippery=is_slippery, max_steps_guard=2000)
        env = EVBatteryWrapper(
            base,
            step_penalty=0.10,
            charge_reward=150.0,
            shaping_reward=0.5,
            core_cost=0.05,
            edge_cost=0.05,
            outer_cost=0.05,
            seed_for_snow=42
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env_phase1])
    use_vecnorm = False
    try:
        from stable_baselines3.common.vec_env import VecNormalize
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        use_vecnorm = True
    except Exception:
        pass

    policy_kwargs = dict(net_arch=[256, 256])
    lc_cb = PPOLearningCurveCallback(log_freq=50_000)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=0.02,
        policy_kwargs=policy_kwargs,
        seed=seed_train,
        device="auto",
    )

    ckpt = CheckpointCallback(save_freq=200_000, save_path=checkpoint_dir, name_prefix="ppo_curriculum")
    bar  = PPOBarHeartbeatCallback(total_timesteps=phase1_steps + phase2_steps, heartbeat_every=200_000)

    print(f"PPO curriculum Phase 1 started ({phase1_steps} timesteps, clip_range={clip_range})...")
    model.learn(total_timesteps=int(phase1_steps), callback=[ckpt, bar, lc_cb], progress_bar=False)

    def make_env_phase2():
        base = make_base_env(seed=seed_train, is_slippery=is_slippery, max_steps_guard=2000)
        return Monitor(EVBatteryWrapper(base, shaping_reward=0.0, seed_for_snow=42))

    if hasattr(vec_env, "venv") and hasattr(vec_env.venv, "envs"):
        vec_env.venv.envs[0] = make_env_phase2()
    elif hasattr(vec_env, "envs"):
        vec_env.envs[0] = make_env_phase2()
    else:
        raise RuntimeError("Cannot swap PPO curriculum env to Phase 2")

    vec_env.reset()

    print(f"PPO curriculum Phase 2 started ({phase2_steps} timesteps)...")
    model.learn(total_timesteps=int(phase2_steps), reset_num_timesteps=False,
                callback=[ckpt, bar, lc_cb], progress_bar=False)

    print("PPO training completed.")
    model.save("ppo_final_model")
    if use_vecnorm:
        vec_env.save("ppo_vecnormalize.pkl")

    return model, vec_env, lc_cb.timesteps, lc_cb.mean_rewards

# =========================================================
# 10) PPO Baseline (no curriculum) — for ablation study
# =========================================================
def train_ppo_baseline(
    is_slippery: bool,
    total_steps: int = 1_200_000,
    seed_train: int = 888,
    checkpoint_dir: str = "ppo_baseline_checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env():
        base = make_base_env(seed=seed_train, is_slippery=is_slippery, max_steps_guard=2000)
        return Monitor(EVBatteryWrapper(base, shaping_reward=0.0, seed_for_snow=42))

    vec_env = DummyVecEnv([make_env])
    use_vecnorm = False
    try:
        from stable_baselines3.common.vec_env import VecNormalize
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        use_vecnorm = True
    except Exception:
        pass

    policy_kwargs = dict(net_arch=[256, 256])
    lc_cb = PPOLearningCurveCallback(log_freq=50_000)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=policy_kwargs,
        seed=seed_train,
        device="auto",
    )

    ckpt = CheckpointCallback(save_freq=200_000, save_path=checkpoint_dir, name_prefix="ppo_baseline")
    bar  = PPOBarHeartbeatCallback(total_timesteps=total_steps, heartbeat_every=200_000)

    print(f"PPO Baseline (no curriculum) training started ({total_steps} timesteps)...")
    model.learn(total_timesteps=int(total_steps), callback=[ckpt, bar, lc_cb], progress_bar=False)

    print("PPO Baseline training completed.")
    model.save("ppo_baseline_model")
    if use_vecnorm:
        vec_env.save("ppo_baseline_vecnormalize.pkl")

    return model, vec_env, lc_cb.timesteps, lc_cb.mean_rewards

# =========================================================
# 11) Eval VecEnv builder
# =========================================================
def make_ppo_eval_vecenv_from_trained(trained_vec_env, seed_eval: int, is_slippery: bool):
    def _make_eval_env():
        base = make_base_env(seed=seed_eval, is_slippery=is_slippery, max_steps_guard=2000)
        return EVBatteryWrapper(base, shaping_reward=0.0, seed_for_snow=42)

    eval_dummy = DummyVecEnv([_make_eval_env])
    if hasattr(trained_vec_env, "save"):
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            tmp_path = "_tmp_vecnormalize_eval.pkl"
            trained_vec_env.save(tmp_path)
            eval_vec = VecNormalize.load(tmp_path, eval_dummy)
            eval_vec.training = False
            eval_vec.norm_reward = False
            return eval_vec
        except Exception:
            pass
    return eval_dummy

# =========================================================
# 14b) PPO Curriculum wrapper with custom learning_rate
#      (for sensitivity analysis)
# =========================================================
def train_ppo_curriculum_lr(is_slippery, phase1_steps, phase2_steps,
                             seed_train, checkpoint_dir, learning_rate):
    """Identical to train_ppo_curriculum but exposes learning_rate for sensitivity analysis."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env_phase1():
        base = make_base_env(seed=seed_train, is_slippery=is_slippery, max_steps_guard=2000)
        env = EVBatteryWrapper(
            base, step_penalty=0.10, charge_reward=150.0, shaping_reward=0.5,
            core_cost=0.05, edge_cost=0.05, outer_cost=0.05, seed_for_snow=42)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env_phase1])
    use_vecnorm = False
    try:
        from stable_baselines3.common.vec_env import VecNormalize
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        use_vecnorm = True
    except Exception:
        pass

    lc_cb = PPOLearningCurveCallback(log_freq=50_000)
    model = PPO(
        "MlpPolicy", vec_env, verbose=0,
        learning_rate=learning_rate,
        n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed_train, device="auto",
    )
    ckpt = CheckpointCallback(save_freq=200_000, save_path=checkpoint_dir, name_prefix="ppo_sens_lr")
    bar  = PPOBarHeartbeatCallback(total_timesteps=phase1_steps + phase2_steps, heartbeat_every=200_000)

    print(f"PPO Phase 1 started ({phase1_steps} timesteps, lr={learning_rate:.0e})...")
    model.learn(total_timesteps=int(phase1_steps), callback=[ckpt, bar, lc_cb], progress_bar=False)

    def make_env_phase2():
        base = make_base_env(seed=seed_train, is_slippery=is_slippery, max_steps_guard=2000)
        return Monitor(EVBatteryWrapper(base, shaping_reward=0.0, seed_for_snow=42))

    if hasattr(vec_env, "venv") and hasattr(vec_env.venv, "envs"):
        vec_env.venv.envs[0] = make_env_phase2()
    elif hasattr(vec_env, "envs"):
        vec_env.envs[0] = make_env_phase2()
    else:
        raise RuntimeError("Cannot swap PPO curriculum env to Phase 2")

    vec_env.reset()
    print(f"PPO Phase 2 started ({phase2_steps} timesteps, lr={learning_rate:.0e})...")
    model.learn(total_timesteps=int(phase2_steps), reset_num_timesteps=False,
                callback=[ckpt, bar, lc_cb], progress_bar=False)
    print("PPO training completed.")
    model.save("ppo_final_model")
    if use_vecnorm:
        vec_env.save("ppo_vecnormalize.pkl")
    return model, vec_env, lc_cb.timesteps, lc_cb.mean_rewards
