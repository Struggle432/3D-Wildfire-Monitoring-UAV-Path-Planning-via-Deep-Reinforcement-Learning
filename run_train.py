"""独立训练脚本 - 使用face_recognition环境运行"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from wildfire_env import WildfireEnv
from config import TRAIN_CONFIG, ENV_CONFIG, REWARD_CONFIG


def make_env(rank=0):
    def _init():
        env = WildfireEnv()
        env = Monitor(env)
        return env
    return _init


def main():
    cfg = TRAIN_CONFIG
    total_timesteps = 500_000
    n_envs = 4

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["model_dir"], exist_ok=True)

    print("=" * 60)
    print("Training PPO for Wildfire Monitoring")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print("=" * 60)

    # 创建并行环境
    if n_envs > 1:
        env_fns = [make_env(i) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv([make_env()])

    # 评估环境
    eval_env = DummyVecEnv([make_env()])

    # PPO模型
    policy_kwargs = dict(
        net_arch=dict(pi=cfg["net_arch_pi"], vf=cfg["net_arch_vf"]),
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=cfg["log_dir"],
        device="auto",
    )

    # 回调
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["save_freq"],
        save_path=cfg["model_dir"],
        name_prefix="ppo_wildfire",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg["model_dir"], "ppo_wildfire_best"),
        log_path=os.path.join(cfg["log_dir"], "eval"),
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )

    # 保存最终模型
    final_path = os.path.join(cfg["model_dir"], "ppo_wildfire_final")
    model.save(final_path)
    print(f"\nModel saved: {final_path}.zip")

    train_env.close()
    eval_env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
