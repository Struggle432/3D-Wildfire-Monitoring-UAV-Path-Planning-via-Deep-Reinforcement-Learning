"""
PPO训练脚本
- 使用stable-baselines3的PPO算法
- 支持并行环境加速训练
- 自动保存模型和训练日志
- 支持断点续训
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from wildfire_env import WildfireEnv
from config import TRAIN_CONFIG, ENV_CONFIG, REWARD_CONFIG


class RewardLoggingCallback(BaseCallback):
    """自定义回调：记录奖励分量到日志"""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        return True


def make_env(env_config=None, reward_config=None, rank=0, seed=0):
    """创建环境的工厂函数"""
    def _init():
        env = WildfireEnv(env_config=env_config, reward_config=reward_config)
        env = Monitor(env)
        return env
    return _init


def create_vec_envs(n_envs, env_config=None, reward_config=None):
    """创建并行向量化环境"""
    if n_envs > 1:
        env_fns = [make_env(env_config, reward_config, rank=i)
                   for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([make_env(env_config, reward_config)])
    return vec_env


def train(total_timesteps=None, save_path=None, continue_training=False):
    """主训练函数"""
    cfg = TRAIN_CONFIG.copy()
    timesteps = total_timesteps or cfg["total_timesteps"]

    # 创建输出目录
    log_dir = cfg["log_dir"]
    model_dir = cfg["model_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    save_path = save_path or os.path.join(model_dir, "ppo_wildfire_best")

    print("=" * 60)
    print("3D野火监测 - PPO训练")
    print("=" * 60)
    print(f"  算法: {cfg['algorithm']}")
    print(f"  总步数: {timesteps:,}")
    print(f"  并行环境: {cfg['n_envs']}")
    print(f"  学习率: {cfg['learning_rate']}")
    print(f"  网络结构: π={cfg['net_arch_pi']}, V={cfg['net_arch_vf']}")
    print("=" * 60)

    # 创建训练环境
    train_env = create_vec_envs(cfg["n_envs"])

    # 创建评估环境
    eval_env = create_vec_envs(1)

    # 策略网络参数
    policy_kwargs = dict(
        net_arch=dict(pi=cfg["net_arch_pi"], vf=cfg["net_arch_vf"]),
    )

    # 加载或创建模型
    if continue_training and os.path.exists(save_path + ".zip"):
        print(f"\n从断点续训: {save_path}.zip")
        model = PPO.load(save_path, env=train_env)
    else:
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
            tensorboard_log=log_dir,
            device="auto",
        )

    # 设置日志格式
    tmp_path = os.path.join(log_dir, "ppo_tmp")
    new_logger = configure(tmp_path, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["save_freq"],
        save_path=model_dir,
        name_prefix="ppo_wildfire",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=5,
        deterministic=True,
    )

    # 开始训练
    print(f"\n开始训练 ({timesteps:,} 步)...")
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n训练被中断，保存当前模型...")

    # 保存最终模型
    final_path = os.path.join(model_dir, "ppo_wildfire_final")
    model.save(final_path)
    print(f"\n模型已保存: {final_path}.zip")

    # 清理环境
    train_env.close()
    eval_env.close()

    return model, final_path


if __name__ == "__main__":
    train()
