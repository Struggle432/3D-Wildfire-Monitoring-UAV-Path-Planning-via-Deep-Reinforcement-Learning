"""
一键运行脚本
- 训练PPO模型
- 运行全面评估
- 生成所有可视化
- 生成视频
"""

import os
import sys
import argparse
import numpy as np

# 确保工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_training(timesteps=None):
    """步骤1：训练模型"""
    print("\n" + "=" * 70)
    print("  步骤 1/4: 训练PPO模型")
    print("=" * 70)
    from train import train
    model, model_path = train(total_timesteps=timesteps)
    return model_path


def run_evaluation(model_path):
    """步骤2：运行评估"""
    print("\n" + "=" * 70)
    print("  步骤 2/4: 运行全面评估")
    print("=" * 70)
    from evaluate import run_full_evaluation, run_multi_scenario, run_ablation
    all_res = run_full_evaluation(model_path)
    scenario_res = run_multi_scenario(model_path)
    ablation_res = run_ablation(model_path)
    return all_res, scenario_res, ablation_res


def run_visualization(model_path):
    """步骤3：生成可视化"""
    print("\n" + "=" * 70)
    print("  步骤 3/4: 生成可视化与视频")
    print("=" * 70)

    from stable_baselines3 import PPO
    from wildfire_env import WildfireEnv
    from visualize import (
        generate_episode_video, generate_comparison_video,
        render_episode_3d, plot_training_curves, create_fire_colormap
    )
    from baselines import RandomAgent, GreedyAgent, AStarAgent
    import matplotlib.pyplot as plt

    model = PPO.load(model_path)
    env = WildfireEnv()

    # 1. 生成PPO监测视频
    print("\n生成PPO监测视频...")
    generate_episode_video(
        env, model,
        "results/videos/ppo_wildfire.gif",
        is_sb3_model=True
    )

    # 2. 生成对比视频
    print("\n生成多算法对比视频...")
    agents = [RandomAgent(env), GreedyAgent(env), AStarAgent(env)]
    generate_comparison_video(
        env, model, agents,
        "results/videos/comparison.gif"
    )

    # 3. 生成3D路径效果图
    print("\n生成3D路径效果图...")
    obs, _ = env.reset(seed=42)
    path = [tuple(env.uav_pos)]
    fire_history = [env.fire_map.copy()]
    cov_history = [env.coverage_map.copy()]

    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        path.append(tuple(env.uav_pos))
        fire_history.append(env.fire_map.copy())
        cov_history.append(env.coverage_map.copy())
        if terminated or truncated:
            break

    render_episode_3d(
        env, path, fire_history, cov_history,
        title="PPO无人机野火监测3D路径",
        save_path="results/plots/path_3d_ppo.png"
    )

    # 4. 为每个基线也生成3D路径图
    for agent in agents:
        obs, _ = env.reset(seed=42)
        path = [tuple(env.uav_pos)]
        fire_history = [env.fire_map.copy()]
        cov_history = [env.coverage_map.copy()]

        for step in range(env.max_steps):
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            path.append(tuple(env.uav_pos))
            fire_history.append(env.fire_map.copy())
            cov_history.append(env.coverage_map.copy())
            if terminated or truncated:
                break

        render_episode_3d(
            env, path, fire_history, cov_history,
            title=f"{agent.name}无人机野火监测3D路径",
            save_path=f"results/plots/path_3d_{agent.name}.png"
        )

    # 5. 训练曲线
    print("\n绘制训练曲线...")
    plot_training_curves()

    # 6. 生成环境示意图
    print("\n生成环境示意图...")
    _draw_env_diagram(env)

    print("\n所有可视化生成完成！")


def _draw_env_diagram(env):
    """绘制环境示意图"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle
    from visualize import create_fire_colormap

    fire_cmap = create_fire_colormap()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：初始火场
    ax1 = axes[0]
    obs, _ = env.reset(seed=42)
    im = ax1.imshow(env.fire_map, cmap=fire_cmap, vmin=0, vmax=1,
                     origin='lower', extent=[0, env.grid_w, 0, env.grid_h])

    # 标注UAV起点
    sx, sy, sz = env.cfg["uav_start"]
    ax1.scatter(sx + 0.5, sy + 0.5, color='blue', s=200, marker='^',
                edgecolors='white', linewidths=2, zorder=5, label='UAV起点')

    # 标注风向
    wx, wy = env.wind_dir * env.wind_speed * 5
    ax1.annotate('', xy=(env.grid_w / 2 + wx, env.grid_h / 2 + wy),
                 xytext=(env.grid_w / 2, env.grid_h / 2),
                 arrowprops=dict(arrowstyle='->', color='cyan', lw=3))
    ax1.text(env.grid_w / 2 + wx + 1, env.grid_h / 2 + wy + 1,
             '风向', color='cyan', fontsize=12, fontweight='bold')

    ax1.set_title('初始火场态势', fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend(loc='upper left')
    plt.colorbar(im, ax=ax1, label='火势强度', shrink=0.8)

    # 右图：火场蔓延后
    ax2 = axes[1]
    # 模拟50步火势蔓延
    for _ in range(50):
        env._spread_fire()

    im2 = ax2.imshow(env.fire_map, cmap=fire_cmap, vmin=0, vmax=1,
                      origin='lower', extent=[0, env.grid_w, 0, env.grid_h])
    ax2.set_title('火势蔓延50步后', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='火势强度', shrink=0.8)

    plt.suptitle('野火监测环境示意图', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("results/plots/env_diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(" 环境示意图已保存")


def run_report(model_path, all_res, scenario_res, ablation_res):
    """步骤4：生成实验报告数据摘要"""
    print("\n" + "=" * 70)
    print("  步骤 4/4: 生成报告数据摘要")
    print("=" * 70)

    report = []
    report.append("=" * 60)
    report.append("实验结果摘要")
    report.append("=" * 60)

    # 算法对比
    report.append("\n算法性能对比:")
    for name, res in all_res.items():
        report.append(f"  {name:10s}: 奖励={res['mean_reward']:8.2f}±{res['std_reward']:.2f}, "
                       f"覆盖率={res['mean_coverage']:.2%}, 步数={res['mean_steps']:.0f}")

    # 多场景
    report.append("\n多场景评估:")
    for name, res in scenario_res.items():
        report.append(f"  {name}: 奖励={res['reward']:.2f}, 覆盖率={res['coverage']:.2%}")

    # 消融实验
    report.append("\n消融实验:")
    for name, val in ablation_res.items():
        report.append(f"  {name}: 覆盖率={val:.2%}")

    report_text = "\n".join(report)
    print(report_text)

    # 保存到文件
    with open("results/report_summary.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(description="3D野火监测RL实验")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练")
    parser.add_argument("--timesteps", type=int, default=None, help="训练步数")
    parser.add_argument("--model-path", type=str, default=None, help="模型路径")
    args = parser.parse_args()

    # 步骤1：训练
    if args.skip_train:
        model_path = args.model_path or "results/models/ppo_wildfire_best"
        if not os.path.exists(model_path + ".zip"):
            # 尝试final模型
            model_path = "results/models/ppo_wildfire_final"
        print(f"跳过训练，使用模型: {model_path}")
    else:
        model_path = run_training(args.timesteps)

    # 步骤2：评估
    all_res, scenario_res, ablation_res = run_evaluation(model_path)

    # 步骤3：可视化
    run_visualization(model_path)

    # 步骤4：报告
    run_report(model_path, all_res, scenario_res, ablation_res)

    print("\n" + "=" * 70)
    print("  所有步骤完成！")
    print("  结果保存在 results/ 目录下")
    print("  图表: results/plots/")
    print("  视频: results/videos/")
    print("  模型: results/models/")
    print("=" * 70)


if __name__ == "__main__":
    main()
