"""运行评估和可视化的独立脚本"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from stable_baselines3 import PPO
from wildfire_env import WildfireEnv
from baselines import RandomAgent, GreedyAgent, AStarAgent, run_baseline
from evaluate import evaluate_ppo, run_full_evaluation, run_multi_scenario, run_ablation
from visualize import (
    generate_episode_video, generate_comparison_video,
    render_episode_3d, plot_training_curves, create_fire_colormap
)

model_path = "results/models/ppo_wildfire_best/best_model.zip"
print(f"Loading model: {model_path}")
model = PPO.load(model_path)
env = WildfireEnv()

# ===== 1. 运行全面评估 =====
print("\n" + "=" * 60)
print("Step 1: Full Evaluation")
print("=" * 60)
n_ep = 10

print("\nEvaluating PPO...")
ppo_results = evaluate_ppo(model_path, env, n_ep)
print(f"  PPO: reward={ppo_results['mean_reward']:.2f}+/-{ppo_results['std_reward']:.2f}, "
      f"coverage={ppo_results['mean_coverage']:.2%}")

baseline_results = {}
for AgentClass in [RandomAgent, GreedyAgent, AStarAgent]:
    agent = AgentClass(env)
    print(f"\nEvaluating {agent.name}...")
    res = run_baseline(env, agent, n_ep)
    baseline_results[agent.name] = res
    print(f"  {agent.name}: reward={res['mean_reward']:.2f}+/-{res['std_reward']:.2f}, "
          f"coverage={res['mean_coverage']:.2%}")

all_results = {"PPO": ppo_results, **baseline_results}

# ===== 2. 绘制对比图 =====
print("\n" + "=" * 60)
print("Step 2: Generate Comparison Plots")
print("=" * 60)
output_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)

# 算法对比柱状图
names = list(all_results.keys())
means_r = [all_results[n]["mean_reward"] for n in names]
stds_r = [all_results[n]["std_reward"] for n in names]
means_c = [all_results[n]["mean_coverage"] * 100 for n in names]
stds_c = [all_results[n]["std_coverage"] * 100 for n in names]
means_s = [all_results[n]["mean_steps"] for n in names]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

axes[0].bar(names, means_r, yerr=stds_r, capsize=5, color=colors)
axes[0].set_title('Average Total Reward', fontsize=14)
axes[0].set_ylabel('Reward')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(names, means_c, yerr=stds_c, capsize=5, color=colors)
axes[1].set_title('Fire Coverage Rate (%)', fontsize=14)
axes[1].set_ylabel('Coverage (%)')
axes[1].grid(axis='y', alpha=0.3)

axes[2].bar(names, means_s, color=colors)
axes[2].set_title('Average Steps', fontsize=14)
axes[2].set_ylabel('Steps')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: algorithm_comparison.png")

# ===== 3. 生成3D路径图 =====
print("\n" + "=" * 60)
print("Step 3: Generate 3D Path Plots")
print("=" * 60)

# PPO路径
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

render_episode_3d(env, path, fire_history, cov_history,
                  title="PPO UAV Wildfire Monitoring 3D Path",
                  save_path="results/plots/path_3d_ppo.png")

# 基线路径
for agent in [RandomAgent(env), GreedyAgent(env), AStarAgent(env)]:
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
    safe_name = {chr(0): ""}  # placeholder
    name_map = {"随机游走": "random", "贪心策略": "greedy", "A*规划": "astar"}
    safe_name = name_map.get(agent.name, "baseline")
    render_episode_3d(env, path, fire_history, cov_history,
                      title=f"{agent.name} UAV 3D Path",
                      save_path=f"results/plots/path_3d_{safe_name}.png")

# ===== 4. 生成视频 =====
print("\n" + "=" * 60)
print("Step 4: Generate Videos")
print("=" * 60)

print("\nGenerating PPO monitoring video...")
generate_episode_video(env, model, "results/videos/ppo_wildfire.gif", is_sb3_model=True)

print("\nGenerating comparison video...")
agents = [RandomAgent(env), GreedyAgent(env), AStarAgent(env)]
generate_comparison_video(env, model, agents, "results/videos/comparison.gif")

# ===== 5. 环境示意图 =====
print("\n" + "=" * 60)
print("Step 5: Environment Diagram")
print("=" * 60)

fire_cmap = create_fire_colormap()
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

obs, _ = env.reset(seed=42)
im = axes[0].imshow(env.fire_map, cmap=fire_cmap, vmin=0, vmax=1,
                     origin='lower', extent=[0, env.grid_w, 0, env.grid_h])
sx, sy, sz = env.cfg["uav_start"]
axes[0].scatter(sx + 0.5, sy + 0.5, color='blue', s=200, marker='^',
                edgecolors='white', linewidths=2, zorder=5, label='UAV Start')
wx, wy = env.wind_dir * env.wind_speed * 5
axes[0].annotate('', xy=(env.grid_w/2 + wx, env.grid_h/2 + wy),
                 xytext=(env.grid_w/2, env.grid_h/2),
                 arrowprops=dict(arrowstyle='->', color='cyan', lw=3))
axes[0].text(env.grid_w/2 + wx + 1, env.grid_h/2 + wy + 1,
             'Wind', color='cyan', fontsize=12, fontweight='bold')
axes[0].set_title('Initial Fire Field', fontsize=14)
axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
axes[0].legend(loc='upper left')
plt.colorbar(im, ax=axes[0], label='Fire Intensity', shrink=0.8)

for _ in range(50):
    env._spread_fire()
im2 = axes[1].imshow(env.fire_map, cmap=fire_cmap, vmin=0, vmax=1,
                      origin='lower', extent=[0, env.grid_w, 0, env.grid_h])
axes[1].set_title('After 50 Steps of Fire Spread', fontsize=14)
axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1], label='Fire Intensity', shrink=0.8)

plt.suptitle('Wildfire Monitoring Environment', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("results/plots/env_diagram.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: env_diagram.png")

# ===== 6. 覆盖率随时间变化 =====
print("\n" + "=" * 60)
print("Step 6: Coverage Over Time")
print("=" * 60)

obs, _ = env.reset(seed=42)
coverages_ppo = []
for step in range(env.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    coverages_ppo.append(info.get("fire_coverage", 0))
    if terminated or truncated:
        break

obs, _ = env.reset(seed=42)
coverages_greedy = []
agent = GreedyAgent(env)
for step in range(env.max_steps):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    coverages_greedy.append(info.get("fire_coverage", 0))
    if terminated or truncated:
        break

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(coverages_ppo, linewidth=2, color='#FF6B6B', label='PPO')
ax.plot(coverages_greedy, linewidth=2, color='#4ECDC4', label='Greedy')
ax.fill_between(range(len(coverages_ppo)), coverages_ppo, alpha=0.15, color='#FF6B6B')
ax.fill_between(range(len(coverages_greedy)), coverages_greedy, alpha=0.15, color='#4ECDC4')
ax.set_title('Fire Coverage Rate Over Time', fontsize=14)
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('Coverage Rate', fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/coverage_over_time.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: coverage_over_time.png")

# ===== 7. 多场景评估 =====
print("\n" + "=" * 60)
print("Step 7: Multi-Scenario Evaluation")
print("=" * 60)

from config import EVAL_CONFIG, ENV_CONFIG, REWARD_CONFIG
scenarios = EVAL_CONFIG["scenarios"]
scenario_results = {}

for scenario in scenarios:
    print(f"\n  Scenario: {scenario['name']}")
    env_cfg = {**ENV_CONFIG}
    env_cfg["n_fire_sources"] = scenario["n_fire_sources"]
    env_cfg["wind_speed"] = scenario["wind_speed"]
    env_cfg["wind_direction"] = scenario["wind_direction"]
    env_s = WildfireEnv(env_config=env_cfg)
    ppo_res = evaluate_ppo(model_path, env_s, n_episodes=5)
    scenario_results[scenario['name']] = {
        "reward": ppo_res["mean_reward"],
        "coverage": ppo_res["mean_coverage"],
        "steps": ppo_res["mean_steps"],
    }
    print(f"    reward={ppo_res['mean_reward']:.2f}, coverage={ppo_res['mean_coverage']:.2%}")

# 多场景对比图
s_names = list(scenario_results.keys())
s_rewards = [scenario_results[n]["reward"] for n in s_names]
s_coverages = [scenario_results[n]["coverage"] * 100 for n in s_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(s_names, s_rewards, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[0].set_title('Reward by Scenario', fontsize=14)
axes[0].set_xlabel('Reward')
axes[0].grid(axis='x', alpha=0.3)
axes[1].barh(s_names, s_coverages, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[1].set_title('Coverage by Scenario (%)', fontsize=14)
axes[1].set_xlabel('Coverage (%)')
axes[1].grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/multi_scenario.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: multi_scenario.png")

# ===== 8. 报告摘要 =====
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for name, res in all_results.items():
    print(f"  {name:12s}: reward={res['mean_reward']:8.2f}+/-{res['std_reward']:.2f}, "
          f"coverage={res['mean_coverage']:.2%}, steps={res['mean_steps']:.0f}")

print("\nAll done!")
