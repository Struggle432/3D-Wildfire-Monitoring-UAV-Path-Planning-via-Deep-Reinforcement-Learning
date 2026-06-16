"""
评估与对比脚本
- 加载训练好的PPO模型
- 运行所有基线算法
- 多场景对比评估
- 消融实验
- 生成对比图表和数据
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from stable_baselines3 import PPO
from wildfire_env import WildfireEnv
from baselines import RandomAgent, GreedyAgent, AStarAgent, run_baseline
from config import ENV_CONFIG, REWARD_CONFIG, EVAL_CONFIG, VIS_CONFIG

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_ppo(model_path, env, n_episodes=10):
    """评估PPO模型"""
    model = PPO.load(model_path)
    results = {
        "rewards": [], "coverages": [], "min_dists": [],
        "steps": [], "paths": [], "fire_histories": [],
        "coverage_histories": [], "reward_components_list": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 200)
        total_reward = 0
        step = 0
        ep_components = []

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if "reward_components" in info:
                ep_components.append(info["reward_components"])

            if terminated or truncated:
                break

        results["rewards"].append(total_reward)
        results["coverages"].append(info.get("fire_coverage", 0))
        results["min_dists"].append(info.get("min_dist_to_fire", 0))
        results["steps"].append(step + 1)
        results["paths"].append(env.info_stats["path"])
        results["fire_histories"].append(env.info_stats["fire_history"])
        results["coverage_histories"].append(env.info_stats["coverage_history"])
        results["reward_components_list"].append(ep_components)

    results["mean_reward"] = np.mean(results["rewards"])
    results["mean_coverage"] = np.mean(results["coverages"])
    results["mean_steps"] = np.mean(results["steps"])
    results["std_reward"] = np.std(results["rewards"])
    results["std_coverage"] = np.std(results["coverages"])

    return results


def run_full_evaluation(model_path, n_episodes=None, output_dir="results/plots"):
    """运行完整评估：PPO vs 所有基线"""
    os.makedirs(output_dir, exist_ok=True)
    n_ep = n_episodes or EVAL_CONFIG["n_eval_episodes"]

    print("=" * 60)
    print("[INFO] 开始全面评估")
    print("=" * 60)

    # 创建环境
    env = WildfireEnv()

    # 评估PPO
    print("\n[INFO] 评估PPO模型...")
    ppo_results = evaluate_ppo(model_path, env, n_ep)
    print(f"   平均奖励: {ppo_results['mean_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
    print(f"   平均覆盖率: {ppo_results['mean_coverage']:.2%}")

    # 评估基线
    baseline_results = {}
    for AgentClass in [RandomAgent, GreedyAgent, AStarAgent]:
        agent = AgentClass(env)
        print(f"\n[INFO] 评估{agent.name}...")
        res = run_baseline(env, agent, n_ep)
        baseline_results[agent.name] = res
        print(f"   平均奖励: {res['mean_reward']:.2f} ± {res['std_reward']:.2f}")
        print(f"   平均覆盖率: {res['mean_coverage']:.2%}")

    # 生成对比图表
    all_results = {"PPO": ppo_results, **baseline_results}
    _plot_comparison(all_results, output_dir)
    _plot_reward_curves(ppo_results, output_dir)
    _plot_coverage_over_steps(ppo_results, output_dir)

    return all_results


def _plot_comparison(all_results, output_dir):
    """绘制算法对比柱状图"""
    names = list(all_results.keys())
    means_r = [all_results[n]["mean_reward"] for n in names]
    stds_r = [all_results[n]["std_reward"] for n in names]
    means_c = [all_results[n]["mean_coverage"] for n in names]
    stds_c = [all_results[n]["std_coverage"] for n in names]
    means_s = [all_results[n]["mean_steps"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 奖励对比
    bars = axes[0].bar(names, means_r, yerr=stds_r, capsize=5,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0].set_title('平均总奖励对比', fontsize=14)
    axes[0].set_ylabel('奖励')
    axes[0].grid(axis='y', alpha=0.3)

    # 覆盖率对比
    bars = axes[1].bar(names, [c * 100 for c in means_c],
                       yerr=[s * 100 for s in stds_c], capsize=5,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1].set_title('火区覆盖率对比 (%)', fontsize=14)
    axes[1].set_ylabel('覆盖率 (%)')
    axes[1].grid(axis='y', alpha=0.3)

    # 步数对比
    axes[2].bar(names, means_s,
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[2].set_title('平均完成步数对比', fontsize=14)
    axes[2].set_ylabel('步数')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 算法对比图已保存")


def _plot_reward_curves(ppo_results, output_dir):
    """绘制PPO各奖励分量随时间变化"""
    components_list = ppo_results.get("reward_components_list", [])
    if not components_list or not components_list[0]:
        return

    # 取第一个回合的奖励分量
    components = components_list[0]
    keys = ["info_gain", "frontier", "move_cost", "safety", "miss", "altitude", "total"]
    labels = ["信息增益", "前沿覆盖", "运动成本", "安全风险", "遗漏惩罚", "高度奖励", "总奖励"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(zip(keys, labels)):
        values = [c.get(key, 0) for c in components]
        axes[idx].plot(values, linewidth=0.8, alpha=0.8)
        axes[idx].set_title(label, fontsize=12)
        axes[idx].set_xlabel('步数')
        axes[idx].grid(alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(keys), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('PPO各奖励分量随时间变化', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_components.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 奖励分量图已保存")


def _plot_coverage_over_steps(ppo_results, output_dir):
    """绘制覆盖率随步数变化"""
    coverage_histories = ppo_results.get("coverage_histories", [])
    fire_histories = ppo_results.get("fire_histories", [])

    if not coverage_histories or not fire_histories:
        return

    # 计算每步的覆盖率
    coverages = []
    for step_idx in range(min(len(fire_histories[0]), len(coverage_histories[0]))):
        fire = fire_histories[0][step_idx]
        cov = coverage_histories[0][step_idx]
        total_fire = np.sum(fire > 0.1)
        if total_fire > 0:
            covered = np.sum((fire > 0.1) & (cov > 0.5))
            coverages.append(covered / total_fire)
        else:
            coverages.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coverages, linewidth=2, color='#FF6B6B')
    ax.fill_between(range(len(coverages)), coverages, alpha=0.2, color='#FF6B6B')
    ax.set_title('火区覆盖率随时间变化', fontsize=14)
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('覆盖率', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coverage_over_time.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 覆盖率曲线已保存")


def run_multi_scenario(model_path, output_dir="results/plots"):
    """多场景评估"""
    os.makedirs(output_dir, exist_ok=True)
    scenarios = EVAL_CONFIG["scenarios"]

    print("\n" + "=" * 60)
    print("[INFO] 多场景评估")
    print("=" * 60)

    scenario_results = {}

    for scenario in scenarios:
        print(f"\n📍 场景: {scenario['name']}")
        env_cfg = {**ENV_CONFIG}
        env_cfg["n_fire_sources"] = scenario["n_fire_sources"]
        env_cfg["wind_speed"] = scenario["wind_speed"]
        env_cfg["wind_direction"] = scenario["wind_direction"]

        env = WildfireEnv(env_config=env_cfg)
        ppo_res = evaluate_ppo(model_path, env, n_episodes=5)

        scenario_results[scenario['name']] = {
            "reward": ppo_res["mean_reward"],
            "coverage": ppo_res["mean_coverage"],
            "steps": ppo_res["mean_steps"],
        }

        print(f"   奖励: {ppo_res['mean_reward']:.2f}, "
              f"覆盖率: {ppo_res['mean_coverage']:.2%}, "
              f"步数: {ppo_res['mean_steps']:.0f}")

    # 绘制多场景对比
    _plot_multi_scenario(scenario_results, output_dir)

    return scenario_results


def _plot_multi_scenario(scenario_results, output_dir):
    """绘制多场景对比图"""
    names = list(scenario_results.keys())
    rewards = [scenario_results[n]["reward"] for n in names]
    coverages = [scenario_results[n]["coverage"] * 100 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(names, rewards, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0].set_title('不同场景下的总奖励', fontsize=14)
    axes[0].set_xlabel('奖励')
    axes[0].grid(axis='x', alpha=0.3)

    axes[1].barh(names, coverages, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1].set_title('不同场景下的覆盖率 (%)', fontsize=14)
    axes[1].set_xlabel('覆盖率 (%)')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_scenario.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 多场景对比图已保存")


def run_ablation(model_path, output_dir="results/plots"):
    """消融实验：逐项去掉奖励分量"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("[INFO] 消融实验")
    print("=" * 60)

    # 完整奖励
    env = WildfireEnv()
    full_res = evaluate_ppo(model_path, env, n_episodes=5)

    # 逐项去掉奖励分量
    ablation_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    ablation_names = ["信息增益", "前沿覆盖", "运动成本", "安全风险", "遗漏惩罚", "高度奖励"]

    ablation_results = {"完整": full_res["mean_coverage"]}

    for key, name in zip(ablation_keys, ablation_names):
        rw_cfg = {**REWARD_CONFIG}
        rw_cfg[key] = 0.0  # 去掉该分量
        env = WildfireEnv(reward_config=rw_cfg)
        res = evaluate_ppo(model_path, env, n_episodes=5)
        ablation_results[f"无{name}"] = res["mean_coverage"]
        print(f"  无{name}: 覆盖率 = {res['mean_coverage']:.2%}")

    # 绘制消融实验图
    names = list(ablation_results.keys())
    values = [ablation_results[n] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#FF6B6B'] + ['#4ECDC4'] * (len(names) - 1)
    bars = ax.bar(names, values, color=colors)
    ax.set_title('消融实验：各奖励分量对覆盖率的影响', fontsize=14)
    ax.set_ylabel('覆盖率 (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # 标注数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_study.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 消融实验图已保存")

    return ablation_results


if __name__ == "__main__":
    model_path = "results/models/ppo_wildfire_best"
    if os.path.exists(model_path + ".zip"):
        all_res = run_full_evaluation(model_path)
        run_multi_scenario(model_path)
        run_ablation(model_path)
    else:
        print("[ERR] 未找到训练好的模型，请先运行 train.py")
