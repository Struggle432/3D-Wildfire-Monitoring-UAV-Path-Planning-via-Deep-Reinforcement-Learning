"""
3D可视化与视频生成
- 3D路径效果图（无人机轨迹 + 火场）
- 训练曲线
- 2D火场热力图动画
- 3D飞行视频
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import imageio

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from wildfire_env import WildfireEnv
from config import ENV_CONFIG, VIS_CONFIG


def create_fire_colormap():
    """创建火势专用颜色映射：黑->红->橙->黄->白"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#000000', '#1a0000', '#4d0000', '#990000',
              '#cc3300', '#ff6600', '#ff9933', '#ffcc00', '#ffff66', '#ffffff']
    return LinearSegmentedColormap.from_list('fire', colors, N=256)


def render_episode_3d(env, path, fire_history, coverage_history,
                       title="UAV野火监测路径", save_path=None):
    """渲染3D路径效果图（静态图）"""
    fire_cmap = create_fire_colormap()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    grid_w, grid_h = env.grid_w, env.grid_h

    # 绘制火场地面（取最终帧）
    if fire_history:
        fire = fire_history[-1]
    else:
        fire = env.fire_map

    # 创建地面网格
    X, Y = np.meshgrid(range(grid_w), range(grid_h))

    # 绘制火势热力图（投影到地面）
    fire_surface = ax.plot_surface(
        X, Y, np.zeros_like(X),
        facecolors=fire_cmap(fire),
        alpha=0.8, shade=False, zorder=0
    )

    # 绘制UAV路径
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        pz = [p[2] for p in path]

        # 路径线
        ax.plot(px, py, pz, color=VIS_CONFIG["path_color"],
                linewidth=2, alpha=0.8, label='UAV路径')

        # 起点标记
        ax.scatter(*path[0], color='blue', s=150, marker='^',
                   label='起点', zorder=5, edgecolors='black')

        # 终点标记
        ax.scatter(*path[-1], color='red', s=150, marker='v',
                   label='终点', zorder=5, edgecolors='black')

        # UAV当前位置
        ax.scatter(*path[-1], color=VIS_CONFIG["uav_color"], s=200,
                   marker='o', label='UAV', zorder=5, edgecolors='black')

        # 路径上的点（稀疏显示）
        step = max(1, len(path) // 20)
        for i in range(0, len(path), step):
            ax.scatter(path[i][0], path[i][1], path[i][2],
                       color=VIS_CONFIG["uav_color"], s=20, alpha=0.5)

        # 垂直投影线（阴影）
        ax.plot(px, py, [0] * len(pz),
                color='gray', linewidth=0.5, alpha=0.3, linestyle='--')

    ax.set_xlim(0, grid_w - 1)
    ax.set_ylim(0, grid_h - 1)
    ax.set_zlim(0, env.max_z + 1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('高度 Z', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=35, azim=45)

    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"  [OK] 3D path saved: {save_path}")
    plt.close()


def generate_episode_video(env, model_or_agent, video_path,
                            n_episodes=1, is_sb3_model=True, title_prefix=""):
    """生成UAV监测过程的视频"""
    fire_cmap = create_fire_colormap()
    frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 300)
        path = [tuple(env.uav_pos)]
        fire_maps = [env.fire_map.copy()]
        cov_maps = [env.coverage_map.copy()]

        for step in range(env.max_steps):
            if is_sb3_model:
                action, _ = model_or_agent.predict(obs, deterministic=True)
            else:
                action, _ = model_or_agent.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            path.append(tuple(env.uav_pos))
            fire_maps.append(env.fire_map.copy())
            cov_maps.append(env.coverage_map.copy())

            if terminated or truncated:
                break

        # 生成每一帧图像
        total_frames = len(fire_maps)
        frame_step = max(1, total_frames // 200)  # 最多200帧

        for idx in range(0, total_frames, frame_step):
            frame = _render_frame(env, fire_maps[idx], cov_maps[idx],
                                  path[:idx + 1], fire_cmap)
            frames.append(frame)

    # 保存视频
    if frames:
        os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
        imageio.mimsave(video_path, frames, fps=VIS_CONFIG["fps"])
        print(f"  [OK] Video saved: {video_path} ({len(frames)} frames)")
    return frames


def _render_frame(env, fire_map, cov_map, path, fire_cmap):
    """渲染单帧图像"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    grid_w, grid_h = env.grid_w, env.grid_h

    # ===== 左图：火场 + UAV位置 =====
    ax1 = axes[0]
    # 火势热力图
    im1 = ax1.imshow(fire_map, cmap=fire_cmap, vmin=0, vmax=1,
                      origin='lower', extent=[0, grid_w, 0, grid_h])

    # 覆盖区域边界（半透明绿色叠加）
    cov_overlay = np.zeros((*fire_map.shape, 4))
    cov_overlay[cov_map > 0.5] = [0, 1, 0, 0.15]  # 绿色半透明
    ax1.imshow(cov_overlay, origin='lower', extent=[0, grid_w, 0, grid_h])

    # UAV路径
    if path:
        px = [p[0] + 0.5 for p in path]
        py = [p[1] + 0.5 for p in path]
        ax1.plot(px, py, color=VIS_CONFIG["path_color"],
                 linewidth=1.5, alpha=0.7)

        # UAV当前位置
        cx, cy, cz = path[-1]
        ax1.scatter(cx + 0.5, cy + 0.5, color=VIS_CONFIG["uav_color"],
                    s=100, marker='o', edgecolors='black', zorder=5)

        # 观测范围圆圈
        obs_radius = env.cfg["obs_base_radius"] + cz * env.cfg["obs_radius_per_z"]
        circle = plt.Circle((cx + 0.5, cy + 0.5), obs_radius,
                             fill=False, color=VIS_CONFIG["obs_color"],
                             linewidth=1.5, linestyle='--', alpha=0.7)
        ax1.add_patch(circle)

    ax1.set_xlim(0, grid_w)
    ax1.set_ylim(0, grid_h)
    ax1.set_title(f'火场态势 (步数: {len(path) - 1})', fontsize=13)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='火势强度', shrink=0.8)

    # ===== 右图：3D视图 =====
    ax2 = fig.add_subplot(122, projection='3d')

    # 火场地面
    X, Y = np.meshgrid(range(grid_w), range(grid_h))
    ax2.plot_surface(X, Y, np.zeros_like(X),
                     facecolors=fire_cmap(fire_map),
                     alpha=0.7, shade=False)

    # UAV 3D路径
    if path:
        px3 = [p[0] for p in path]
        py3 = [p[1] for p in path]
        pz3 = [p[2] for p in path]
        ax2.plot(px3, py3, pz3, color=VIS_CONFIG["path_color"],
                 linewidth=2, alpha=0.8)
        ax2.scatter(*path[-1], color=VIS_CONFIG["uav_color"],
                    s=100, marker='o', edgecolors='black', zorder=5)

    ax2.set_xlim(0, grid_w - 1)
    ax2.set_ylim(0, grid_h - 1)
    ax2.set_zlim(0, env.max_z + 1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('高度')
    ax2.set_title('3D视图', fontsize=13)
    ax2.view_init(elev=35, azim=45 + len(path) * 0.3)  # 缓慢旋转

    plt.tight_layout()

    # 转换为图像数组
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return frame


def generate_comparison_video(env, model, agents, video_path, seed=300):
    """生成多算法对比视频"""
    fire_cmap = create_fire_colormap()
    frames = []

    # 先跑所有算法收集数据
    all_data = {}

    # PPO
    obs, _ = env.reset(seed=seed)
    path, fire_maps = [tuple(env.uav_pos)], [env.fire_map.copy()]
    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        path.append(tuple(env.uav_pos))
        fire_maps.append(env.fire_map.copy())
        if terminated or truncated:
            break
    all_data["PPO"] = {"path": path, "fire_maps": fire_maps}

    # 基线
    for agent in agents:
        obs, _ = env.reset(seed=seed)
        path, fire_maps = [tuple(env.uav_pos)], [env.fire_map.copy()]
        for step in range(env.max_steps):
            action, _ = agent.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            path.append(tuple(env.uav_pos))
            fire_maps.append(env.fire_map.copy())
            if terminated or truncated:
                break
        all_data[agent.name] = {"path": path, "fire_maps": fire_maps}

    # 生成对比帧
    max_len = max(len(d["fire_maps"]) for d in all_data.values())
    frame_step = max(1, max_len // 150)

    for idx in range(0, max_len, frame_step):
        frame = _render_comparison_frame(env, all_data, idx, fire_cmap)
        frames.append(frame)

    if frames:
        imageio.mimsave(video_path, frames, fps=VIS_CONFIG["fps"])
        print(f"  [OK] Comparison video saved: {video_path}")
    return frames


def _render_comparison_frame(env, all_data, idx, fire_cmap):
    """渲染多算法对比单帧"""
    n_agents = len(all_data)
    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5))
    if n_agents == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, all_data.items()):
        fire_maps = data["fire_maps"]
        path = data["path"]

        fi = min(idx, len(fire_maps) - 1)
        pi = min(idx, len(path) - 1)

        ax.imshow(fire_maps[fi], cmap=fire_cmap, vmin=0, vmax=1,
                  origin='lower', extent=[0, env.grid_w, 0, env.grid_h])

        if path:
            px = [p[0] + 0.5 for p in path[:pi + 1]]
            py = [p[1] + 0.5 for p in path[:pi + 1]]
            ax.plot(px, py, color=VIS_CONFIG["path_color"],
                    linewidth=1.5, alpha=0.7)
            ax.scatter(path[pi][0] + 0.5, path[pi][1] + 0.5,
                       color=VIS_CONFIG["uav_color"], s=80,
                       marker='o', edgecolors='black', zorder=5)

        ax.set_title(f'{name} (步:{idx})', fontsize=12)
        ax.set_xlim(0, env.grid_w)
        ax.set_ylim(0, env.grid_h)

    plt.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return frame


def plot_training_curves(log_dir="results/logs", output_dir="results/plots"):
    """绘制训练曲线"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [WARN] tensorboard not installed, skipping training curves")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 查找事件文件
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, f))

    if not event_files:
        print("  [WARN] No training log files found")
        return

    # 读取训练数据
    for ef in event_files[:1]:  # 只读最新的
        ea = EventAccumulator(ef)
        ea.Reload()

        tags = ea.Tags().get('scalars', [])
        if 'rollout/ep_rew_mean' in tags:
            events = ea.Scalars('rollout/ep_rew_mean')
            steps = [e.step for e in events]
            rewards = [e.value for e in events]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, rewards, linewidth=1.5, color='#FF6B6B')
            ax.set_title('训练奖励曲线', fontsize=14)
            ax.set_xlabel('训练步数', fontsize=12)
            ax.set_ylabel('平均回合奖励', fontsize=12)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_curve.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Training curve saved")


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from baselines import RandomAgent, GreedyAgent, AStarAgent

    model_path = "results/models/ppo_wildfire_best/best_model"
    env = WildfireEnv()

    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print("生成PPO监测视频...")
        generate_episode_video(env, model, "results/videos/ppo_wildfire.gif",
                               is_sb3_model=True)

        print("生成对比视频...")
        agents = [RandomAgent(env), GreedyAgent(env), AStarAgent(env)]
        generate_comparison_video(env, model, agents,
                                  "results/videos/comparison.gif")
    else:
        print("[ERROR] Model not found, please train first")
