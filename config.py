"""
项目配置文件：基于深度强化学习的3D野火场景下无人机监测路径规划
所有超参数、环境参数、奖励权重、训练参数集中管理
"""

import numpy as np

# ======================== 环境参数 ========================
ENV_CONFIG = {
    # 网格地图尺寸
    "grid_w": 30,               # 地面网格宽度
    "grid_h": 30,               # 地面网格高度
    "max_z": 8,                 # 最大高度层 (1~8)

    # 无人机参数
    "uav_start": (2, 2, 4),     # 起始位置 (x, y, z)
    "obs_base_radius": 3,       # 基础观测半径（z=1时）
    "obs_radius_per_z": 1,      # 每增加1层高度增加的观测半径
    "obs_map_size": 9,          # 局部观测地图尺寸 (9x9)

    # 火势参数
    "fire_decay_rate": 0.015,   # 火势衰减速率
    "fire_spread_base": 0.12,   # 基础扩散概率
    "fire_spread_wind": 0.15,   # 风向加成扩散概率
    "n_fire_sources": 2,        # 初始火源数量
    "fire_source_intensity": 1.0, # 初始火源强度

    # 风场参数
    "wind_speed": 0.6,          # 风速 (0~1)
    "wind_direction": (1.0, 0.5), # 风向 (归一化向量)

    # 回合参数
    "max_steps": 300,           # 最大步数
    "danger_distance": 2,       # 危险距离阈值（UAV与火源的最小安全距离）
}

# ======================== 奖励函数权重 ========================
REWARD_CONFIG = {
    "alpha": 2.0,       # 信息增益权重（观测到新的燃烧区域）
    "beta": 1.0,        # 前沿覆盖权重（观测火势前沿）
    "gamma": 0.15,      # 运动成本权重（转弯>直行，升降>平移）
    "delta": 3.0,       # 安全风险权重（距火源过近的惩罚）
    "epsilon": 0.05,    # 遗漏惩罚权重（未观测到的燃烧区域）
    "zeta": 0.3,        # 高度奖励权重（适当高度的奖励）
    "eta": 0.02,        # 时间步惩罚（鼓励高效行动）
    "arrival_bonus": 10.0,  # 成功覆盖奖励
}

# ======================== 训练参数 ========================
TRAIN_CONFIG = {
    "algorithm": "PPO",
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_envs": 4,                # 并行环境数
    "net_arch_pi": [256, 256],  # 策略网络结构
    "net_arch_vf": [256, 256],  # 价值网络结构
    "save_freq": 50_000,        # 模型保存频率
    "eval_freq": 10_000,        # 评估频率
    "log_dir": "results/logs",
    "model_dir": "results/models",
}

# ======================== 评估参数 ========================
EVAL_CONFIG = {
    "n_eval_episodes": 20,      # 评估回合数
    "scenarios": [
        {"name": "单火源无风", "n_fire_sources": 1, "wind_speed": 0.0, "wind_direction": (1, 0)},
        {"name": "单火源有风", "n_fire_sources": 1, "wind_speed": 0.6, "wind_direction": (1, 0.5)},
        {"name": "双火源有风", "n_fire_sources": 2, "wind_speed": 0.6, "wind_direction": (1, 0.5)},
        {"name": "三火源强风", "n_fire_sources": 3, "wind_speed": 0.9, "wind_direction": (0.7, 1.0)},
    ],
}

# ======================== 可视化参数 ========================
VIS_CONFIG = {
    "fps": 8,                   # 视频帧率
    "dpi": 150,                 # 图像分辨率
    "figsize_3d": (12, 9),      # 3D图尺寸
    "figsize_2d": (10, 8),      # 2D图尺寸
    # 火势颜色映射
    "fire_cmap": "hot",
    "uav_color": "#00BFFF",     # 无人机颜色（深天蓝）
    "path_color": "#00FF7F",    # 路径颜色（春绿）
    "obs_color": "#87CEEB",     # 观测区域颜色（天蓝）
    "danger_color": "#FF4500",  # 危险区域颜色
}

# ======================== 动作映射 ========================
ACTION_MEANING = {
    0: "北 (y-1)",
    1: "南 (y+1)",
    2: "西 (x-1)",
    3: "东 (x+1)",
    4: "上升 (z+1)",
    5: "下降 (z-1)",
    6: "悬停观测",
}

# 动作对位移的影响
ACTION_DELTA = {
    0: (0, -1, 0),   # 北
    1: (0,  1, 0),   # 南
    2: (-1, 0, 0),   # 西
    3: ( 1, 0, 0),   # 东
    4: (0,  0, 1),   # 上升
    5: (0,  0,-1),   # 下降
    6: (0,  0, 0),   # 悬停
}

# 水平动作集合（用于判断转弯）
HORIZONTAL_ACTIONS = {0, 1, 2, 3}
