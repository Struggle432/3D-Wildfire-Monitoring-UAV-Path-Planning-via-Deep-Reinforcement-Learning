# 基于深度强化学习的3D野火场景下无人机监测路径规划

## 项目简介

本项目使用深度强化学习（PPO算法）训练无人机在三维空间中对动态蔓延的野火进行智能监测。无人机需要自主规划飞行路径，在保证安全的前提下最大化火区覆盖率，同时最小化运动成本。

## 环境要求

- Python 3.8+
- PyTorch 2.4+ (CUDA)
- gymnasium 1.0+
- stable-baselines3 2.4+
- numpy, matplotlib, scipy, imageio, shapely

## 快速开始

### 1. 训练模型

```bash
# 激活环境
conda activate face_recognition

# 训练PPO模型（约60分钟，GPU加速）
python run_train.py
```

训练过程中会自动保存：
- 最佳模型：`results/models/ppo_wildfire_best/best_model.zip`
- 最终模型：`results/models/ppo_wildfire_final.zip`
- 训练日志：`results/logs/`（可用TensorBoard查看）

### 2. 查看训练曲线

```bash
tensorboard --logdir results/logs
```

### 3. 运行评估与可视化

```bash
# 一键运行：评估 + 生成图表 + 生成视频
python run_eval_and_viz.py
```

### 4. 跳过训练，直接评估

如果已有训练好的模型，可直接运行评估：
```bash
python run_eval_and_viz.py
```

## 输出文件

### 模型文件 (`results/models/`)
- `ppo_wildfire_best/best_model.zip` — 最佳模型
- `ppo_wildfire_final.zip` — 最终模型

### 图表 (`results/plots/`)
- `algorithm_comparison.png` — 算法对比柱状图（奖励/覆盖率/步数）
- `path_3d_ppo.png` — PPO 3D路径效果图
- `path_3d_random.png` — 随机游走3D路径
- `path_3d_greedy.png` — 贪心策略3D路径
- `path_3d_astar.png` — A*规划3D路径
- `env_diagram.png` — 环境示意图（初始火场 + 蔓延后）
- `coverage_over_time.png` — 覆盖率随时间变化曲线
- `multi_scenario.png` — 多场景评估对比

### 视频 (`results/videos/`)
- `ppo_wildfire.gif` — PPO无人机监测过程动画
- `comparison.gif` — 多算法对比动画

## 项目结构

```
work_final/
├── config.py              # 所有配置参数
├── wildfire_env.py        # 3D野火监测Gymnasium环境
├── train.py               # PPO训练脚本
├── run_train.py           # 独立训练入口
├── baselines.py           # 基线算法（随机/贪心/A*）
├── evaluate.py            # 评估与对比
├── visualize.py           # 3D可视化与视频生成
├── run_eval_and_viz.py    # 评估+可视化一键脚本
├── run_all.py             # 全流程一键脚本
├── results/               # 输出目录
│   ├── models/            # 训练好的模型
│   ├── plots/             # 图表
│   ├── videos/            # 视频/GIF
│   └── logs/              # 训练日志
└── README.md
```

## 关键设计

### 环境设计
- **3D空间**：30x30地面网格 + 8层高度
- **火势模型**：元胞自动机，受风向/风速影响
- **观测模型**：高空观测范围大但精度低，低空范围小但精度高

### 奖励函数
```
R = α·信息增益 + β·前沿覆盖 - γ·运动成本 - δ·安全风险 - ε·遗漏惩罚 + ζ·高度奖励 - η·时间惩罚
```

