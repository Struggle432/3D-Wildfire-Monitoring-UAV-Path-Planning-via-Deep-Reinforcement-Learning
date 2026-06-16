"""
3D野火监测环境 (Gymnasium)
- 3D空间中无人机运动 (x, y, z)
- 2D地面火势蔓延 (元胞自动机)
- 有限视角观测模型 (高度影响观测范围和精度)
- 多目标奖励函数 (信息增益/前沿覆盖/运动成本/安全风险/遗漏惩罚/高度权衡)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import ENV_CONFIG, REWARD_CONFIG, ACTION_DELTA, HORIZONTAL_ACTIONS


class WildfireEnv(gym.Env):
    """3D野火场景下无人机监测路径规划环境"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_config=None, reward_config=None, render_mode=None):
        super().__init__()

        # 加载配置
        self.cfg = {**ENV_CONFIG, **(env_config or {})}
        self.rw_cfg = {**REWARD_CONFIG, **(reward_config or {})}

        # 环境基本参数
        self.grid_w = self.cfg["grid_w"]
        self.grid_h = self.cfg["grid_h"]
        self.max_z = self.cfg["max_z"]
        self.max_steps = self.cfg["max_steps"]

        # 动作空间：7个离散动作
        self.action_space = spaces.Discrete(7)

        # 观测空间设计 (固定维度向量)
        # - UAV位置: 3 (x, y, z 归一化)
        # - 局部火势图: obs_size^2
        # - 局部覆盖图: obs_size^2
        # - 全局火势概览: global_size^2 (下采样)
        # - 风场信息: 2 (wind_x, wind_y)
        # - 标量特征: 5 (火前沿距离, 覆盖率, 未观测火比例, 上一步动作, 时间步)
        self.obs_map_size = self.cfg["obs_map_size"]  # 9
        self.global_map_size = 10  # 全局下采样尺寸
        obs_dim = (3 + self.obs_map_size**2 + self.obs_map_size**2
                   + self.global_map_size**2 + 2 + 5)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.obs_dim = obs_dim

        # 渲染模式
        self.render_mode = render_mode

        # 环境状态 (在reset中初始化)
        self.uav_pos = None       # (x, y, z)
        self.fire_map = None      # (grid_h, grid_w) 火势强度 0~1
        self.coverage_map = None   # (grid_h, fire_w) 是否被观测过
        self.wind_dir = None      # 风向向量
        self.wind_speed = None    # 风速
        self.step_count = 0
        self.prev_action = 6      # 初始为悬停
        self.total_reward = 0.0

        # 统计信息
        self.info_stats = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化无人机位置
        start = self.cfg["uav_start"]
        self.uav_pos = list(start)
        self.prev_action = 6
        self.step_count = 0
        self.total_reward = 0.0

        # 初始化火势地图 (全0)
        self.fire_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # 设置火源
        n_sources = self.cfg["n_fire_sources"]
        intensity = self.cfg["fire_source_intensity"]
        rng = np.random.RandomState(seed if seed is not None else 42)

        for _ in range(n_sources):
            # 火源在地图中偏中心区域
            fx = rng.randint(self.grid_w // 4, 3 * self.grid_w // 4)
            fy = rng.randint(self.grid_h // 4, 3 * self.grid_h // 4)
            # 火源占3x3区域
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = fy + di, fx + dj
                    if 0 <= ni < self.grid_h and 0 <= nj < self.grid_w:
                        dist = abs(di) + abs(dj)
                        self.fire_map[ni, nj] = intensity * (1.0 - 0.2 * dist)

        # 初始化覆盖地图
        self.coverage_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # 初始化风场
        wd = self.cfg["wind_direction"]
        norm = np.sqrt(wd[0]**2 + wd[1]**2) + 1e-8
        self.wind_dir = np.array([wd[0] / norm, wd[1] / norm], dtype=np.float32)
        self.wind_speed = self.cfg["wind_speed"]

        # 初始观测
        obs = self._get_obs()

        # 统计信息
        self.info_stats = {
            "total_fire_cells": 0,
            "total_covered_fire": 0,
            "min_dist_to_fire": float('inf'),
            "path": [tuple(self.uav_pos)],
            "fire_history": [self.fire_map.copy()],
            "coverage_history": [self.coverage_map.copy()],
            "reward_components": [],
        }

        return obs, {}

    def step(self, action):
        # ===== 1. 执行UAV动作 =====
        action = int(action)  # 确保action是Python int
        dx, dy, dz = ACTION_DELTA[action]
        new_x = np.clip(self.uav_pos[0] + dx, 0, self.grid_w - 1)
        new_y = np.clip(self.uav_pos[1] + dy, 0, self.grid_h - 1)
        new_z = np.clip(self.uav_pos[2] + dz, 1, self.max_z)

        self.uav_pos = [int(new_x), int(new_y), int(new_z)]

        # ===== 2. 更新观测覆盖 =====
        old_coverage = self.coverage_map.copy()
        self._update_coverage()

        # ===== 3. 火势蔓延 =====
        self._spread_fire()

        # ===== 4. 计算奖励 =====
        reward, rw_components = self._compute_reward(action, old_coverage)

        # ===== 5. 更新状态 =====
        self.prev_action = action
        self.step_count += 1
        self.total_reward += reward

        # ===== 6. 记录统计 =====
        self.info_stats["path"].append(tuple(self.uav_pos))
        self.info_stats["fire_history"].append(self.fire_map.copy())
        self.info_stats["coverage_history"].append(self.coverage_map.copy())
        self.info_stats["reward_components"].append(rw_components)

        # ===== 7. 判断终止 =====
        terminated = False
        truncated = False

        # 安全终止：UAV距离火源过近且高度很低
        fire_dist = self._min_distance_to_fire()
        if fire_dist < self.cfg["danger_distance"] and self.uav_pos[2] <= 2:
            terminated = True
            reward -= 5.0  # 坠毁惩罚

        # 超时截断
        if self.step_count >= self.max_steps:
            truncated = True

        # 成功：覆盖了足够多的火区
        fire_coverage = self._fire_coverage_rate()
        if fire_coverage > 0.9 and self.step_count > 20:
            terminated = True
            reward += self.rw_cfg["arrival_bonus"]

        # ===== 8. 获取观测 =====
        obs = self._get_obs()

        info = {
            "fire_coverage": fire_coverage,
            "min_dist_to_fire": fire_dist,
            "step": self.step_count,
            "reward_components": rw_components,
            "uav_pos": tuple(self.uav_pos),
        }

        return obs, float(reward), terminated, truncated, info

    def _update_coverage(self):
        """根据UAV当前位置和高度更新覆盖地图"""
        x, y, z = self.uav_pos
        # 观测半径随高度增加
        obs_radius = self.cfg["obs_base_radius"] + z * self.cfg["obs_radius_per_z"]

        # 标记观测范围内的地面格子
        for dy in range(-obs_radius, obs_radius + 1):
            for dx in range(-obs_radius, obs_radius + 1):
                # 圆形观测区域
                if dx**2 + dy**2 <= obs_radius**2:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        # 高空观测有概率遗漏（模拟精度下降）
                        if z <= 3 or np.random.random() < 0.85:
                            self.coverage_map[ny, nx] = 1.0

    def _spread_fire(self):
        """元胞自动机火势蔓延模型"""
        new_fire = self.fire_map.copy()
        decay = self.cfg["fire_decay_rate"]
        base_prob = self.cfg["fire_spread_base"]
        wind_prob = self.cfg["fire_spread_wind"]

        for i in range(self.grid_h):
            for j in range(self.grid_w):
                if self.fire_map[i, j] > 0.05:  # 正在燃烧
                    # 火势衰减
                    new_fire[i, j] = max(0, new_fire[i, j] - decay)

                    # 尝试向邻居扩散
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.grid_h and 0 <= nj < self.grid_w:
                                if self.fire_map[ni, nj] < 0.05:  # 邻居未燃烧
                                    # 计算风向对扩散的影响
                                    # (di,dj)是蔓延方向，与风向越一致概率越高
                                    dir_len = np.sqrt(di**2 + dj**2)
                                    alignment = (dj * self.wind_dir[0] + di * self.wind_dir[1]) / dir_len

                                    # 扩散概率
                                    prob = base_prob + wind_prob * self.wind_speed * alignment
                                    prob = max(0.01, min(0.5, prob))  # 限制范围

                                    # 对角线方向概率降低
                                    if abs(di) + abs(dj) == 2:
                                        prob *= 0.7

                                    if np.random.random() < prob * self.fire_map[i, j]:
                                        new_fire[ni, nj] = max(new_fire[ni, nj],
                                                               0.8 + 0.2 * np.random.random())

        self.fire_map = np.clip(new_fire, 0, 1.0)

    def _compute_reward(self, action, old_coverage):
        """多目标奖励函数"""
        rw = {}

        # 1. 信息增益：本轮新观测到的燃烧格子数
        newly_observed = (self.coverage_map - old_coverage) > 0
        new_fire_observed = np.sum(newly_observed & (self.fire_map > 0.1))
        rw["info_gain"] = float(new_fire_observed)

        # 2. 前沿覆盖：观测到火势前沿（燃烧区边界）的格子数
        frontier = self._get_fire_frontier()
        frontier_observed = np.sum(frontier & (self.coverage_map > 0))
        rw["frontier"] = float(frontier_observed) / max(1, float(np.sum(frontier)))

        # 3. 运动成本
        move_cost = 0.0
        if action == 6:  # 悬停
            move_cost = 0.02
        elif action in {4, 5}:  # 升降
            move_cost = 0.2
        else:  # 水平移动
            move_cost = 0.1
            # 转弯惩罚：水平方向改变
            if self.prev_action in HORIZONTAL_ACTIONS and action in HORIZONTAL_ACTIONS:
                if action != self.prev_action and not self._is_same_direction(action, self.prev_action):
                    move_cost += 0.1  # 转弯额外成本
        rw["move_cost"] = move_cost

        # 4. 安全风险
        fire_dist = self._min_distance_to_fire()
        danger_dist = self.cfg["danger_distance"]
        if fire_dist < danger_dist:
            safety = np.exp(-fire_dist) * (1.0 + (danger_dist - fire_dist))
        else:
            safety = 0.0
        # 低空更危险
        if self.uav_pos[2] <= 2:
            safety *= 2.0
        rw["safety"] = safety

        # 5. 遗漏惩罚：未观测到的燃烧格子比例
        total_fire = np.sum(self.fire_map > 0.1)
        unobserved_fire = np.sum((self.fire_map > 0.1) & (self.coverage_map < 0.5))
        rw["miss"] = float(unobserved_fire) / max(1, float(total_fire))

        # 6. 高度奖励：鼓励在中等高度观测
        z = self.uav_pos[2]
        # 最佳高度在3~5层，过高信息模糊，过低危险
        optimal_z = 4.0
        alt_bonus = np.exp(-0.3 * (z - optimal_z)**2)
        rw["altitude"] = alt_bonus

        # 计算总奖励
        reward = (
            self.rw_cfg["alpha"] * rw["info_gain"]
            + self.rw_cfg["beta"] * rw["frontier"]
            - self.rw_cfg["gamma"] * rw["move_cost"]
            - self.rw_cfg["delta"] * rw["safety"]
            - self.rw_cfg["epsilon"] * rw["miss"]
            + self.rw_cfg["zeta"] * rw["altitude"]
            - self.rw_cfg["eta"]
        )

        rw["total"] = reward
        return reward, rw

    def _is_same_direction(self, a1, a2):
        """判断两个水平动作是否同方向（直行 vs 转弯）"""
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        return a1 == a2 or opposites.get(a1) == a2

    def _get_fire_frontier(self):
        """获取火势前沿（燃烧区边界）"""
        burning = self.fire_map > 0.1
        frontier = np.zeros_like(burning)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                shifted = np.zeros_like(burning)
                si = slice(max(0, -di), self.grid_h - max(0, di))
                sj = slice(max(0, -dj), self.grid_w - max(0, dj))
                di2 = slice(max(0, di), self.grid_h - max(0, -di))
                dj2 = slice(max(0, dj), self.grid_w - max(0, -dj))
                shifted[di2, dj2] = burning[si, sj]
                frontier |= (shifted & ~burning)
        return frontier

    def _min_distance_to_fire(self):
        """计算UAV到最近燃烧格子的距离"""
        x, y, z = self.uav_pos
        burning_y, burning_x = np.where(self.fire_map > 0.1)
        if len(burning_x) == 0:
            return float(self.grid_w)
        # 3D欧氏距离（z轴权重降低，因为火在地面上）
        dists = np.sqrt((burning_x - x)**2 + (burning_y - y)**2 + z**2 * 0.3)
        return float(np.min(dists))

    def _fire_coverage_rate(self):
        """已观测的燃烧格子占比"""
        total_fire = np.sum(self.fire_map > 0.1)
        if total_fire == 0:
            return 0.0
        covered_fire = np.sum((self.fire_map > 0.1) & (self.coverage_map > 0.5))
        return float(covered_fire) / float(total_fire)

    def _get_obs(self):
        """构建观测向量"""
        x, y, z = self.uav_pos
        obs_list = []

        # 1. UAV位置 (归一化)
        obs_list.append(x / (self.grid_w - 1))
        obs_list.append(y / (self.grid_h - 1))
        obs_list.append(z / self.max_z)

        # 2. 局部火势图 (obs_map_size x obs_map_size)
        local_fire = self._get_local_map(self.fire_map, x, y, z)
        obs_list.extend(local_fire.flatten())

        # 3. 局部覆盖图
        local_cov = self._get_local_map(self.coverage_map, x, y, z)
        obs_list.extend(local_cov.flatten())

        # 4. 全局火势概览 (下采样到 global_map_size x global_map_size)
        global_fire = self._downsample(self.fire_map, self.global_map_size)
        obs_list.extend(global_fire.flatten())

        # 5. 风场信息
        obs_list.append(float(self.wind_dir[0]) * self.wind_speed)
        obs_list.append(float(self.wind_dir[1]) * self.wind_speed)

        # 6. 标量特征
        # 到火前沿的距离
        fire_dist = self._min_distance_to_fire()
        obs_list.append(np.clip(fire_dist / self.grid_w, 0, 1))

        # 火区覆盖率
        obs_list.append(self._fire_coverage_rate())

        # 未观测火区比例
        total_fire = np.sum(self.fire_map > 0.1)
        unobserved = np.sum((self.fire_map > 0.1) & (self.coverage_map < 0.5))
        obs_list.append(float(unobserved) / max(1, float(total_fire)))

        # 上一步动作
        obs_list.append(self.prev_action / 6.0)

        # 时间步
        obs_list.append(self.step_count / self.max_steps)

        obs = np.array(obs_list, dtype=np.float32)
        # 确保维度匹配
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        return np.clip(obs, -1.0, 1.0)

    def _get_local_map(self, grid_map, cx, cy, z):
        """获取以UAV为中心的局部地图，观测范围随高度增加但精度降低"""
        size = self.obs_map_size  # 9
        local = np.zeros((size, size), dtype=np.float32)

        # 观测半径：高度越高，看得越远
        obs_radius = self.cfg["obs_base_radius"] + z * self.cfg["obs_radius_per_z"]

        # 步长：高度越高，采样越稀疏（精度降低）
        stride = max(1, z // 2)

        half = size // 2
        for i in range(size):
            for j in range(size):
                # 局部坐标到全局坐标
                gi = cy + (i - half) * stride
                gj = cx + (j - half) * stride
                if 0 <= gi < self.grid_h and 0 <= gj < self.grid_w:
                    # 检查是否在观测半径内
                    dist = np.sqrt(((i - half) * stride)**2 + ((j - half) * stride)**2)
                    if dist <= obs_radius:
                        local[i, j] = grid_map[gi, gj]
                    else:
                        local[i, j] = -1.0  # 超出观测范围标记
                else:
                    local[i, j] = -1.0  # 超出边界标记

        return local

    def _downsample(self, grid_map, target_size):
        """将地图下采样到目标尺寸"""
        h, w = grid_map.shape
        result = np.zeros((target_size, target_size), dtype=np.float32)
        bh = h / target_size
        bw = w / target_size
        for i in range(target_size):
            for j in range(target_size):
                si = int(i * bh)
                ei = int((i + 1) * bh)
                sj = int(j * bw)
                ej = int((j + 1) * bw)
                block = grid_map[si:min(ei, h), sj:min(ej, w)]
                result[i, j] = np.mean(block) if block.size > 0 else 0.0
        return result

    def get_fire_map(self):
        """获取当前火势地图（用于可视化）"""
        return self.fire_map.copy()

    def get_coverage_map(self):
        """获取当前覆盖地图（用于可视化）"""
        return self.coverage_map.copy()

    def get_uav_pos(self):
        """获取UAV位置"""
        return tuple(self.uav_pos)
