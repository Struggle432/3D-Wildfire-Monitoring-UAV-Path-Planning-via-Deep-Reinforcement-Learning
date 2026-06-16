"""
基线算法
- RandomAgent: 随机游走
- GreedyAgent: 贪心策略（向最近未观测火区移动）
- AStarAgent: A*路径规划（静态规划，逐步重规划）
"""

import numpy as np
import heapq
from wildfire_env import WildfireEnv
from config import ENV_CONFIG, REWARD_CONFIG


class RandomAgent:
    """随机策略基线"""

    def __init__(self, env):
        self.env = env
        self.name = "随机游走"

    def predict(self, obs, deterministic=True):
        return self.env.action_space.sample(), None


class GreedyAgent:
    """贪心策略：每步选择使信息增益最大的动作"""

    def __init__(self, env):
        self.env = env
        self.name = "贪心策略"

    def predict(self, obs, deterministic=True):
        """评估每个动作的信息增益，选择最优"""
        best_action = 6  # 默认悬停
        best_score = -float('inf')

        x, y, z = self.env.uav_pos
        current_fire_coverage = self.env._fire_coverage_rate()

        for action in range(7):
            score = self._evaluate_action(action, x, y, z)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action, None

    def _evaluate_action(self, action, x, y, z):
        """评估单个动作的期望收益"""
        from config import ACTION_DELTA
        dx, dy, dz = ACTION_DELTA[action]
        nx = np.clip(x + dx, 0, self.env.grid_w - 1)
        ny = np.clip(y + dy, 0, self.env.grid_h - 1)
        nz = np.clip(z + dz, 1, self.env.max_z)

        # 计算新位置的观测范围内的火区
        obs_radius = self.env.cfg["obs_base_radius"] + nz * self.env.cfg["obs_radius_per_z"]
        fire_in_view = 0
        unobserved_fire_in_view = 0

        for ddy in range(-obs_radius, obs_radius + 1):
            for ddx in range(-obs_radius, obs_radius + 1):
                if ddx**2 + ddy**2 <= obs_radius**2:
                    gx, gy = int(nx + ddx), int(ny + ddy)
                    if 0 <= gx < self.env.grid_w and 0 <= gy < self.env.grid_h:
                        if self.env.fire_map[gy, gx] > 0.1:
                            fire_in_view += 1
                            if self.env.coverage_map[gy, gx] < 0.5:
                                unobserved_fire_in_view += 1

        # 评估得分：优先未观测火区，其次已观测火区
        score = unobserved_fire_in_view * 2.0 + fire_in_view * 0.5

        # 安全惩罚
        burning_y, burning_x = np.where(self.env.fire_map > 0.1)
        if len(burning_x) > 0:
            min_dist = np.min(np.sqrt((burning_x - nx)**2 + (burning_y - ny)**2 + nz**2 * 0.3))
            if min_dist < self.env.cfg["danger_distance"] and nz <= 2:
                score -= 10.0

        # 高度偏好
        score -= 0.1 * abs(nz - 4)

        return score


class AStarAgent:
    """A*路径规划基线：规划到最近未观测火区，逐步重规划"""

    def __init__(self, env):
        self.env = env
        self.name = "A*规划"
        self.path = []
        self.path_idx = 0

    def predict(self, obs, deterministic=True):
        """每步重规划到最近的未观测火区"""
        x, y, z = self.env.uav_pos

        # 找到最近的未观测火区目标
        target = self._find_nearest_unobserved_fire(x, y, z)

        if target is None:
            # 没有未观测火区，悬停
            return 6, None

        tx, ty = target

        # 简化A*：在2D地面上规划路径，高度保持最优
        path = self._astar_2d((x, y), (tx, ty))

        if path and len(path) > 1:
            next_pos = path[1]
            # 确定动作
            action = self._pos_to_action(x, y, z, next_pos[0], next_pos[1])
            return action, None

        return 6, None  # 无法规划路径，悬停

    def _find_nearest_unobserved_fire(self, x, y, z):
        """找到最近的未观测燃烧格子"""
        burning_y, burning_x = np.where(
            (self.env.fire_map > 0.1) & (self.env.coverage_map < 0.5)
        )
        if len(burning_x) == 0:
            return None

        dists = (burning_x - x)**2 + (burning_y - y)**2
        idx = np.argmin(dists)
        return int(burning_x[idx]), int(burning_y[idx])

    def _astar_2d(self, start, goal):
        """2D A*路径搜索"""
        sx, sy = start
        gx, gy = goal

        # 简单A*
        open_set = [(0, sx, sy)]
        came_from = {}
        g_score = {(sx, sy): 0}

        obstacles = self.env.fire_map > 0.7  # 强火区视为障碍

        while open_set:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                # 重建路径
                path = [(cx, cy)]
                while (cx, cy) in came_from:
                    cx, cy = came_from[(cx, cy)]
                    path.append((cx, cy))
                path.reverse()
                return path

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.env.grid_w and 0 <= ny < self.env.grid_h:
                    # 避开强火区
                    cost = 1.0
                    if obstacles[ny, nx]:
                        cost = 10.0  # 高成本但不完全禁止

                    new_g = g_score[(cx, cy)] + cost
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_g
                        h = abs(nx - gx) + abs(ny - gy)  # 曼哈顿距离
                        heapq.heappush(open_set, (new_g + h, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)

            # 限制搜索范围
            if len(g_score) > 500:
                break

        # A*失败，直走
        return [start, goal]

    def _pos_to_action(self, x, y, z, nx, ny):
        """将位置变化转换为动作"""
        dx, dy = nx - x, ny - y
        if dx == 1:
            return 3  # 东
        elif dx == -1:
            return 2  # 西
        elif dy == 1:
            return 1  # 南
        elif dy == -1:
            return 0  # 北
        return 6  # 悬停


def run_baseline(env, agent, n_episodes=10, max_steps=None):
    """运行基线算法并返回统计结果"""
    max_steps = max_steps or env.max_steps
    results = {
        "rewards": [],
        "coverages": [],
        "min_dists": [],
        "steps": [],
        "paths": [],
        "fire_histories": [],
        "coverage_histories": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 100)
        total_reward = 0
        step = 0

        for step in range(max_steps):
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # 记录结果
        results["rewards"].append(total_reward)
        results["coverages"].append(info.get("fire_coverage", 0))
        results["min_dists"].append(info.get("min_dist_to_fire", 0))
        results["steps"].append(step + 1)
        results["paths"].append(env.info_stats["path"])
        results["fire_histories"].append(env.info_stats["fire_history"])
        results["coverage_histories"].append(env.info_stats["coverage_history"])

    # 计算均值
    results["mean_reward"] = np.mean(results["rewards"])
    results["mean_coverage"] = np.mean(results["coverages"])
    results["mean_steps"] = np.mean(results["steps"])
    results["std_reward"] = np.std(results["rewards"])
    results["std_coverage"] = np.std(results["coverages"])

    return results
