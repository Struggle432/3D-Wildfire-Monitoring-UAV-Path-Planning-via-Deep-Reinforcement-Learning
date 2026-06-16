"""测试环境是否能正常运行"""
import sys
sys.path.insert(0, '.')
from wildfire_env import WildfireEnv
import numpy as np

env = WildfireEnv()
obs, info = env.reset(seed=42)
print('OK - Environment created')
print(f'  Obs space: {env.observation_space.shape}')
print(f'  Action space: {env.action_space.n}')
print(f'  Obs dim: {len(obs)}')
print(f'  UAV start: {env.uav_pos}')
print(f'  Fire cells: {np.sum(env.fire_map > 0.1)}')

total_reward = 0
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f'  10-step reward: {total_reward:.2f}')
print(f'  UAV pos after 10 steps: {env.uav_pos}')
print(f'  Fire coverage: {env._fire_coverage_rate():.2%}')
print(f'  Burning cells after 10 steps: {np.sum(env.fire_map > 0.1)}')
print('OK - Environment test passed!')
