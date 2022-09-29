import random

from envs.SimpleATC_env_five_v2 import SimpleEnv

env = SimpleEnv()
env.reset()
for _ in range(5):
    done = False
    episode_timestep = 0
    while not done and episode_timestep < 500:
        action = random.randint(0, 8)
        _, _, done, _ = env.step(action)
        env.render()
