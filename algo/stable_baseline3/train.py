import gymnasium as gym
import highway_env
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.nn import functional as F

if __name__ == "__main__":
    try:
        env_name = "highway-v0"
        algo_name = "ppo"
        train = True
        if train:
            n_cpu = 1
            batch_size = 64
            env = make_vec_env(
                env_name,
                seed=1,
                n_envs=n_cpu,
                vec_env_cls=SubprocVecEnv,
                env_kwargs={"render_mode": "human"},
            )
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.8,
                verbose=2,
                tensorboard_log=f"models/stable_baseline3/{env_name}_{algo_name}/",
            )
            # Train the agent
            model.learn(total_timesteps=int(2e5))
            # Save the agent
            model.save(f"models/stable_baseline3/{env_name}_{algo_name}/model")
    except EOFError as e:
        print(e)
