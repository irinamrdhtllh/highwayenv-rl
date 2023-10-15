import gymnasium as gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    env_name = "highway-v0"
    algo_name = "ppo"
    try:
        model = PPO.load(f"models/stable_baseline3/{env_name}_{algo_name}/model")
        env = gym.make(env_name, render_mode="human")
        for _ in range(5):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
    except EOFError as e:
        print(e)
