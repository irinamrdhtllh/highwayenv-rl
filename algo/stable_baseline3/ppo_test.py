import gymnasium as gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    try:
        model = PPO.load("highway_ppo/model")
        env = gym.make("highway-fast-v0", render_mode="human")
        for _ in range(5):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
    except EOFError as e:
        print(e)
