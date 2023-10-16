import torch

from env.highway_v0_env import create_env

if __name__ == "__main__":
    try:
        env, env_name = create_env()
        model = torch.load(f"models/non_library/{env_name}_vpg.pth")
        for _ in range(5):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
    except EOFError as e:
        print(e)
