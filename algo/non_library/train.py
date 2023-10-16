import argparse

from gymnasium.wrappers import FlattenObservation
from vpg.vpg import vpg

from env.highway_v0_env import create_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--maxlen", type=int, default=1000)
    parser.add_argument("--savefreq", type=float, default=10)

    args = parser.parse_args()

    try:
        env, env_name = create_env()
        env = FlattenObservation(env)

        vpg(
            env=env,
            env_name=env_name,
            ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
            seed=args.seed,
            steps_per_epoch=args.steps,
            epochs=args.epochs,
            gamma=args.gamma,
            max_ep_len=args.maxlen,
            save_freq=args.savefreq,
        )
    except EOFError as e:
        print(e)
