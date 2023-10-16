import gymnasium as gym
import highway_env

config = {
    "observation": {"type": "Kinematics"},
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 30,
    "initial_spacing": 2,
    "collision_reward": -1,
    "reward_speed_range": [20, 30],
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
}


def create_env():
    env_name = "highway-v0"
    env = gym.make(env_name, render_mode="human")
    env.configure(config)

    return env, env_name
