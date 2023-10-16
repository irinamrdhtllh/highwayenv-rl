import time

import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam

from ..actor_critic import MLPActorCritic
from ..helper import combined_shape, discount_cumsum


class Buffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buffer = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buffer = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buffer = np.zeros(size, dtype=np.float32)
        self.rew_buffer = np.zeros(size, dtype=np.float32)
        self.ret_buffer = np.zeros(size, dtype=np.float32)
        self.val_buffer = np.zeros(size, dtype=np.float32)
        self.logp_buffer = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam

        self.idx, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.idx < self.max_size
        self.obs_buffer[self.idx] = obs
        self.act_buffer[self.idx] = act
        self.rew_buffer[self.idx] = rew
        self.val_buffer[self.idx] = val
        self.logp_buffer[self.idx] = logp

        self.idx += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.idx)
        rewards = np.append(self.rew_buffer[path_slice], last_val)
        values = np.append(self.val_buffer[path_slice], last_val)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buffer[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        self.ret_buffer[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.idx

    def get(self):
        assert self.idx == self.max_size
        self.idx, self.path_start_idx = 0, 0

        data = dict(
            obs=self.obs_buffer,
            act=self.act_buffer,
            ret=self.ret_buffer,
            adv=self.adv_buffer,
            logp=self.logp_buffer,
        )

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(
    env,
    env_name,
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4_000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.95,
    max_ep_len=1_000,
    target_kl=0.01,
    save_freq=10,
):
    seed += 10_000
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    local_steps_per_epoch = int(steps_per_epoch)
    buffer = Buffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.get(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clip_frac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        loss_v = ((ac.v(obs) - ret) ** 2).mean()
        return loss_v

    def update():
        data = buffer.get()

        pi_loss_old, pi_info_old = compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = compute_loss_v(data).item()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info["kl"]
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

    start_time = time.time()
    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            act, val, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))

            next_obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            ep_ret += rew
            ep_len += 1

            buffer.store(obs, act, rew, val, logp)

            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                if timeout or epoch_ended:
                    _, val, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    val = 0
                buffer.finish_path(val)
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs):
            torch.save(ac.state_dict, f"models/non_library/{env_name}_ppo.pth")

        update()
