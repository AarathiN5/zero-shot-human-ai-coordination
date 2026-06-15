"""
PPO Agent — Proximal Policy Optimisation for Dynamic Task Sequencing.

Architecture
────────────
  Observation → [Shared MLP encoder]
                        │
               ┌────────┴────────┐
           [Actor head]    [Critic head]
               │
         Action logits (N*2)
               │
         Masked softmax  ← only eligible (task, assignee) pairs
               │
           Sampled action

Key design choices
──────────────────
  1. Action masking: prevents the agent from ever picking a task whose
     prerequisites are unmet or that is already in-progress / done.
     This dramatically speeds up learning.
  2. Separate actor/critic heads sharing a common trunk (standard PPO).
  3. Entropy bonus encourages exploration of task assignment strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, List

from env.task_definitions import NUM_TASKS
from agent.human_observer import EXTRA_OBS_DIM


# ── Network ───────────────────────────────────────────────────────────── #

class TaskSequencingNetwork(nn.Module):
    """Shared encoder + actor + critic heads."""

    def __init__(self, obs_dim: int, hidden: int = 256, action_dim: int = NUM_TASKS * 2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal init (standard for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, obs: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[Categorical, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.actor(h)

        if mask is not None:
            # Set illegal actions to -inf before softmax
            logits = logits.masked_fill(~mask, float("-1e9"))

        dist  = Categorical(logits=logits)
        value = self.critic(h).squeeze(-1)
        return dist, value


# ── Rollout buffer ────────────────────────────────────────────────────── #

class RolloutBuffer:
    """Stores a single PPO rollout."""

    def __init__(self, size: int, obs_dim: int, device: torch.device):
        self.size   = size
        self.device = device
        self.obs    = torch.zeros(size, obs_dim)
        self.acts   = torch.zeros(size, dtype=torch.long)
        self.rews   = torch.zeros(size)
        self.vals   = torch.zeros(size)
        self.logps  = torch.zeros(size)
        self.dones  = torch.zeros(size)
        self.masks  = torch.zeros(size, NUM_TASKS * 2, dtype=torch.bool)
        self.ptr    = 0

    def store(self, obs, act, rew, val, logp, done, mask):
        i = self.ptr
        self.obs[i]   = torch.as_tensor(obs)
        self.acts[i]  = int(act)
        self.rews[i]  = float(rew)
        self.vals[i]  = float(val)
        self.logps[i] = float(logp)
        self.dones[i] = float(done)
        self.masks[i] = mask
        self.ptr      = (self.ptr + 1) % self.size

    def compute_returns(self, last_value: float, gamma: float, lam: float
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE advantage estimation."""
        advantages = torch.zeros(self.size)
        gae = 0.0
        next_val = last_value
        for t in reversed(range(self.size)):
            td = self.rews[t] + gamma * next_val * (1 - self.dones[t]) - self.vals[t]
            gae = td + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
            next_val = self.vals[t].item()
        returns = advantages + self.vals
        return advantages.to(self.device), returns.to(self.device)

    def to(self, device: torch.device):
        self.obs   = self.obs.to(device)
        self.acts  = self.acts.to(device)
        self.rews  = self.rews.to(device)
        self.vals  = self.vals.to(device)
        self.logps = self.logps.to(device)
        self.dones = self.dones.to(device)
        self.masks = self.masks.to(device)
        self.device = device
        return self


# ── PPO Agent ─────────────────────────────────────────────────────────── #

class PPOAgent:
    """
    Proximal Policy Optimisation agent for dynamic task sequencing.

    Parameters
    ----------
    obs_dim     : dimension of (optionally enhanced) observation
    hidden      : hidden layer size
    lr          : learning rate
    gamma       : discount factor
    lam         : GAE lambda
    clip_eps    : PPO clip ratio ε
    ent_coef    : entropy coefficient
    vf_coef     : value-function loss coefficient
    update_epochs: number of PPO epochs per rollout
    minibatch   : minibatch size for PPO updates
    rollout_len : steps per rollout buffer
    """

    def __init__(
        self,
        obs_dim:       int   = None,   # set automatically in train.py
        hidden:        int   = 256,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        lam:           float = 0.95,
        clip_eps:      float = 0.2,
        ent_coef:      float = 0.01,
        vf_coef:       float = 0.5,
        update_epochs: int   = 4,
        minibatch:     int   = 64,
        rollout_len:   int   = 2048,
        device:        str   = "cpu",
    ):
        from env.task_definitions import NUM_TASKS
        from agent.human_observer import EXTRA_OBS_DIM
        from env.collaborative_env import OBS_DIM

        self.obs_dim = obs_dim or (OBS_DIM + EXTRA_OBS_DIM)
        self.gamma   = gamma
        self.lam     = lam
        self.clip_eps      = clip_eps
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.update_epochs = update_epochs
        self.minibatch     = minibatch
        self.rollout_len   = rollout_len
        self.device        = torch.device(device)

        self.net = TaskSequencingNetwork(
            obs_dim    = self.obs_dim,
            hidden     = hidden,
            action_dim = NUM_TASKS * 2,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer(rollout_len, self.obs_dim, self.device)

        self._total_updates = 0

    # ── action selection ─────────────────────────────────────────────── #
    def build_mask(self, info: dict) -> torch.Tensor:
        """
        Build a boolean mask over the action space.
        True  = action is legal
        False = action is masked out
        """
        mask = torch.zeros(NUM_TASKS * 2, dtype=torch.bool)
        eligible = info.get("eligible_tasks", list(range(NUM_TASKS)))
        ai_free  = info.get("ai_task") is None
        human_free = not info.get("human_busy", False)

        for t in eligible:
            if human_free:
                mask[t * 2]     = True   # assign to human
            if ai_free:
                mask[t * 2 + 1] = True   # assign to AI

        # fallback: if nothing is legal, allow all (env will penalise)
        if not mask.any():
            mask[:] = True
        return mask

    @torch.no_grad()
    def select_action(self, obs: np.ndarray,
                      info: dict,
                      deterministic: bool = False
                      ) -> Tuple[int, float, float]:
        """
        Select an action.
        Returns (action, log_prob, value).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask  = self.build_mask(info).unsqueeze(0).to(self.device)

        dist, value = self.net(obs_t, mask)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    # ── store transition ─────────────────────────────────────────────── #
    def store(self, obs, action, reward, value, logp, done, info):
        mask = self.build_mask(info)
        self.buffer.store(obs, action, reward, value, logp, float(done), mask)

    # ── PPO update ───────────────────────────────────────────────────── #
    def update(self, last_obs: np.ndarray, last_info: dict) -> dict:
        """Run PPO update after a complete rollout. Returns loss metrics."""
        last_obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
        mask_t     = self.build_mask(last_info).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, last_val = self.net(last_obs_t, mask_t)

        advantages, returns = self.buffer.compute_returns(
            last_val.item(), self.gamma, self.lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.buffer.to(self.device)
        idx = torch.arange(self.rollout_len)

        total_pg = total_vf = total_ent = 0.0
        for _ in range(self.update_epochs):
            perm = torch.randperm(self.rollout_len)
            for start in range(0, self.rollout_len, self.minibatch):
                mb = perm[start:start + self.minibatch]

                obs_mb  = self.buffer.obs[mb]
                act_mb  = self.buffer.acts[mb]
                adv_mb  = advantages[mb]
                ret_mb  = returns[mb]
                logp_mb = self.buffer.logps[mb]
                mask_mb = self.buffer.masks[mb]

                dist, val = self.net(obs_mb, mask_mb)
                new_logp  = dist.log_prob(act_mb)
                entropy   = dist.entropy().mean()

                ratio  = (new_logp - logp_mb).exp()
                pg_loss = -torch.min(
                    ratio * adv_mb,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                ).mean()

                vf_loss = F.mse_loss(val, ret_mb)
                loss    = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                total_pg  += pg_loss.item()
                total_vf  += vf_loss.item()
                total_ent += entropy.item()

        self._total_updates += 1
        n = self.update_epochs * max(1, self.rollout_len // self.minibatch)
        return {
            "policy_loss": total_pg / n,
            "value_loss":  total_vf / n,
            "entropy":     total_ent / n,
        }

    # ── save / load ──────────────────────────────────────────────────── #
    def save(self, path: str):
        torch.save({
            "net":       self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates":   self._total_updates,
        }, path)
        print(f"[PPOAgent] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._total_updates = ckpt.get("updates", 0)
        print(f"[PPOAgent] Loaded checkpoint ← {path}  (updates={self._total_updates})")
