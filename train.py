"""
train.py — Main training script for DTS-ZSC.

Usage:
    python train.py                         # default settings
    python train.py --total_steps 500000    # longer run
    python train.py --device cuda           # GPU if available

Training loop
─────────────
  1. Reset env (random human profile each episode → ZSC training)
  2. Collect rollout of length ROLLOUT_LEN
  3. Run PPO update
  4. Log metrics to console (and optionally CSV)
  5. Save checkpoint every SAVE_INTERVAL steps
"""

import os
import csv
import argparse
import numpy as np
import torch

from env.collaborative_env import CollaborativeTaskEnv, OBS_DIM, MAX_EPISODE_STEPS
from agent.ppo_agent        import PPOAgent
from agent.human_observer   import HumanBehaviorObserver, EXTRA_OBS_DIM
from utils.metrics          import CoordinationMetrics, EpisodeStats
from env.task_definitions   import NUM_TASKS


# ── CLI args ──────────────────────────────────────────────────────────── #
def parse_args():
    p = argparse.ArgumentParser(description="Train DTS-ZSC PPO agent")
    p.add_argument("--total_steps",    type=int,   default=200_000)
    p.add_argument("--rollout_len",    type=int,   default=2048)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--lam",            type=float, default=0.95)
    p.add_argument("--clip_eps",       type=float, default=0.2)
    p.add_argument("--ent_coef",       type=float, default=0.01)
    p.add_argument("--update_epochs",  type=int,   default=4)
    p.add_argument("--minibatch",      type=int,   default=64)
    p.add_argument("--device",         type=str,   default="cpu")
    p.add_argument("--save_dir",       type=str,   default="checkpoints")
    p.add_argument("--save_interval",  type=int,   default=50_000)
    p.add_argument("--log_interval",   type=int,   default=2048)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────── #
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_episode_stats(profile, steps, total_reward,
                       tasks_done, h_tasks, ai_tasks,
                       role_sw, parallel, conflicts) -> EpisodeStats:
    return EpisodeStats(
        profile          = profile,
        steps            = steps,
        total_reward     = total_reward,
        tasks_completed  = tasks_done,
        human_tasks      = h_tasks,
        ai_tasks         = ai_tasks,
        role_switches    = role_sw,
        parallel_steps   = parallel,
        conflict_count   = conflicts,
        completion_time  = steps if tasks_done == NUM_TASKS else MAX_EPISODE_STEPS,
    )


# ── main ──────────────────────────────────────────────────────────────── #
def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    env     = CollaborativeTaskEnv()              # random profile each episode
    hbo     = HumanBehaviorObserver(window=10)
    metrics = CoordinationMetrics(window=50)

    obs_dim = OBS_DIM + EXTRA_OBS_DIM
    agent   = PPOAgent(
        obs_dim       = obs_dim,
        hidden        = args.hidden,
        lr            = args.lr,
        gamma         = args.gamma,
        lam           = args.lam,
        clip_eps      = args.clip_eps,
        ent_coef      = args.ent_coef,
        update_epochs = args.update_epochs,
        minibatch     = args.minibatch,
        rollout_len   = args.rollout_len,
        device        = args.device,
    )

    # ── CSV logger ──────────────────────────────────────────────────── #
    log_path = os.path.join(args.save_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["step", "episodes", "mean_reward", "mean_ces",
                     "completion_pct", "policy_loss", "value_loss", "entropy"])

    # ── rollout state ────────────────────────────────────────────────── #
    obs, info = env.reset(seed=args.seed)
    hbo.reset()
    obs = hbo.enhance(obs)

    global_step  = 0
    episode      = 0
    ep_reward    = 0.0
    ep_steps     = 0
    ep_h_tasks   = 0
    ep_ai_tasks  = 0
    ep_role_sw   = 0
    ep_parallel  = 0
    ep_conflicts = 0
    prev_assignee = None

    print(f"[Train] Starting DTS-ZSC | obs_dim={obs_dim} | device={args.device}")
    print(f"        Total steps={args.total_steps:,}  Rollout={args.rollout_len}")
    print("─" * 70)

    while global_step < args.total_steps:

        # ── collect transitions ─────────────────────────────────────── #
        for _ in range(args.rollout_len):
            action, logp, value = agent.select_action(obs, info)

            task_id  = action // 2
            assignee = action  % 2   # 0=human, 1=AI

            # role-switch tracking
            if prev_assignee is not None and assignee != prev_assignee:
                ep_role_sw += 1
            prev_assignee = assignee

            next_obs, reward, terminated, truncated, next_info = env.step(action)

            # parallel step bonus tracking
            ai_busy    = next_info.get("ai_task") is not None
            human_busy = next_info.get("human_metrics", {}).get("tasks_done", 0) > 0
            if ai_busy and human_busy:
                ep_parallel += 1

            # conflict tracking
            if reward <= -2.0:
                ep_conflicts += 1

            # task assignment tracking
            if assignee == 0:
                ep_h_tasks += 1
                hbo.record_assignment()
            else:
                ep_ai_tasks += 1

            hbo.update(next_info)
            enhanced_next = hbo.enhance(next_obs)

            done = terminated or truncated
            agent.store(obs, action, reward, value, logp, done, info)

            obs   = enhanced_next
            info  = next_info
            ep_reward += reward
            ep_steps  += 1
            global_step += 1

            if done:
                profile = next_info.get("human_metrics", {}).get("profile", "?")
                stats   = make_episode_stats(
                    profile, ep_steps, ep_reward,
                    NUM_TASKS - int(sum(1 for t in range(NUM_TASKS)
                                       if next_info.get("eligible_tasks") is not None)),
                    ep_h_tasks, ep_ai_tasks,
                    ep_role_sw, ep_parallel, ep_conflicts,
                )
                metrics.record(stats)
                episode += 1

                # reset episode accumulators
                ep_reward = ep_steps = ep_h_tasks = ep_ai_tasks = 0
                ep_role_sw = ep_parallel = ep_conflicts = 0
                prev_assignee = None

                obs, info = env.reset()
                hbo.reset()
                obs = hbo.enhance(obs)

            if global_step >= args.total_steps:
                break

        # ── PPO update ──────────────────────────────────────────────── #
        loss_info = agent.update(obs, info)

        # ── logging ─────────────────────────────────────────────────── #
        if global_step % args.log_interval < args.rollout_len:
            mean_r   = metrics.mean_reward(n=20)
            mean_ces = metrics.mean_ces(n=20)
            comp_pct = metrics.mean_completion(n=20) * 100
            print(
                f"Step {global_step:>8,} | Ep {episode:>5} | "
                f"Reward={mean_r:+7.1f} | CES={mean_ces:.3f} | "
                f"Compl={comp_pct:4.1f}% | "
                f"πLoss={loss_info['policy_loss']:+.4f} | "
                f"VLoss={loss_info['value_loss']:.4f}"
            )
            writer.writerow([
                global_step, episode, round(mean_r, 2), round(mean_ces, 4),
                round(comp_pct, 1), round(loss_info["policy_loss"], 5),
                round(loss_info["value_loss"], 5), round(loss_info["entropy"], 5)
            ])
            log_file.flush()

        # ── checkpoint ──────────────────────────────────────────────── #
        if global_step % args.save_interval < args.rollout_len:
            ckpt = os.path.join(args.save_dir, f"agent_{global_step:08d}.pt")
            agent.save(ckpt)

    # ── final save ──────────────────────────────────────────────────── #
    agent.save(os.path.join(args.save_dir, "agent_final.pt"))
    log_file.close()

    print("\n" + "═" * 70)
    print("Training complete.")
    print(metrics.summary())
    zsc = metrics.zsc_gap()
    print("ZSC generalisation gap per profile:")
    for k, v in sorted(zsc.items()):
        print(f"  {k:<15} CES={v:.3f}")
    print("═" * 70)


if __name__ == "__main__":
    args = parse_args()
    train(args)
