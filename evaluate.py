"""
evaluate.py — Evaluate a trained DTS-ZSC agent.

Usage:
    python evaluate.py --checkpoint checkpoints/agent_final.pt
    python evaluate.py --checkpoint checkpoints/agent_final.pt --render
    python evaluate.py --checkpoint checkpoints/agent_final.pt --episodes 100
"""

import argparse
import numpy as np
import torch

from env.collaborative_env import CollaborativeTaskEnv, OBS_DIM
from agent.ppo_agent        import PPOAgent
from agent.human_observer   import HumanBehaviorObserver, EXTRA_OBS_DIM
from utils.metrics          import CoordinationMetrics, EpisodeStats
from human.simulated_human  import HUMAN_PROFILES
from env.task_definitions   import NUM_TASKS, WAREHOUSE_TASKS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes",   type=int, default=50,
                   help="Episodes per human profile")
    p.add_argument("--render",     action="store_true")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def run_episode(env, agent, hbo, render=False):
    """Run one episode, return (total_reward, stats_dict)."""
    obs, info = env.reset()
    hbo.reset()
    obs = hbo.enhance(obs)

    total_reward = 0.0
    steps = 0
    h_tasks = ai_tasks = role_sw = parallel = conflicts = 0
    prev_assignee = None

    while True:
        action, _, _ = agent.select_action(obs, info, deterministic=True)
        task_id  = action // 2
        assignee = action  % 2

        if prev_assignee is not None and assignee != prev_assignee:
            role_sw += 1
        prev_assignee = assignee

        obs, reward, terminated, truncated, info = env.step(action)

        if render:
            env.render()

        hbo.update(info)
        hbo.record_assignment() if assignee == 0 else None
        obs = hbo.enhance(obs)

        if assignee == 0:
            h_tasks += 1
        else:
            ai_tasks += 1

        if reward <= -2.0:
            conflicts += 1

        ai_busy    = info.get("ai_task") is not None
        human_busy = info.get("human_metrics", {}).get("tasks_done", 0) > 0
        if ai_busy and human_busy:
            parallel += 1

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    profile = info.get("human_metrics", {}).get("profile", "unknown")
    tasks_done = int(np.sum([
        1 for t in WAREHOUSE_TASKS
        if info.get("eligible_tasks") is not None
    ]))

    return total_reward, {
        "profile":    profile,
        "steps":      steps,
        "reward":     total_reward,
        "h_tasks":    h_tasks,
        "ai_tasks":   ai_tasks,
        "role_sw":    role_sw,
        "parallel":   parallel,
        "conflicts":  conflicts,
    }


def evaluate(args):
    device = torch.device(args.device)
    obs_dim = OBS_DIM + EXTRA_OBS_DIM

    agent = PPOAgent(obs_dim=obs_dim, device=args.device)
    agent.load(args.checkpoint)
    agent.net.eval()

    metrics = CoordinationMetrics(window=10_000)
    hbo     = HumanBehaviorObserver()

    print("=" * 70)
    print(f"Evaluating: {args.checkpoint}")
    print(f"Episodes per profile: {args.episodes}")
    print("=" * 70)

    profile_results = {}

    for profile_name, profile in HUMAN_PROFILES.items():
        env = CollaborativeTaskEnv(
            human_profile=profile,
            render_mode="human" if args.render else None
        )
        rewards = []

        for ep in range(args.episodes):
            seed = args.seed + ep
            env.reset(seed=seed)   # re-seed env
            reward, ep_info = run_episode(env, agent, hbo, render=args.render)
            rewards.append(reward)

            s = EpisodeStats(
                profile         = profile_name,
                steps           = ep_info["steps"],
                total_reward    = ep_info["reward"],
                tasks_completed = NUM_TASKS,   # approximation
                human_tasks     = ep_info["h_tasks"],
                ai_tasks        = ep_info["ai_tasks"],
                role_switches   = ep_info["role_sw"],
                parallel_steps  = ep_info["parallel"],
                conflict_count  = ep_info["conflicts"],
                completion_time = ep_info["steps"],
            )
            metrics.record(s)

        mean_r = np.mean(rewards)
        std_r  = np.std(rewards)
        profile_results[profile_name] = mean_r
        print(f"  {profile_name:<12}  reward={mean_r:+7.1f} ± {std_r:.1f}  "
              f"CES={metrics.mean_ces(n=args.episodes):.3f}")

    print("\n" + "─" * 70)
    print("Zero-Shot Generalisation Summary")
    print("─" * 70)
    zsc = metrics.zsc_gap()
    for k, v in sorted(zsc.items()):
        bar_len = int(v * 40)
        bar = "█" * bar_len
        print(f"  {k:<15} {bar:<40} {v:.3f}")

    print("\n" + metrics.summary())
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
