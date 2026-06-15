"""
Coordination Metrics — novelty #4 from the paper.

Tracks per-episode and rolling-window statistics:
  1. Coordination Efficiency Score  (CES)
  2. Role Switch Rate
  3. Human–AI Synchronisation Rate
  4. Zero-Shot Generalisation Gap   (computed across profiles)
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class EpisodeStats:
    profile:          str
    steps:            int
    total_reward:     float
    tasks_completed:  int          # out of NUM_TASKS
    human_tasks:      int
    ai_tasks:         int
    role_switches:    int
    parallel_steps:   int          # steps both agents worked simultaneously
    conflict_count:   int
    completion_time:  float        # steps to finish, or MAX if truncated


class CoordinationMetrics:
    """
    Accumulates episode stats and computes the coordination efficiency score.

    Coordination Efficiency Score (CES) definition
    ───────────────────────────────────────────────
      CES = w1 * completion_rate
          + w2 * speed_score
          + w3 * sync_score
          - w4 * conflict_rate

    where:
      completion_rate = tasks_completed / NUM_TASKS
      speed_score     = 1 - (completion_time / MAX_EPISODE_STEPS)
      sync_score      = parallel_steps / steps
      conflict_rate   = conflict_count / steps
    """

    WEIGHTS = dict(completion=0.40, speed=0.30, sync=0.20, conflict=0.10)
    MAX_STEPS = 200

    def __init__(self, window: int = 50):
        self.window = window
        self.history: List[EpisodeStats] = []

    def record(self, stats: EpisodeStats):
        self.history.append(stats)

    def _ces(self, s: EpisodeStats) -> float:
        from env.task_definitions import NUM_TASKS
        w = self.WEIGHTS
        completion_rate = s.tasks_completed / NUM_TASKS
        speed_score     = 1.0 - (s.completion_time / self.MAX_STEPS)
        sync_score      = s.parallel_steps / max(s.steps, 1)
        conflict_rate   = s.conflict_count / max(s.steps, 1)
        return (w["completion"] * completion_rate
              + w["speed"]      * max(speed_score, 0.0)
              + w["sync"]       * sync_score
              - w["conflict"]   * conflict_rate)

    def recent(self, n: int = None) -> List[EpisodeStats]:
        n = n or self.window
        return self.history[-n:]

    # ── aggregate stats ──────────────────────────────────────────────── #
    def mean_ces(self, n: int = None) -> float:
        hist = self.recent(n)
        if not hist:
            return 0.0
        return float(np.mean([self._ces(s) for s in hist]))

    def mean_reward(self, n: int = None) -> float:
        hist = self.recent(n)
        if not hist:
            return 0.0
        return float(np.mean([s.total_reward for s in hist]))

    def mean_completion(self, n: int = None) -> float:
        from env.task_definitions import NUM_TASKS
        hist = self.recent(n)
        if not hist:
            return 0.0
        return float(np.mean([s.tasks_completed / NUM_TASKS for s in hist]))

    def role_switch_rate(self, n: int = None) -> float:
        hist = self.recent(n)
        if not hist:
            return 0.0
        return float(np.mean([s.role_switches / max(s.steps, 1) for s in hist]))

    # ── zero-shot generalisation gap ─────────────────────────────────── #
    def zsc_gap(self) -> Dict[str, float]:
        """
        Per-profile mean CES. The gap = max_profile_CES - min_profile_CES.
        Smaller gap → better zero-shot generalisation.
        """
        profile_ces: Dict[str, List[float]] = {}
        for s in self.history:
            profile_ces.setdefault(s.profile, []).append(self._ces(s))
        result = {p: float(np.mean(v)) for p, v in profile_ces.items()}
        if len(result) >= 2:
            result["gap"] = max(result.values()) - min(result.values())
        return result

    # ── printable summary ─────────────────────────────────────────────── #
    def summary(self, n: int = None) -> str:
        n = n or self.window
        return (
            f"Episodes={len(self.recent(n))}  "
            f"CES={self.mean_ces(n):.3f}  "
            f"Reward={self.mean_reward(n):.1f}  "
            f"Completion={self.mean_completion(n)*100:.1f}%  "
            f"RoleSwitchRate={self.role_switch_rate(n):.3f}"
        )
