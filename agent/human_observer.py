"""
HumanBehaviorObserver

Maintains a rolling-window estimate of the human partner's:
  - speed          (relative to baseline)
  - reliability    (1 - error_rate)
  - fatigue        (current value from env)
  - workload       (tasks assigned vs completed)

These features are **appended** to the base environment observation
so the RL agent can make better task-assignment decisions.

Usage:
    obs, info = env.reset()
    hbo = HumanBehaviorObserver(window=10)
    hbo.reset()

    enhanced_obs = hbo.enhance(obs, info)
    ...
    hbo.update(info)
    enhanced_obs = hbo.enhance(obs, info)
"""

import numpy as np
from collections import deque
from typing import Dict, Any


EXTRA_OBS_DIM = 5   # dims added on top of base env observation


class HumanBehaviorObserver:
    """
    Stateful observer that produces an enhanced observation vector
    by appending derived human-behaviour features to the raw env obs.
    """

    def __init__(self, window: int = 10):
        self.window = window
        self.reset()

    def reset(self):
        self._speed_hist    = deque(maxlen=self.window)
        self._error_hist    = deque(maxlen=self.window)
        self._tasks_assigned: int   = 0
        self._tasks_completed: int  = 0
        self._prev_human_done: int  = 0

    # ── update from env info ─────────────────────────────────────────── #
    def update(self, info: Dict[str, Any]):
        m = info.get("human_metrics", {})
        if m:
            self._speed_hist.append(m.get("speed", 0.5))
            self._error_hist.append(m.get("error_rate", 0.0))

            new_done = m.get("tasks_done", 0)
            if new_done > self._prev_human_done:
                self._tasks_completed += (new_done - self._prev_human_done)
            self._prev_human_done = new_done

    def record_assignment(self):
        """Call once every time the agent assigns a task to the human."""
        self._tasks_assigned += 1

    # ── compute features ─────────────────────────────────────────────── #
    @property
    def smooth_speed(self) -> float:
        if not self._speed_hist:
            return 0.5
        return float(np.mean(self._speed_hist))

    @property
    def smooth_error_rate(self) -> float:
        if not self._error_hist:
            return 0.0
        return float(np.mean(self._error_hist))

    @property
    def completion_ratio(self) -> float:
        if self._tasks_assigned == 0:
            return 1.0
        return self._tasks_completed / self._tasks_assigned

    @property
    def speed_trend(self) -> float:
        """Positive = speeding up, negative = slowing down."""
        if len(self._speed_hist) < 4:
            return 0.0
        half = len(self._speed_hist) // 2
        recent   = np.mean(list(self._speed_hist)[half:])
        earlier  = np.mean(list(self._speed_hist)[:half])
        return float(np.clip(recent - earlier, -1.0, 1.0))

    def extra_features(self) -> np.ndarray:
        """Returns a fixed-length float32 array of behavioural features."""
        return np.array([
            self.smooth_speed,
            self.smooth_error_rate,
            self.completion_ratio,
            self.speed_trend,
            float(self._tasks_completed),  # absolute (will be normalised downstream)
        ], dtype=np.float32)

    # ── combine with raw env obs ─────────────────────────────────────── #
    def enhance(self, obs: np.ndarray) -> np.ndarray:
        """Concatenate extra features to the raw environment observation."""
        return np.concatenate([obs, self.extra_features()], axis=-1)

    # ── readable summary ─────────────────────────────────────────────── #
    def summary(self) -> str:
        return (f"speed={self.smooth_speed:.2f}  err={self.smooth_error_rate:.2f}  "
                f"compl_ratio={self.completion_ratio:.2f}  trend={self.speed_trend:+.2f}")
