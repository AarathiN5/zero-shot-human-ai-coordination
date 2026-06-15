"""
SimulatedHuman — models a human partner with:
  - speed_factor  : multiplier on base task time  (fast=0.5 … slow=2.0)
  - fatigue       : accumulates over time, slows performance
  - error_rate    : probability of needing a redo
  - strategy_bias : preference for certain task types

On each step() call the human works on their assigned task (if any)
and returns whether the task was completed this tick.
"""

import numpy as np
from typing import Optional
from env.task_definitions import WAREHOUSE_TASKS, Task


class HumanProfile:
    """Defines a human 'type' — used for zero-shot generalization."""
    def __init__(self, name: str, speed: float, error_rate: float,
                 fatigue_rate: float, strategy: str = "sequential"):
        self.name = name
        self.speed = speed              # base speed multiplier
        self.error_rate = error_rate    # 0–1
        self.fatigue_rate = fatigue_rate  # fatigue gained per second of work
        self.strategy = strategy        # "sequential" | "easy_first" | "hard_first"

    def __repr__(self):
        return f"HumanProfile({self.name}, spd={self.speed:.1f}, err={self.error_rate:.2f})"


# Library of human profiles for zero-shot testing
HUMAN_PROFILES = {
    "average":    HumanProfile("average",   speed=1.0, error_rate=0.05, fatigue_rate=0.01),
    "fast":       HumanProfile("fast",      speed=0.6, error_rate=0.08, fatigue_rate=0.015),
    "slow":       HumanProfile("slow",      speed=1.6, error_rate=0.03, fatigue_rate=0.008),
    "novice":     HumanProfile("novice",    speed=1.8, error_rate=0.18, fatigue_rate=0.02),
    "expert":     HumanProfile("expert",    speed=0.7, error_rate=0.02, fatigue_rate=0.005),
    "tired":      HumanProfile("tired",     speed=1.3, error_rate=0.12, fatigue_rate=0.025),
    "easy_first": HumanProfile("easy_first",speed=1.0, error_rate=0.05, fatigue_rate=0.01,
                               strategy="easy_first"),
}


class SimulatedHuman:
    """
    A simulated human collaborator.

    The human works on one task at a time.
    Fatigue accumulates linearly and is reset between episodes.
    An error triggers a redo (doubles remaining work on current task).
    """

    def __init__(self, profile: Optional[HumanProfile] = None, seed: int = 42):
        self.profile = profile or HUMAN_PROFILES["average"]
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.fatigue: float = 0.0          # 0 (fresh) → 1 (exhausted)
        self.current_task_id: Optional[int] = None
        self.time_on_task: float = 0.0     # seconds spent on current task
        self.task_budget: float = 0.0      # total time needed for current task
        self.tasks_completed: int = 0
        self.total_errors: int = 0
        # Observed metrics for the HumanObserver
        self.recent_speeds: list = []
        self.recent_errors: list = []

    # ------------------------------------------------------------------ #
    def assign_task(self, task_id: int):
        """Assign a new task to the human."""
        task: Task = WAREHOUSE_TASKS[task_id]
        effective_speed = self.profile.speed * (1.0 + self.fatigue * 0.5)
        self.task_budget = task.human_time * effective_speed
        self.time_on_task = 0.0
        self.current_task_id = task_id

    # ------------------------------------------------------------------ #
    def step(self, dt: float = 1.0) -> bool:
        """
        Advance the human by dt seconds.
        Returns True if the current task was completed this tick.
        """
        if self.current_task_id is None:
            return False

        self.time_on_task += dt
        self.fatigue = min(1.0, self.fatigue + self.profile.fatigue_rate * dt)

        # Random error: extend task budget
        if self.rng.random() < self.profile.error_rate * dt:
            task: Task = WAREHOUSE_TASKS[self.current_task_id]
            redo_time = task.human_time * 0.5
            self.task_budget += redo_time
            self.total_errors += 1
            self.recent_errors.append(1)
        else:
            self.recent_errors.append(0)

        # Keep recent window at 20 ticks
        if len(self.recent_errors) > 20:
            self.recent_errors.pop(0)

        if self.time_on_task >= self.task_budget:
            speed_ratio = WAREHOUSE_TASKS[self.current_task_id].human_time / max(self.time_on_task, 0.01)
            self.recent_speeds.append(speed_ratio)
            if len(self.recent_speeds) > 10:
                self.recent_speeds.pop(0)
            self.tasks_completed += 1
            self.current_task_id = None
            return True

        return False

    # ------------------------------------------------------------------ #
    @property
    def observed_speed(self) -> float:
        """Normalised speed estimate [0,1] where 1=fast, 0=slow."""
        if not self.recent_speeds:
            return 0.5
        return float(np.clip(np.mean(self.recent_speeds), 0.0, 1.0))

    @property
    def observed_error_rate(self) -> float:
        """Recent error rate estimate [0,1]."""
        if not self.recent_errors:
            return 0.0
        return float(np.mean(self.recent_errors))

    @property
    def is_busy(self) -> bool:
        return self.current_task_id is not None

    def get_metrics(self) -> dict:
        return {
            "fatigue":     round(self.fatigue, 3),
            "speed":       round(self.observed_speed, 3),
            "error_rate":  round(self.observed_error_rate, 3),
            "tasks_done":  self.tasks_completed,
            "profile":     self.profile.name,
        }
