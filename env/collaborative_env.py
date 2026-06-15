"""
CollaborativeTaskEnv — Custom Gymnasium environment.

Observation space (flat vector, length = OBS_DIM):
  [0:N]        task_status          one-hot-like: 0=pending, 0.5=in_progress, 1=done
  [N:2N]       task_difficulties    normalised difficulty of each task
  [2N:3N]      who_is_doing_task    0=nobody, 0.5=human, 1=AI
  [3N]         human_speed          observed human speed [0,1]
  [3N+1]       human_error_rate     observed human error rate [0,1]
  [3N+2]       human_fatigue        current human fatigue [0,1]
  [3N+3]       time_ratio           elapsed_time / max_episode_time

Action space — Discrete(N * 2):
  action = task_id * 2 + assignee
  assignee: 0 = human,  1 = AI

Reward function:
  +task_reward   per completed task (scales with difficulty)
  +completion_bonus  if all tasks done
  -time_penalty  per time step (encourages speed)
  +sync_bonus    if human & AI work in parallel
  -conflict_penalty  if action tries to assign task with unmet deps
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any

from env.task_definitions import WAREHOUSE_TASKS, NUM_TASKS, Task
from human.simulated_human import SimulatedHuman, HumanProfile, HUMAN_PROFILES


# ── Tuneable constants ──────────────────────────────────────────────── #
MAX_EPISODE_STEPS   = 200
TIME_PENALTY        = -0.02
TASK_REWARD_SCALE   = 10.0
COMPLETION_BONUS    = 25.0
SYNC_BONUS          = 0.5
CONFLICT_PENALTY    = -2.0

OBS_DIM = NUM_TASKS * 3 + 4      # flat observation vector length


class CollaborativeTaskEnv(gym.Env):
    """
    Two-agent collaborative warehouse environment.
    Only the RL agent (AI) is controlled externally.
    The human is simulated internally.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 human_profile: Optional[HumanProfile] = None,
                 dt: float = 1.0,
                 render_mode: Optional[str] = None):

        super().__init__()
        self.dt = dt
        self.render_mode = render_mode
        self.human_profile = human_profile  # None → randomise each episode

        # ── spaces ────────────────────────────────────────────────────── #
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        )
        # action = task_id * 2 + assignee  (0=human, 1=AI)
        self.action_space = gym.spaces.Discrete(NUM_TASKS * 2)

        # ── internal state (initialised in reset) ─────────────────────── #
        self.human: Optional[SimulatedHuman] = None
        self._task_status: np.ndarray  = np.zeros(NUM_TASKS, dtype=np.float32)
        self._ai_current_task: Optional[int] = None
        self._ai_time_on_task: float = 0.0
        self._ai_task_budget: float  = 0.0
        self._step_count: int = 0

    # ── helpers ───────────────────────────────────────────────────────── #
    def _prerequisites_met(self, task_id: int) -> bool:
        task: Task = WAREHOUSE_TASKS[task_id]
        return all(self._task_status[p] == 1.0 for p in task.prerequisites)

    def _eligible_tasks(self) -> list:
        return [
            t.id for t in WAREHOUSE_TASKS
            if self._task_status[t.id] == 0.0 and self._prerequisites_met(t.id)
        ]

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:NUM_TASKS] = self._task_status
        obs[NUM_TASKS:2*NUM_TASKS] = [t.difficulty for t in WAREHOUSE_TASKS]
        # who is doing each task
        for i in range(NUM_TASKS):
            if self._task_status[i] == 0.5:
                if self.human.current_task_id == i:
                    obs[2*NUM_TASKS + i] = 0.5
                elif self._ai_current_task == i:
                    obs[2*NUM_TASKS + i] = 1.0
        obs[3*NUM_TASKS]   = self.human.observed_speed
        obs[3*NUM_TASKS+1] = self.human.observed_error_rate
        obs[3*NUM_TASKS+2] = self.human.fatigue
        obs[3*NUM_TASKS+3] = self._step_count / MAX_EPISODE_STEPS
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step":           self._step_count,
            "eligible_tasks": self._eligible_tasks(),
            "human_metrics":  self.human.get_metrics(),
            "ai_task":        self._ai_current_task,
        }

    # ── reset ─────────────────────────────────────────────────────────── #
    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Randomise human profile if none specified (for ZSC training)
        if self.human_profile is None:
            profile_name = self.np_random.choice(list(HUMAN_PROFILES.keys()))
            profile = HUMAN_PROFILES[profile_name]
        else:
            profile = self.human_profile

        self.human = SimulatedHuman(profile=profile,
                                    seed=int(self.np_random.integers(0, 1_000_000)))
        self._task_status       = np.zeros(NUM_TASKS, dtype=np.float32)
        self._ai_current_task   = None
        self._ai_time_on_task   = 0.0
        self._ai_task_budget    = 0.0
        self._step_count        = 0

        return self._get_obs(), self._get_info()

    # ── step ──────────────────────────────────────────────────────────── #
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = TIME_PENALTY
        task_id  = action // 2
        assignee = action  % 2   # 0=human, 1=AI

        task: Task = WAREHOUSE_TASKS[task_id]

        # ── process AI's action ───────────────────────────────────────── #
        valid_action = False
        if (self._task_status[task_id] == 0.0 and
                self._prerequisites_met(task_id) and
                self._ai_current_task is None):
            valid_action = True

            if assignee == 1:    # AI does it
                self._ai_current_task  = task_id
                self._ai_task_budget   = task.ai_time
                self._ai_time_on_task  = 0.0
                self._task_status[task_id] = 0.5

            else:                # suggest task to human
                if not self.human.is_busy:
                    self.human.assign_task(task_id)
                    self._task_status[task_id] = 0.5

        elif not valid_action:
            reward += CONFLICT_PENALTY

        # ── advance AI work ───────────────────────────────────────────── #
        if self._ai_current_task is not None:
            self._ai_time_on_task += self.dt
            if self._ai_time_on_task >= self._ai_task_budget:
                self._task_status[self._ai_current_task] = 1.0
                reward += TASK_REWARD_SCALE * WAREHOUSE_TASKS[self._ai_current_task].difficulty
                self._ai_current_task = None

        # ── advance human work ────────────────────────────────────────── #
        human_done = self.human.step(dt=self.dt)
        if human_done and self.human.current_task_id is None:
            # find which task just finished
            for t in WAREHOUSE_TASKS:
                if (self._task_status[t.id] == 0.5 and
                        self.human.tasks_completed > 0):
                    self._task_status[t.id] = 1.0
                    reward += TASK_REWARD_SCALE * t.difficulty
                    break

        # ── sync bonus: both agents working in parallel ───────────────── #
        if self._ai_current_task is not None and self.human.is_busy:
            reward += SYNC_BONUS

        # ── termination ───────────────────────────────────────────────── #
        self._step_count += 1
        all_done = bool(np.all(self._task_status == 1.0))
        if all_done:
            reward += COMPLETION_BONUS

        truncated = self._step_count >= MAX_EPISODE_STEPS
        terminated = all_done

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ── render ────────────────────────────────────────────────────────── #
    def render(self):
        status_map = {0.0: "⬜", 0.5: "🔄", 1.0: "✅"}
        print(f"\n── Step {self._step_count:3d} ──")
        for t in WAREHOUSE_TASKS:
            icon = status_map.get(self._task_status[t.id], "?")
            print(f"  {icon} {t.name:<15} diff={t.difficulty:.1f}")
        m = self.human.get_metrics()
        print(f"  Human: speed={m['speed']:.2f}  fatigue={m['fatigue']:.2f}  "
              f"err={m['error_rate']:.2f}  profile={m['profile']}")
        print(f"  AI task: {self._ai_current_task}")
