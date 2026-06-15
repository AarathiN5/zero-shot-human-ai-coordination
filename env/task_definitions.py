"""
Task definitions for the warehouse collaborative scenario.

Task dependency graph:
  fetch_a ──┐
             ├──▶ pack ──▶ label ──▶ weigh ──▶ dispatch
  fetch_b ──┘
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Task:
    id: int
    name: str
    difficulty: float          # 0.0 (trivial) to 1.0 (very hard)
    human_time: float          # seconds a baseline human takes
    ai_time: float             # seconds the AI takes
    prerequisites: List[int] = field(default_factory=list)
    description: str = ""


WAREHOUSE_TASKS: List[Task] = [
    Task(id=0, name="fetch_item_A",  difficulty=0.3, human_time=4.0,  ai_time=2.5,
         prerequisites=[],    description="Retrieve item A from shelf"),
    Task(id=1, name="fetch_item_B",  difficulty=0.4, human_time=5.0,  ai_time=3.0,
         prerequisites=[],    description="Retrieve item B from storage"),
    Task(id=2, name="pack_box",      difficulty=0.6, human_time=8.0,  ai_time=6.0,
         prerequisites=[0,1], description="Pack both items into a box"),
    Task(id=3, name="label_box",     difficulty=0.2, human_time=2.0,  ai_time=1.5,
         prerequisites=[2],   description="Apply shipping label"),
    Task(id=4, name="weigh_box",     difficulty=0.2, human_time=2.0,  ai_time=1.0,
         prerequisites=[3],   description="Weigh and verify shipment"),
    Task(id=5, name="dispatch",      difficulty=0.3, human_time=3.0,  ai_time=2.0,
         prerequisites=[4],   description="Place on conveyor belt"),
]

NUM_TASKS = len(WAREHOUSE_TASKS)
TASK_NAMES = [t.name for t in WAREHOUSE_TASKS]
