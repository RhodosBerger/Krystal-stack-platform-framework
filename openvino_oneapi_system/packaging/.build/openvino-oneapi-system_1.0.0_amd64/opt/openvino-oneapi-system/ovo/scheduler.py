import time
import uuid
from dataclasses import dataclass
from typing import List


@dataclass
class ScheduledTask:
    task_id: str
    name: str
    period_seconds: float
    next_deadline: float


class TimerScheduler:
    def __init__(self) -> None:
        self.tasks: List[ScheduledTask] = []

    def add_periodic_task(self, name: str, period_seconds: float) -> str:
        task_id = str(uuid.uuid4())
        self.tasks.append(
            ScheduledTask(
                task_id=task_id,
                name=name,
                period_seconds=period_seconds,
                next_deadline=time.time() + period_seconds,
            )
        )
        return task_id

    def due_tasks(self, now: float) -> List[ScheduledTask]:
        due: List[ScheduledTask] = []
        for task in self.tasks:
            if now >= task.next_deadline:
                due.append(task)
                task.next_deadline = now + task.period_seconds
        return due

