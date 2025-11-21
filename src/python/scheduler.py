"""
GAMESA Scheduler - Task Scheduling and Time Management

Provides:
- Priority-based task queue
- Periodic task scheduling
- Deadline-aware execution
- Rate limiting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
from collections import deque
import time
import heapq
import threading


class TaskPriority(Enum):
    """Task priority levels."""
    IMMEDIATE = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    IDLE = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    DEFERRED = auto()


@dataclass
class ScheduledTask:
    """A scheduled task."""
    task_id: str
    callback: Callable[[], Any]
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_time: float = 0.0
    deadline: Optional[float] = None
    repeat_interval: Optional[float] = None
    max_retries: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    completed: Optional[float] = None

    def __lt__(self, other):
        """Compare for heap ordering (priority, then time)."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.scheduled_time < other.scheduled_time


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: int = 1) -> float:
        """Get wait time for tokens to become available."""
        with self._lock:
            if self._tokens >= tokens:
                return 0.0
            needed = tokens - self._tokens
            return needed / self.rate


class TaskScheduler:
    """
    GAMESA task scheduler with priority queuing.

    Features:
    - Priority-based execution
    - Deadline awareness
    - Periodic tasks
    - Rate limiting
    """

    def __init__(self):
        self._queue: List[ScheduledTask] = []
        self._tasks: Dict[str, ScheduledTask] = {}
        self._periodic: Dict[str, ScheduledTask] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.RLock()
        self._task_counter = 0
        self._stats = {
            "scheduled": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }

    def _generate_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter:06d}"

    def schedule(self, callback: Callable[[], Any],
                 priority: TaskPriority = TaskPriority.NORMAL,
                 delay: float = 0.0,
                 deadline: Optional[float] = None,
                 task_id: Optional[str] = None) -> str:
        """
        Schedule a task for execution.

        Args:
            callback: Function to execute
            priority: Task priority
            delay: Delay in seconds before execution
            deadline: Optional deadline timestamp
            task_id: Optional custom task ID

        Returns:
            Task ID
        """
        with self._lock:
            tid = task_id or self._generate_id()
            task = ScheduledTask(
                task_id=tid,
                callback=callback,
                priority=priority,
                scheduled_time=time.time() + delay,
                deadline=deadline
            )
            heapq.heappush(self._queue, task)
            self._tasks[tid] = task
            self._stats["scheduled"] += 1
            return tid

    def schedule_periodic(self, callback: Callable[[], Any],
                          interval: float,
                          priority: TaskPriority = TaskPriority.NORMAL,
                          task_id: Optional[str] = None) -> str:
        """
        Schedule a repeating task.

        Args:
            callback: Function to execute
            interval: Repeat interval in seconds
            priority: Task priority
            task_id: Optional custom task ID

        Returns:
            Task ID
        """
        with self._lock:
            tid = task_id or self._generate_id()
            task = ScheduledTask(
                task_id=tid,
                callback=callback,
                priority=priority,
                scheduled_time=time.time(),
                repeat_interval=interval
            )
            self._periodic[tid] = task
            heapq.heappush(self._queue, task)
            self._tasks[tid] = task
            self._stats["scheduled"] += 1
            return tid

    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                self._stats["cancelled"] += 1
                if task_id in self._periodic:
                    del self._periodic[task_id]
                return True
            return False

    def add_rate_limiter(self, name: str, rate: float, burst: int = 1):
        """Add a named rate limiter."""
        self._rate_limiters[name] = RateLimiter(rate, burst)

    def check_rate_limit(self, name: str, tokens: int = 1) -> bool:
        """Check if rate limit allows execution."""
        limiter = self._rate_limiters.get(name)
        if not limiter:
            return True
        return limiter.acquire(tokens)

    def tick(self) -> List[Dict]:
        """
        Process ready tasks.

        Returns:
            List of executed task results
        """
        results = []
        now = time.time()

        with self._lock:
            while self._queue:
                task = self._queue[0]

                # Check if task is ready
                if task.scheduled_time > now:
                    break

                heapq.heappop(self._queue)

                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    continue

                # Check deadline
                if task.deadline and now > task.deadline:
                    task.status = TaskStatus.FAILED
                    task.error = "Deadline exceeded"
                    self._stats["failed"] += 1
                    results.append({
                        "task_id": task.task_id,
                        "status": "deadline_exceeded"
                    })
                    continue

                # Execute task
                task.status = TaskStatus.RUNNING
                task.started = now

                try:
                    task.result = task.callback()
                    task.status = TaskStatus.COMPLETED
                    task.completed = time.time()
                    self._stats["completed"] += 1
                    results.append({
                        "task_id": task.task_id,
                        "status": "completed",
                        "result": task.result,
                        "duration": task.completed - task.started
                    })
                except Exception as e:
                    task.error = str(e)
                    task.retries += 1

                    if task.retries <= task.max_retries:
                        task.status = TaskStatus.PENDING
                        task.scheduled_time = now + 1.0  # Retry after 1s
                        heapq.heappush(self._queue, task)
                    else:
                        task.status = TaskStatus.FAILED
                        self._stats["failed"] += 1
                        results.append({
                            "task_id": task.task_id,
                            "status": "failed",
                            "error": task.error
                        })

                # Re-schedule periodic tasks
                if task.task_id in self._periodic and task.status == TaskStatus.COMPLETED:
                    task.status = TaskStatus.PENDING
                    task.scheduled_time = now + task.repeat_interval
                    heapq.heappush(self._queue, task)

        return results

    def get_pending(self) -> List[Dict]:
        """Get pending tasks."""
        with self._lock:
            return [
                {
                    "task_id": t.task_id,
                    "priority": t.priority.name,
                    "scheduled": t.scheduled_time,
                    "deadline": t.deadline
                }
                for t in self._queue if t.status == TaskStatus.PENDING
            ]

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        with self._lock:
            return {
                **self._stats,
                "pending": len([t for t in self._queue if t.status == TaskStatus.PENDING]),
                "periodic": len(self._periodic)
            }


class DeadlineScheduler:
    """
    Earliest Deadline First (EDF) scheduler.

    Prioritizes tasks by deadline for real-time requirements.
    """

    def __init__(self):
        self._tasks: List[ScheduledTask] = []
        self._lock = threading.Lock()

    def add(self, callback: Callable[[], Any], deadline: float,
            task_id: Optional[str] = None) -> str:
        """Add task with deadline."""
        with self._lock:
            tid = task_id or f"edf_{len(self._tasks)}"
            task = ScheduledTask(
                task_id=tid,
                callback=callback,
                deadline=deadline,
                scheduled_time=deadline  # Use deadline for sorting
            )
            heapq.heappush(self._tasks, task)
            return tid

    def get_next(self) -> Optional[ScheduledTask]:
        """Get next task by earliest deadline."""
        with self._lock:
            while self._tasks:
                task = heapq.heappop(self._tasks)
                if task.status == TaskStatus.PENDING:
                    return task
            return None

    def tick(self) -> Optional[Dict]:
        """Execute earliest deadline task if due."""
        now = time.time()
        task = self.get_next()

        if not task:
            return None

        if task.deadline and now > task.deadline:
            return {"task_id": task.task_id, "status": "missed_deadline"}

        try:
            result = task.callback()
            return {"task_id": task.task_id, "status": "completed", "result": result}
        except Exception as e:
            return {"task_id": task.task_id, "status": "failed", "error": str(e)}


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate scheduler functionality."""
    print("=== GAMESA Scheduler Demo ===\n")

    scheduler = TaskScheduler()

    # Schedule various tasks
    scheduler.schedule(lambda: "immediate", TaskPriority.IMMEDIATE)
    scheduler.schedule(lambda: "high", TaskPriority.HIGH)
    scheduler.schedule(lambda: "normal", TaskPriority.NORMAL)
    scheduler.schedule(lambda: "delayed", delay=0.1)

    # Periodic task
    counter = {"value": 0}

    def increment():
        counter["value"] += 1
        return counter["value"]

    scheduler.schedule_periodic(increment, interval=0.05, task_id="counter")

    # Rate limiter
    scheduler.add_rate_limiter("api", rate=2, burst=5)

    print("Processing tasks:")
    for _ in range(10):
        results = scheduler.tick()
        for r in results:
            print(f"  {r['task_id']}: {r['status']} - {r.get('result', r.get('error', ''))}")
        time.sleep(0.05)

    # Cancel periodic
    scheduler.cancel("counter")

    print(f"\nFinal counter: {counter['value']}")
    print(f"Stats: {scheduler.get_stats()}")

    # Rate limit test
    print("\n--- Rate Limiter Test ---")
    for i in range(8):
        allowed = scheduler.check_rate_limit("api")
        print(f"  Request {i+1}: {'allowed' if allowed else 'blocked'}")


if __name__ == "__main__":
    demo()
