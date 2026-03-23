from __future__ import annotations

import asyncio
import dataclasses
import heapq
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Generic, TypeVar

from .logging_utils import get_logger

T = TypeVar("T")

logger = get_logger("scheduler")

class RequestPriority(IntEnum):
    """
    Lower values = Higher priority.
    """
    SUPPLEMENT = 0          # Highest: Missing/failed data retries
    SAME_MODEL_SAME_BLOCK = 1 # Sequential continuity for one model/block
    SAME_MODEL_DIFF_BLOCK = 2
    DIFF_MODEL_SAME_BLOCK = 3
    DIFF_MODEL_DIFF_BLOCK = 4
    LOWEST = 10

@dataclass(order=True)
class PrioritizedTask:
    priority: int
    timestamp: float = field(default_factory=time.time)
    task_id: str = field(compare=False, default="task")
    coro_func: Callable[[], Coroutine[Any, Any, T]] = field(compare=False, default=None)
    future: asyncio.Future[T] = field(compare=False, default_factory=lambda: asyncio.Future())
    # Support both single test_id and list for batch tracing
    test_id: str | list[str] = field(compare=False, default="")
    slot_id: str = field(compare=False, default="")
    model: str = field(compare=False, default="")

class PriorityJudgeScheduler:
    """
    Global scheduler for judge requests with priority queue.
    Ensures that retries and related blocks are prioritized correctly.
    """
    def __init__(self, parallelism: int, logger: Any = None, tracer: Any = None):
        self.parallelism = max(1, parallelism)
        self.queue: asyncio.PriorityQueue[PrioritizedTask] = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._active_requests = 0
        self._lock = asyncio.Lock()
        self.logger = logger or get_logger("scheduler")
        self.tracer = tracer
        
        # Statistics
        self.total_submitted = 0
        self.total_completed = 0
        self.total_failed = 0
        self.request_timestamps = [] # List of (timestamp, success_bool)
        self._stats_task = None

    async def start(self):
        async with self._lock:
            if self._running:
                return
            self._running = True
            self._workers = [
                asyncio.create_task(self._worker_loop(i)) 
                for i in range(self.parallelism)
            ]
            self._stats_task = asyncio.create_task(self._log_stats_loop())
            self.logger.info(f"优先级调度器启动 (并发数={self.parallelism})")

    async def _log_stats_loop(self):
        """Periodically log RPS/RPM and queue status."""
        while self._running:
            await asyncio.sleep(10)
            now = time.time()
            # Clean old timestamps (keep last 60s)
            self.request_timestamps = [t for t in self.request_timestamps if now - t[0] < 60]
            
            rpm = len(self.request_timestamps)
            rps = len([t for t in self.request_timestamps if now - t[0] < 1])
            
            self.logger.info(
                f"[STATS] 队列={self.queue.qsize()} | 运行中={self._active_requests}/{self.parallelism} | "
                f"RPS={rps} | RPM={rpm} | 总提交={self.total_submitted} | 总完成={self.total_completed}"
            )

    async def stop(self):
        async with self._lock:
            if not self._running:
                return
            self._running = False
            if self._stats_task:
                self._stats_task.cancel()
            for w in self._workers:
                w.cancel()
            self._workers = []
            self.logger.info("优先级调度器停止")

    async def submit(
        self, 
        coro_func: Callable[[], Coroutine[Any, Any, T]], 
        priority: RequestPriority = RequestPriority.DIFF_MODEL_DIFF_BLOCK,
        task_id: str = "task",
        test_id: str | list[str] = "",
        slot_id: str = "",
        model: str = ""
    ) -> asyncio.Future[T]:
        """Submit a task to the priority queue."""
        if not self._running:
            await self.start()
            
        task = PrioritizedTask(
            priority=int(priority),
            task_id=task_id,
            coro_func=coro_func,
            timestamp=time.time(),
            test_id=test_id,
            slot_id=slot_id,
            model=model
        )
        self.total_submitted += 1
        
        if self.tracer:
            tids = [test_id] if isinstance(test_id, str) else test_id
            for tid in tids:
                if tid:
                    self.tracer.record(tid, slot_id, model, "submit")
        
        self.logger.debug(f"[SUBMIT] task_id={task_id} priority={int(priority)}")
        await self.queue.put(task)
        return task.future

    async def _worker_loop(self, worker_id: int):
        import random
        while self._running:
            try:
                task = await self.queue.get()
                
                # JITTER: Prevent multiple workers from hitting the proxy at the exact same millisecond
                await asyncio.sleep(random.uniform(0.1, 2.0))
                
                start_time = time.time()
                wait_time = start_time - task.timestamp
                
                async with self._lock:
                    self._active_requests += 1
                
                if self.tracer:
                    tids = [task.test_id] if isinstance(task.test_id, str) else task.test_id
                    for tid in tids:
                        if tid:
                            self.tracer.record(tid, task.slot_id, task.model, "start")

                self.logger.debug(f"[START] task_id={task.task_id} priority={task.priority} worker={worker_id} wait_time={wait_time:.4f}s")
                
                try:
                    result = await task.coro_func()
                    
                    # Record success
                    self.request_timestamps.append((time.time(), True))
                    self.total_completed += 1
                    
                    if not task.future.done():
                        task.future.set_result(result)
                    
                    self.logger.debug(f"[RELEASE] task_id={task.task_id} priority={task.priority} status=success")
                except asyncio.CancelledError:
                    self.logger.debug(f"[RELEASE] task_id={task.task_id} priority={task.priority} status=cancelled")
                    if not task.future.done():
                        task.future.set_exception(asyncio.CancelledError())
                    raise
                except Exception as e:
                    self.total_failed += 1
                    self.logger.error(f"[RELEASE] task_id={task.task_id} priority={task.priority} status=failed error={e}")
                    if not task.future.done():
                        task.future.set_exception(e)
                    
                    # CRITICAL: If a task failed (likely a WAF block), force a LONG cooldown
                    # This prevents the "immediate retry" loop from hitting the proxy too fast
                    await asyncio.sleep(5.0 + random.random() * 5.0)
                finally:
                    if self.tracer:
                        tids = [task.test_id] if isinstance(task.test_id, str) else task.test_id
                        for tid in tids:
                            if tid:
                                self.tracer.record(tid, task.slot_id, task.model, "proc_done")
                    
                    # Normal post-task cooling
                    await asyncio.sleep(0.5 + random.random()) 
                    
                    async with self._lock:
                        self._active_requests -= 1
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} crashed: {e}")
                await asyncio.sleep(0.1)

    @property
    def active_requests(self) -> int:
        return self._active_requests

    @property
    def pending_tasks(self) -> int:
        return self.queue.qsize()
