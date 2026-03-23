from __future__ import annotations

import time
import json
import collections
from dataclasses import dataclass, field, asdict
from typing import Any, DefaultDict, Dict, List


@dataclass
class TaskEvent:
    test_id: str
    slot_id: str
    model: str
    event_type: str  # submit, start, api_call, api_done, proc_done
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DiagnosticTracer:
    def __init__(self):
        self.events: List[TaskEvent] = []
        self._start_time = time.time()

    def record(self, test_id: str, slot_id: str, model: str, event_type: str, metadata: Dict[str, Any] = None):
        self.events.append(TaskEvent(
            test_id=test_id,
            slot_id=slot_id,
            model=model,
            event_type=event_type,
            timestamp=time.time(),
            metadata=metadata or {}
        ))

    def get_report(self) -> Dict[str, Any]:
        if not self.events:
            return {}

        # group events by (test_id, slot_id)
        task_groups: DefaultDict[tuple, Dict[str, float]] = collections.defaultdict(dict)
        # also track model/block associations
        task_meta: Dict[tuple, Dict[str, str]] = {}

        for ev in self.events:
            key = (ev.test_id, ev.slot_id)
            task_groups[key][ev.event_type] = ev.timestamp
            if key not in task_meta:
                task_meta[key] = {"model": ev.model}

        # Calculate metrics
        wait_times: List[float] = []
        api_latencies: List[float] = []
        cpu_proc_times: List[float] = []
        network_times: List[float] = []

        model_stats: DefaultDict[str, List[float]] = collections.defaultdict(list) # model -> wait_times

        for key, timestamps in task_groups.items():
            model = task_meta[key]["model"]
            
            # Queue Wait Time: start - submit
            if "start" in timestamps and "submit" in timestamps:
                wait = timestamps["start"] - timestamps["submit"]
                wait_times.append(wait)
                model_stats[model].append(wait)

            # API Latency: api_done - api_call
            if "api_done" in timestamps and "api_call" in timestamps:
                lat = timestamps["api_done"] - timestamps["api_call"]
                api_latencies.append(lat)
                network_times.append(lat) # Simplified: Network time is API latency

            # Total CPU processing time: (start -> api_call) + (api_done -> proc_done)
            cpu_time = 0.0
            if "start" in timestamps and "api_call" in timestamps:
                cpu_time += (timestamps["api_call"] - timestamps["start"])
            if "api_done" in timestamps and "proc_done" in timestamps:
                cpu_time += (timestamps["proc_done"] - timestamps["api_done"])
            
            if cpu_time > 0:
                cpu_proc_times.append(cpu_time)

        report = {
            "summary": {
                "total_tasks": len(task_groups),
                "avg_queue_wait_time": sum(wait_times) / len(wait_times) if wait_times else 0,
                "avg_api_latency": sum(api_latencies) / len(api_latencies) if api_latencies else 0,
                "total_cpu_proc_time": sum(cpu_proc_times),
                "total_network_time": sum(network_times),
            },
            "per_model_avg_wait": {
                model: sum(waits) / len(waits) for model, waits in model_stats.items()
            }
        }
        return report

    def save_report(self, path: str):
        report = self.get_report()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
