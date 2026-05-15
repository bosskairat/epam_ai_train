"""
app/observability/resource.py
------------------------------
Background resource sampler that updates Prometheus gauges using psutil.
"""
from __future__ import annotations

import asyncio
from app.core.logger import get_logger
from app.observability import metrics
from typing import Optional

logger = get_logger(__name__)


async def start_resource_monitoring(interval_s: int = 5) -> Optional[asyncio.Task]:
    try:
        import psutil
    except Exception as exc:
        logger.warning(f"psutil not available; resource monitoring disabled: {exc}")
        return None

    async def _monitor():
        proc = psutil.Process()
        while True:
            try:
                # system-level
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                # process-level
                p_cpu = proc.cpu_percent(interval=None)
                rss = proc.memory_info().rss / (1024 * 1024)

                metrics.system_cpu_percent.set(cpu)
                metrics.system_memory_percent.set(mem)
                metrics.process_cpu_percent.set(p_cpu)
                metrics.process_memory_mb.set(rss)
            except Exception as exc:
                logger.warning(f"Resource monitor error: {exc}")
            await asyncio.sleep(interval_s)

    task = asyncio.create_task(_monitor())
    return task
