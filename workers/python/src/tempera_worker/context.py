"""Run context handed to every algorithm."""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tempera_worker.events import EventSink


@dataclass
class RunContext:
    """What an algorithm can see and do while training.

    * `log_metric` / `log` / `progress` report to the agent.
    * `should_stop()` turns true on SIGTERM/SIGINT (cancel, budget reached, preemption):
      the algorithm should save a checkpoint and return.
    * `checkpoint_due()` tells iterative algorithms when to checkpoint.
    """

    events: EventSink
    work_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    resume_from: Path | None = None
    checkpoint_interval_s: float = 600.0
    _stop: bool = field(default=False, init=False, repr=False)
    _last_checkpoint: float = field(default_factory=time.monotonic, init=False, repr=False)

    def install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: object) -> None:
            self._stop = True
            self.events.log(f"received signal {signum}; stopping after the current step", level="warn")

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def request_stop(self) -> None:
        self._stop = True

    def should_stop(self) -> bool:
        return self._stop

    def checkpoint_due(self) -> bool:
        return self._stop or time.monotonic() - self._last_checkpoint >= self.checkpoint_interval_s

    def log_metric(self, name: str, value: float, step: int | None = None, epoch: int | None = None) -> None:
        self.events.metric(name, value, step=step, epoch=epoch)

    def log(self, msg: str) -> None:
        self.events.log(msg)

    def progress(self, step: int, total_steps: int | None = None) -> None:
        self.events.progress(step, total_steps)

    def saved_checkpoint(self, path: Path, step: int) -> None:
        """Call after writing a checkpoint so the agent can upload it."""
        self._last_checkpoint = time.monotonic()
        self.events.checkpoint(str(path), step)
