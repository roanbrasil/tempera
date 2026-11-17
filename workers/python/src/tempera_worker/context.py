"""Run context handed to every algorithm."""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tempera_worker.events import EventSink

if TYPE_CHECKING:
    from tempera_worker.manifest import Feature, Task


@dataclass
class RunContext:
    """What an algorithm can see and do while training.

    * `log_metric` / `log` / `progress` report to the agent.
    * `should_stop()` turns true on SIGTERM/SIGINT (cancel, budget reached, preemption):
      the algorithm should save a checkpoint and return.
    * `checkpoint_due()` tells iterative algorithms when to checkpoint.
    * `task`, `features`, `classes` and `seed` come from the fit manifest (Contracts.md, section 7).
    """

    events: EventSink
    work_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    task: Task = "regression"
    features: list[Feature] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    seed: int = 0
    resume_from: Path | None = None
    checkpoint_interval_s: float = 600.0
    _stop: bool = field(default=False, init=False, repr=False)
    _signal: int | None = field(default=None, init=False, repr=False)
    _last_checkpoint: float = field(default_factory=time.monotonic, init=False, repr=False)

    def install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: object) -> None:
            # Never write events here: the handler can interrupt a write in progress on stdout.
            self._stop = True
            self._signal = signum

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def request_stop(self) -> None:
        self._stop = True

    def should_stop(self) -> bool:
        if self._signal is not None:
            signum, self._signal = self._signal, None
            self.events.log(f"received signal {signum}; stopping after the current step", level="warn")
        return self._stop

    def checkpoint_due(self) -> bool:
        return self.should_stop() or time.monotonic() - self._last_checkpoint >= self.checkpoint_interval_s

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
