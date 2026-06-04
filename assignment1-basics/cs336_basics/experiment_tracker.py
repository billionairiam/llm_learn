"""Experiment tracking with wandb and local JSON fallback.

Logs metrics against both gradient step and wall-clock time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetricRecord:
    step: int
    wall_time: float
    metrics: dict[str, float]


class ExperimentTracker:
    """Unified experiment tracker: logs to wandb (if enabled) and local JSON."""

    def __init__(
        self,
        project: str = "cs336",
        run_name: str | None = None,
        config: dict | None = None,
        log_dir: str = "logs",
        use_wandb: bool = False,
    ):
        self._start_time = time.time()
        self._records: list[MetricRecord] = []
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._wandb_run = None

        if use_wandb:
            import wandb

            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
            )

        self._config = config or {}
        self._run_name = run_name or f"run_{int(self._start_time)}"

    @property
    def wall_time(self) -> float:
        return time.time() - self._start_time

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics at a given gradient step. Wall-clock time is recorded automatically."""
        wt = self.wall_time
        record = MetricRecord(step=step, wall_time=wt, metrics=metrics)
        self._records.append(record)

        if self._wandb_run is not None:
            import wandb

            wandb.log({**metrics, "wall_time": wt}, step=step)

    def finish(self) -> Path:
        """Flush local JSON log and finish wandb run. Returns path to the JSON log."""
        log_path = self._log_dir / f"{self._run_name}.json"
        payload = {
            "config": self._config,
            "records": [
                {"step": r.step, "wall_time": r.wall_time, **r.metrics}
                for r in self._records
            ],
        }
        log_path.write_text(json.dumps(payload, indent=2))

        if self._wandb_run is not None:
            self._wandb_run.finish()

        return log_path
