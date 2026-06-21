"""Shared MLflow logging for the training benchmark scripts.

Thin wrapper so every run logs the same way: a configurable tracking URI
(default a local server on :5000), an ``enable_mlflow`` gate (default ON), and a
single object that logs params up front and metrics as the run progresses.
"""

from __future__ import annotations

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT = "orb-v3-train"


class MlflowLogger:
    """Start a run + log params eagerly; metrics/tags flow in as the run progresses.

    A no-op when mlflow logging is disabled or mlflow can't be imported, so a run
    behaves identically with or without a tracking server.
    """

    def __init__(self, args, run_name: str, params: dict) -> None:
        self.enabled = bool(getattr(args, "enable_mlflow", True))
        self._mlflow = None
        if not self.enabled:
            return
        try:
            import mlflow  # local import: a disabled run never needs it
        except Exception as e:  # noqa: BLE001
            print(f"  mlflow        : WARNING import failed ({type(e).__name__}: {e}); skipping")
            self.enabled = False
            return
        self._mlflow = mlflow
        self.run_name = getattr(args, "run_name", None) or run_name
        self.experiment = args.mlflow_experiment
        self.uri = args.mlflow_uri
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment)
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            self._mlflow.log_metrics(metrics, step=step)

    def finish(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        self._mlflow.end_run(status=status)
        print(f"  mlflow        : logged '{self.run_name}' (status={status}) "
              f"to {self.experiment} @ {self.uri}")
