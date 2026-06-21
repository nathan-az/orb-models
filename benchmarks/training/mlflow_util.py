"""Shared MLflow logging for the bucket-feasibility probes (and train scripts).

Thin wrapper so every probe logs the SAME way: a configurable tracking URI
(default a local server on :5000), an ``--enable-mlflow`` gate (default ON), and a
single object that logs params up front and metrics/tags as the run progresses --
so the OOM path can still record ``result=OOM`` before the script exits.
"""

from __future__ import annotations

import argparse

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT = "orb-v3-train"


def add_mlflow_args(
    p: argparse.ArgumentParser, experiment: str = DEFAULT_EXPERIMENT
) -> None:
    """Add the shared mlflow flags to an argparse parser."""
    g = p.add_argument_group("mlflow")
    g.add_argument("--mlflow-uri", default=DEFAULT_TRACKING_URI,
                   help="MLflow tracking URI (default: local server on :5000).")
    g.add_argument("--mlflow-experiment", default=experiment,
                   help="MLflow experiment name.")
    g.add_argument("--enable-mlflow", dest="enable_mlflow",
                   action="store_true", default=True,
                   help="Log this run to MLflow (default: on).")
    g.add_argument("--no-mlflow", dest="enable_mlflow", action="store_false",
                   help="Disable MLflow logging.")
    g.add_argument("--run-name", default=None,
                   help="MLflow run name (defaults to a per-probe name).")


class MlflowLogger:
    """Start a run + log params eagerly; metrics/tags flow in as the probe runs.

    A no-op when ``--no-mlflow`` is passed or mlflow can't be imported, so the
    probes run identically with or without a tracking server.
    """

    def __init__(self, args, run_name: str, params: dict) -> None:
        self.enabled = bool(getattr(args, "enable_mlflow", True))
        self._mlflow = None
        if not self.enabled:
            return
        try:
            import mlflow  # local import: a --no-mlflow run never needs it
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

    def set_tags(self, tags: dict) -> None:
        if self.enabled:
            self._mlflow.set_tags(tags)

    def finish(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        self._mlflow.end_run(status=status)
        print(f"  mlflow        : logged '{self.run_name}' (status={status}) "
              f"to {self.experiment} @ {self.uri}")
