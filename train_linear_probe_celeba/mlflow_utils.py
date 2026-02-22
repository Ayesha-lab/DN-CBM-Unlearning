"""
MLflow utilities for train_linear_probe_celeba.

Goal: keep MLflow wiring out of trainer/model code as much as possible.

Usage (typical):
    from mlflow_utils import maybe_start_mlflow_run

    with maybe_start_mlflow_run(
        enabled=args.use_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        tags={"attribute": args.attribute},
        params=vars(args),
    ) as mlf:
        ... training loop ...
        mlf.log_metrics({"train/loss": 0.1}, step=epoch)
        mlf.log_artifact("path/to/checkpoint.pt")
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, Mapping, Optional

import os


def _as_str_dict(d: Mapping[str, Any]) -> Dict[str, str]:
    """MLflow tags must be string:string."""
    out: Dict[str, str] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def _sanitize_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    MLflow params should be simple scalars/strings.
    Convert common non-serializable types to strings.
    """
    clean: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue

        if isinstance(v, (str, int, float, bool)):
            clean[str(k)] = v
        else:
            # lists, Paths, enums, etc.
            clean[str(k)] = str(v)
    return clean


@dataclass
class MLflowLogger:
    """
    Thin wrapper so the rest of the code can call logger.log_* safely
    even when MLflow is disabled.
    """

    enabled: bool = False
    run_id: Optional[str] = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        import mlflow  # local import so file can be imported without mlflow installed

        mlflow.log_params(_sanitize_params(params))

    def log_tags(self, tags: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        import mlflow

        mlflow.set_tags(_as_str_dict(tags))

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        import mlflow

        mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        import mlflow

        # mlflow.log_metrics supports step kwarg in recent versions; to be safe, loop.
        for k, v in metrics.items():
            mlflow.log_metric(str(k), float(v), step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if not self.enabled:
            return
        import mlflow

        if artifact_path is None:
            mlflow.log_artifact(local_path)
        else:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)


@contextmanager
def maybe_start_mlflow_run(
    *,
    enabled: bool,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> MLflowLogger:
    """
    Context manager that starts an MLflow run iff enabled=True.

    Parameters:
        enabled: if False, yields a disabled MLflowLogger and does nothing.
        tracking_uri: e.g. "file:./mlruns" or "http://127.0.0.1:5000"
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        tags: extra tags to attach to the run
        params: params to log at run start
    """
    if not enabled:
        yield MLflowLogger(enabled=False, run_id=None)
        return

    try:
        import mlflow
    except Exception as e:
        # Fall back silently to disabled logger if mlflow isn't installed/misconfigured.
        print(f"[mlflow_utils] MLflow unavailable; continuing without logging. Error: {e}")
        yield MLflowLogger(enabled=False, run_id=None)
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # Ensure we don't accidentally inherit an outer run unless desired
    # (MLflow supports nested runs; keeping it simple here).
    active = mlflow.active_run()
    if active is not None:
        # If there's already an active run, reuse it rather than starting a new one.
        run = active
        logger = MLflowLogger(enabled=True, run_id=run.info.run_id)
        if tags:
            logger.log_tags(tags)
        if params:
            logger.log_params(params)
        try:
            yield logger
        finally:
            # Do not end a run we didn't start.
            pass
        return

    with mlflow.start_run(run_name=run_name) as run:
        logger = MLflowLogger(enabled=True, run_id=run.info.run_id)

        # Optional: record some environment info that is often useful.
        # (kept minimal; add more if you want)
        env_tags = {
            "cwd": os.getcwd(),
        }
        logger.log_tags(env_tags)

        if tags:
            logger.log_tags(tags)

        if params:
            logger.log_params(params)

        yield logger