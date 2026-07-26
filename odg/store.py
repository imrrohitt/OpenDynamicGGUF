"""Filesystem checkpoint store — durable per-step artifacts for resume."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from odg.steps import STEPS, STEPS_BY_ID, step_dir_name

StepStatus = Literal["pending", "running", "done", "failed", "skipped"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip()).strip("-").lower()
    return (s or "run")[:max_len]


@dataclass
class StepRecord:
    step_id: str
    status: StepStatus
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunMeta:
    run_id: str
    model_ref: str
    created_at: str
    updated_at: str
    root: str
    current_step: str | None = None
    status: StepStatus = "pending"
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMeta:
        return cls(**data)


class RunStore:
    """
    Layout::

        artifacts/
          runs/
            <run_id>/
              run.json
              steps/
                01_resolve/
                  status.json
                  input.json
                  output.json
                  log.txt
                  error.txt          # only on failure
                  <extra files…>
          models/
            <model_slug>/
              CURRENT               # text file with active run_id
    """

    def __init__(self, artifacts_root: str | Path | None = None) -> None:
        self.root = Path(artifacts_root or Path.cwd() / "artifacts").resolve()
        self.runs_dir = self.root / "runs"
        self.models_dir = self.root / "models"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(self, model_ref: str, run_id: str | None = None) -> RunMeta:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rid = run_id or f"{stamp}-{_slug(model_ref)}"
        run_path = self.runs_dir / rid
        if run_path.exists():
            raise FileExistsError(f"Run already exists: {run_path}")

        steps_meta = {
            s.id: StepRecord(step_id=s.id, status="pending").to_dict() for s in STEPS
        }
        now = _utc_now()
        meta = RunMeta(
            run_id=rid,
            model_ref=model_ref,
            created_at=now,
            updated_at=now,
            root=str(run_path),
            status="pending",
            steps=steps_meta,
        )
        run_path.mkdir(parents=True)
        (run_path / "steps").mkdir()
        for s in STEPS:
            (run_path / "steps" / s.dir_name).mkdir()
        self._write_run_meta(meta)
        self._point_model_current(model_ref, rid)
        return meta

    def load_run(self, run_id: str) -> RunMeta:
        path = self.runs_dir / run_id / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"No run.json for run_id={run_id!r}")
        return RunMeta.from_dict(json.loads(path.read_text()))

    def latest_run_for_model(self, model_ref: str) -> RunMeta | None:
        ptr = self.models_dir / _slug(model_ref) / "CURRENT"
        if not ptr.is_file():
            return None
        rid = ptr.read_text().strip()
        if not rid:
            return None
        try:
            return self.load_run(rid)
        except FileNotFoundError:
            return None

    def list_runs(self) -> list[RunMeta]:
        out: list[RunMeta] = []
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            if (d / "run.json").is_file():
                out.append(self.load_run(d.name))
        return out

    def get_or_create_run(
        self,
        model_ref: str,
        *,
        run_id: str | None = None,
        resume: bool = True,
    ) -> RunMeta:
        """
        Resume the model's CURRENT run if ``resume`` and it exists;
        otherwise create a new run.
        """
        if run_id:
            try:
                return self.load_run(run_id)
            except FileNotFoundError:
                return self.create_run(model_ref, run_id=run_id)
        if resume:
            existing = self.latest_run_for_model(model_ref)
            if existing is not None:
                return existing
        return self.create_run(model_ref)

    # ------------------------------------------------------------------
    # Step checkpoints
    # ------------------------------------------------------------------

    def step_path(self, run_id: str, step_id: str) -> Path:
        return self.runs_dir / run_id / "steps" / step_dir_name(step_id)

    def begin_step(
        self,
        run_id: str,
        step_id: str,
        input_data: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> Path:
        """
        Mark step running. Returns the step directory.

        If step is already ``done`` and ``force`` is False, raises
        ``StepAlreadyDone`` so callers can skip recompute.
        """
        if step_id not in STEPS_BY_ID:
            raise KeyError(f"Unknown step_id: {step_id}")

        meta = self.load_run(run_id)
        prev = meta.steps.get(step_id, {})
        if prev.get("status") == "done" and not force:
            raise StepAlreadyDone(run_id, step_id)

        step_dir = self.step_path(run_id, step_id)
        step_dir.mkdir(parents=True, exist_ok=True)

        # Clear previous failure/output when re-running
        for name in ("output.json", "error.txt", "log.txt"):
            p = step_dir / name
            if p.exists() and force:
                p.unlink()

        if input_data is not None:
            self.write_json(step_dir / "input.json", input_data)

        rec = StepRecord(
            step_id=step_id,
            status="running",
            started_at=_utc_now(),
            artifacts=list(prev.get("artifacts") or []),
        )
        self._save_step_record(run_id, rec)
        meta = self.load_run(run_id)
        meta.current_step = step_id
        meta.status = "running"
        meta.updated_at = _utc_now()
        self._write_run_meta(meta)
        self.write_json(step_dir / "status.json", rec.to_dict())
        return step_dir

    def complete_step(
        self,
        run_id: str,
        step_id: str,
        output_data: dict[str, Any],
        *,
        log_text: str | None = None,
        extra_artifacts: dict[str, Path | str | bytes] | None = None,
    ) -> Path:
        step_dir = self.step_path(run_id, step_id)
        self.write_json(step_dir / "output.json", output_data)
        if log_text is not None:
            (step_dir / "log.txt").write_text(log_text)

        artifact_names = ["output.json", "status.json"]
        if (step_dir / "input.json").exists():
            artifact_names.append("input.json")
        if log_text is not None:
            artifact_names.append("log.txt")

        if extra_artifacts:
            for name, payload in extra_artifacts.items():
                dest = step_dir / name
                if isinstance(payload, bytes):
                    dest.write_bytes(payload)
                elif isinstance(payload, Path):
                    if payload.is_file():
                        shutil.copy2(payload, dest)
                    else:
                        dest.write_text(str(payload))
                else:
                    dest.write_text(str(payload))
                artifact_names.append(name)

        rec = StepRecord(
            step_id=step_id,
            status="done",
            started_at=self._started_at(run_id, step_id),
            finished_at=_utc_now(),
            artifacts=sorted(set(artifact_names)),
        )
        self._save_step_record(run_id, rec)
        self.write_json(step_dir / "status.json", rec.to_dict())

        meta = self.load_run(run_id)
        meta.updated_at = _utc_now()
        # Advance pointer to next pending step (if any)
        next_pending = None
        for s in STEPS:
            st = meta.steps[s.id].get("status")
            if st != "done":
                next_pending = s.id
                break
        meta.current_step = next_pending
        meta.status = "done" if next_pending is None else "running"
        self._write_run_meta(meta)
        return step_dir

    def fail_step(self, run_id: str, step_id: str, error: str, log_text: str | None = None) -> None:
        step_dir = self.step_path(run_id, step_id)
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "error.txt").write_text(error)
        if log_text is not None:
            (step_dir / "log.txt").write_text(log_text)
        rec = StepRecord(
            step_id=step_id,
            status="failed",
            started_at=self._started_at(run_id, step_id),
            finished_at=_utc_now(),
            error=error,
            artifacts=["error.txt", "status.json"],
        )
        self._save_step_record(run_id, rec)
        self.write_json(step_dir / "status.json", rec.to_dict())
        meta = self.load_run(run_id)
        meta.status = "failed"
        meta.current_step = step_id
        meta.updated_at = _utc_now()
        self._write_run_meta(meta)

    def read_step_output(self, run_id: str, step_id: str) -> dict[str, Any] | None:
        path = self.step_path(run_id, step_id) / "output.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def is_step_done(self, run_id: str, step_id: str) -> bool:
        meta = self.load_run(run_id)
        return meta.steps.get(step_id, {}).get("status") == "done"

    def summary(self, run_id: str) -> str:
        meta = self.load_run(run_id)
        lines = [
            f"run_id     : {meta.run_id}",
            f"model_ref  : {meta.model_ref}",
            f"status     : {meta.status}",
            f"current    : {meta.current_step}",
            f"root       : {meta.root}",
            "",
            f"{'STEP':<22} {'STATUS':<10} FINISHED",
            "-" * 50,
        ]
        for s in STEPS:
            st = meta.steps.get(s.id, {})
            lines.append(
                f"{s.dir_name:<22} {st.get('status', 'pending'):<10} {st.get('finished_at') or '-'}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")

    def _write_run_meta(self, meta: RunMeta) -> None:
        path = Path(meta.root) / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n")

    def _save_step_record(self, run_id: str, rec: StepRecord) -> None:
        meta = self.load_run(run_id)
        meta.steps[rec.step_id] = rec.to_dict()
        meta.updated_at = _utc_now()
        self._write_run_meta(meta)

    def _started_at(self, run_id: str, step_id: str) -> str | None:
        meta = self.load_run(run_id)
        return meta.steps.get(step_id, {}).get("started_at")

    def _point_model_current(self, model_ref: str, run_id: str) -> None:
        d = self.models_dir / _slug(model_ref)
        d.mkdir(parents=True, exist_ok=True)
        (d / "CURRENT").write_text(run_id + "\n")


class StepAlreadyDone(Exception):
    def __init__(self, run_id: str, step_id: str) -> None:
        self.run_id = run_id
        self.step_id = step_id
        super().__init__(
            f"Step {step_id!r} already done for run {run_id!r}. "
            "Pass force=True / --force to redo."
        )
