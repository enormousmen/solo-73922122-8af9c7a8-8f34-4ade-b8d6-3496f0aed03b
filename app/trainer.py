"""Single-job training manager.

One background trainer thread runs the loop below; FastAPI handlers only
manipulate flags / wait on conditions, so slow training, slow checkpoint
saves and slow recovery never block status queries.

Reproducibility design (bit-exact resume):

* ``torch.set_num_threads(1)`` and deterministic algorithms are enabled, so a
  given sequence of ops always produces the same float32 bits on this build.
* A fresh run seeds the global torch RNG once (``torch.manual_seed(seed)``).
  Model init and Dropout then draw from that single stream.
* The per-epoch sample order is a pure function of ``(seed, epoch)`` via a
  dedicated ``random.Random`` instance, so the shuffle of any epoch can be
  re-derived after a restart without storing it.
* Every checkpoint stores: model + Adam state, the torch RNG state, the
  position (``epoch``, ``next_batch``), the full loss history, the resolved
  config and a ``completed`` flag.  Checkpoints are only taken at batch
  boundaries, capturing the exact state "ready to run batch k".
* Resume after a *pause* continues the in-memory state (which is at least as
  new as the persisted checkpoint).  Resume after a *failure* is different:
  the failing batch already ran ``optimizer.step()`` but its progress was
  never recorded, so the in-memory model/optimizer/RNG are contaminated.  A
  failed-job resume therefore DISCARDS the in-memory scene and fully
  restores model, Adam, RNG, batch position and loss history from the latest
  successful checkpoint of the SAME run, then continues the deterministic
  sequence — every subsequent batch loss and the final parameters are
  bit-identical to an uninterrupted run.
* If no usable checkpoint exists, resume is refused.  If loading fails, the
  original checkpoint file is kept untouched and the reason is queryable via
  GET /train/status; the job stays ``failed`` and never starts from a
  half-restored scene.

TEST-ONLY HOOK: a file named ``INJECT_BATCH_FAILURE_AT`` in the runs
directory, containing ``"epoch:batch"`` (e.g. ``"0:5"``), makes the trainer
raise a transient error at that batch, AFTER ``optimizer.step()`` but BEFORE
the progress/loss is recorded — precisely the "params updated, progress not
yet recorded" window.  Removing the file makes the next resume succeed.
"""
from __future__ import annotations

import io
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import (
    CheckpointError,
    CheckpointLoadError,
    checkpoint_path,
    cleanup_stale_tmp_files,
    load_checkpoint,
    save_checkpoint,
)
from .config import TrainConfig
from .dataset import load_classification_csv
from .model import build_model

INJECT_BATCH_FAILURE_FILENAME = "INJECT_BATCH_FAILURE_AT"

# Every field a complete checkpoint must contain.  Loading rejects anything
# missing one of these, so a half-written / foreign file can never be applied.
REQUIRED_CKPT_KEYS = (
    "run_id", "config", "epoch", "next_batch", "completed", "losses",
    "model_state_dict", "optimizer_state_dict", "rng_state",
)


class JobState(str, Enum):
    IDLE = "idle"            # no run exists
    RUNNING = "running"      # trainer thread is working through batches
    PAUSING = "pausing"      # pause requested; finishing batch + checkpoint
    PAUSED = "paused"        # trainer stopped; resumable (see checkpoint_dirty)
    RECOVERING = "recovering"  # failed job is being restored from checkpoint
    FINISHED = "finished"    # all epochs done and final checkpoint persisted
    FAILED = "failed"        # training/recovery error (see last_error)


class ConflictError(Exception):
    """409 - request incompatible with current state."""


class NotFoundError(Exception):
    """404 - no run / checkpoint exists."""


class PauseTimeoutError(Exception):
    """504 - trainer did not stop within pause_timeout_seconds."""


class CheckpointSaveError(Exception):
    """500 - pause could not persist a complete checkpoint."""


def epoch_permutation(seed: int, epoch: int, n: int) -> list[int]:
    """Deterministic per-epoch shuffle, re-derivable after a restart."""
    rng = random.Random(f"{seed}:{epoch}")
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_batch_failure_point(runs_dir: Path) -> tuple[int, int] | None:
    """Test-only hook: parse ``INJECT_BATCH_FAILURE_AT`` ('epoch:batch')."""
    path = runs_dir / INJECT_BATCH_FAILURE_FILENAME
    if not path.is_file():
        return None
    try:
        epoch_s, batch_s = path.read_text(encoding="utf-8").strip().split(":")
        return int(epoch_s), int(batch_s)
    except (OSError, ValueError):
        return None


class JobManager:
    def __init__(self, runs_dir: str, default_config: TrainConfig):
        # Bit-exact CPU reproducibility: fixed thread count + deterministic ops.
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover - depends on torch build
            pass

        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.default_config = default_config

        self._cond = threading.Condition()      # guards all mutable fields below
        self._save_lock = threading.Lock()      # serializes checkpoint writes
        self._pause_requested = False
        self._thread: threading.Thread | None = None
        self._resume_rng_state: torch.Tensor | None = None

        self._reset_runtime()
        self._load_dataset()
        self._recover_from_disk()

    # ------------------------------------------------------------------ setup

    def _reset_runtime(self) -> None:
        self.state = JobState.IDLE
        self.cfg: TrainConfig | None = None
        self.run_id: str | None = None
        self.model = None
        self.optimizer = None
        self.epoch = 0            # epoch of the NEXT batch to run
        self.next_batch = 0       # index (within epoch) of the NEXT batch to run
        self.losses: list[dict] = []
        self.completed = False
        self.recovered = False
        self.saving = False
        self.checkpoint_dirty = False   # in-memory progress newer than disk
        self.last_error: str | None = None
        self.recovery_error: str | None = None
        self.last_checkpoint_at: str | None = None
        self.started_at: str | None = None
        self.fail_epoch: int | None = None
        self.fail_batch: int | None = None
        self.resume_attempts = 0
        self._saved_point: tuple[int, int] | None = None
        self._resume_rng_state = None

    def _load_dataset(self) -> None:
        path = self.default_config.dataset_path
        self.X, self.y, self.feature_cols = load_classification_csv(path)
        self.num_samples = int(self.X.shape[0])
        self.input_dim = int(self.X.shape[1])
        self.num_classes = int(self.y.max().item()) + 1
        self.dataset_path = path

    def _build_training_objects(self) -> None:
        self.model = build_model(self.input_dim, self.cfg.hidden_dim,
                                 self.cfg.dropout_p, self.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def _recover_from_disk(self) -> None:
        """Startup path: adopt the on-disk checkpoint if one is usable.

        A corrupt/unreadable checkpoint does not crash the service and never
        leads to a half-restored scene: the file is kept, the reason is stored
        in ``recovery_error``/``last_error`` and resume is refused until a
        usable checkpoint exists.
        """
        removed = cleanup_stale_tmp_files(self.runs_dir)
        if removed:
            print(f"[trainer] removed stale temp checkpoint files: {removed}")
        try:
            ckpt = load_checkpoint(self.runs_dir)
        except CheckpointLoadError as exc:
            self.state = JobState.FAILED
            self.recovery_error = str(exc)
            self.last_error = (
                f"startup recovery refused: no usable checkpoint "
                f"({exc}); original checkpoint kept, inspect the file or "
                f"start a new run with force=true"
            )
            print(f"[trainer] {self.last_error}")
            return
        if ckpt is None:
            return
        try:
            self._apply_loaded_checkpoint(ckpt)
        except CheckpointLoadError as exc:
            # Roll back every partial mutation: stay in a clean idle-ish
            # failed state; nothing from the bad checkpoint is partially live.
            self._reset_runtime()
            self.state = JobState.FAILED
            self.recovery_error = str(exc)
            self.last_error = (
                f"startup recovery refused: checkpoint rejected "
                f"({exc}); original checkpoint kept"
            )
            print(f"[trainer] {self.last_error}")
            return
        self.state = JobState.FINISHED if self.completed else JobState.PAUSED
        self.recovered = True
        print(f"[trainer] recovered checkpoint run={self.run_id} "
              f"epoch={self.epoch} next_batch={self.next_batch} "
              f"completed={self.completed}")

    def _validate_checkpoint(self, ckpt: dict) -> None:
        """Structural validation. Raises CheckpointLoadError on any problem."""
        if not isinstance(ckpt, dict):
            raise CheckpointLoadError(
                f"checkpoint has unexpected type {type(ckpt).__name__}")
        missing = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
        if missing:
            raise CheckpointLoadError(
                f"checkpoint is missing required fields: {missing}")
        if not isinstance(ckpt["run_id"], str) or not ckpt["run_id"]:
            raise CheckpointLoadError("checkpoint has an invalid run_id")
        if not isinstance(ckpt["losses"], list):
            raise CheckpointLoadError("checkpoint losses must be a list")
        if not isinstance(ckpt["epoch"], int) or not isinstance(ckpt["next_batch"], int):
            raise CheckpointLoadError("checkpoint position fields must be ints")
        if ckpt["epoch"] < 0 or ckpt["next_batch"] < 0:
            raise CheckpointLoadError("checkpoint position fields must be >= 0")

    def _apply_loaded_checkpoint(self, ckpt: dict) -> None:
        """Validate and fully apply a checkpoint to the live runtime.

        Either every component (config, model, Adam, RNG, position, history)
        is restored, or CheckpointLoadError is raised and the caller keeps the
        previous scene.  The checkpoint file itself is never modified here.
        """
        self._validate_checkpoint(ckpt)
        try:
            cfg = TrainConfig.from_dict(ckpt["config"])
        except (ValueError, TypeError) as exc:
            raise CheckpointLoadError(f"invalid config in checkpoint: {exc}") from exc

        model = build_model(self.input_dim, cfg.hidden_dim,
                            cfg.dropout_p, self.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            torch.set_rng_state(ckpt["rng_state"])
        except Exception as exc:  # noqa: BLE001 - any mismatch = unusable point
            raise CheckpointLoadError(
                f"failed to restore model/optimizer/RNG state: {exc!r}") from exc

        # Commit only after every restore step above succeeded.
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self._resume_rng_state = ckpt["rng_state"]
        self.epoch = int(ckpt["epoch"])
        self.next_batch = int(ckpt["next_batch"])
        self.losses = list(ckpt["losses"])
        self.completed = bool(ckpt["completed"])
        self.run_id = ckpt["run_id"]
        self.started_at = ckpt.get("started_at")
        self._saved_point = (self.epoch, self.next_batch)
        self.checkpoint_dirty = False
        self.fail_epoch = None
        self.fail_batch = None
        try:
            mtime = checkpoint_path(self.runs_dir).stat().st_mtime
            self.last_checkpoint_at = datetime.fromtimestamp(
                mtime, timezone.utc).isoformat()
        except OSError:
            pass

    # --------------------------------------------------------------- commands

    def start(self, overrides: dict | None = None, force: bool = False) -> dict:
        with self._cond:
            if self.state in (JobState.RUNNING, JobState.PAUSING,
                              JobState.RECOVERING):
                raise ConflictError("training is already running")
            if self.run_id is not None and not force:
                raise ConflictError(
                    "a run already exists; use POST /train/resume to continue it "
                    "or POST /train/start with force=true to discard it")
            cfg = self.default_config.merged(overrides or {})
            if cfg.dataset_path != self.default_config.dataset_path:
                raise ValueError("dataset_path is service-level and cannot be overridden")

            old_run_id = self.run_id
            self._reset_runtime()
            self.cfg = cfg
            torch.manual_seed(cfg.seed)          # seed model init + dropout stream
            self._build_training_objects()
            self.run_id = uuid.uuid4().hex[:12]
            self.started_at = _now_iso()
            # A fresh run replaces any previous checkpoint on disk, so a forced
            # new run can never resume from or mix with the old run's result.
            checkpoint_path(self.runs_dir).unlink(missing_ok=True)
            self.state = JobState.RUNNING
            self._pause_requested = False
            self._thread = threading.Thread(target=self._train_loop,
                                            name="trainer", daemon=True)
            self._thread.start()
            if old_run_id is not None:
                print(f"[trainer] force-started new run {self.run_id} "
                      f"(discarded previous run {old_run_id})")
            return self.status()

    def resume(self) -> dict:
        with self._cond:
            if self.state in (JobState.RUNNING, JobState.PAUSING,
                              JobState.RECOVERING):
                raise ConflictError("training is already running")
            if self.state == JobState.FINISHED:
                raise ConflictError(
                    "run already finished; start a new one with force=true")
            if self.run_id is None or self.cfg is None:
                hint = f" reason: {self.recovery_error}" if self.recovery_error else ""
                raise NotFoundError(
                    "no run or usable checkpoint to resume from;"
                    f" resume is refused.{hint}")
            failed = self.state == JobState.FAILED
            paused = self.state == JobState.PAUSED
            if not (failed or paused):
                raise ConflictError(
                    f"cannot resume while state={self.state.value}")
            target_run_id = self.run_id
            if failed:
                # A failed batch already advanced model/optimizer/RNG without
                # recording progress: the in-memory scene is contaminated and
                # must NOT be reused.  Restore from disk instead.
                ckpt_file = checkpoint_path(self.runs_dir)
                if not ckpt_file.exists():
                    raise NotFoundError(
                        "resume refused: the job failed and no successful "
                        "checkpoint exists for this run, so there is nothing "
                        "to restore from; start a new run with force=true")
                self.resume_attempts += 1
                attempt = self.resume_attempts
                self.state = JobState.RECOVERING
                self.recovery_error = None
                self._cond.notify_all()

        if failed:
            # Slow I/O happens outside the condition lock: status queries stay
            # responsive while the restore is in flight.
            try:
                ckpt = load_checkpoint(self.runs_dir)
                if ckpt is None:
                    raise CheckpointLoadError(
                        "checkpoint disappeared while resuming")
                ckpt_run_id = ckpt.get("run_id") if isinstance(ckpt, dict) else None
                if ckpt_run_id != target_run_id:
                    raise ConflictError(
                        "resume refused: the checkpoint belongs to a different "
                        f"run (checkpoint run_id={ckpt_run_id!r}, "
                        f"failed run_id={target_run_id!r}); the old run's "
                        "recovery result must not be mixed into this job")
                with self._cond:
                    self._apply_loaded_checkpoint(ckpt)
                    # _apply_loaded_checkpoint restores run_id etc.; keep the
                    # attempt counter for the failed job's history.
                    self.resume_attempts = attempt
                    self.last_error = None
                    self.state = JobState.RUNNING
                    self._pause_requested = False
                    self._thread = threading.Thread(
                        target=self._train_loop, name="trainer", daemon=True)
                    self._thread.start()
                    status = self.status()
            except ConflictError:
                with self._cond:
                    self.state = JobState.FAILED
                    self._cond.notify_all()
                raise
            except CheckpointLoadError as exc:
                # Keep the original checkpoint and the failed scene untouched;
                # the reason is queryable and the resume can be retried.
                with self._cond:
                    self.recovery_error = str(exc)
                    self.last_error = (
                        f"resume attempt {attempt} aborted: checkpoint could "
                        f"not be loaded/restored ({exc}); the original "
                        "checkpoint is intact and no training was started")
                    self.state = JobState.FAILED
                    self._cond.notify_all()
                raise
            return status

        # PAUSED: the in-memory scene is the checkpoint state (or newer, when
        # the last save failed but every batch boundary was recorded).
        with self._cond:
            if self._resume_rng_state is not None:
                torch.set_rng_state(self._resume_rng_state)
            self.state = JobState.RUNNING
            self._pause_requested = False
            self.last_error = None
            self._thread = threading.Thread(target=self._train_loop,
                                            name="trainer", daemon=True)
            self._thread.start()
            return self.status()

    def pause(self) -> dict:
        """Stop training and return only after a full checkpoint is persisted."""
        do_inline_retry = False
        with self._cond:
            if self.state == JobState.RUNNING:
                self._pause_requested = True
                self.state = JobState.PAUSING
            elif self.state == JobState.PAUSED and self.checkpoint_dirty:
                do_inline_retry = True      # retry the failed pause-save
            elif self.state == JobState.PAUSED:
                return self.status()        # already paused and persisted
            elif self.state == JobState.PAUSING:
                pass                        # concurrent pause: just wait
            else:
                raise ConflictError(f"cannot pause while state={self.state.value}")

            if not do_inline_retry:
                timeout = self.cfg.pause_timeout_seconds if self.cfg else 30.0
                deadline = time.monotonic() + timeout
                while self.state == JobState.PAUSING:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PauseTimeoutError(
                            "trainer did not stop within "
                            f"{timeout}s; pause is still in progress, "
                            "poll GET /train/status")
                    self._cond.wait(remaining)
                if self.state == JobState.FINISHED:
                    # The last batch completed before the pause landed; the run
                    # is stopped and the final checkpoint is persisted.
                    return self.status()

        # State is PAUSED here.  If the trainer's pause-save failed, retry it
        # once inline (outside the lock, so queries stay responsive).
        if self.checkpoint_dirty:
            try:
                self._save_checkpoint()
            except CheckpointError as exc:
                raise CheckpointSaveError(
                    f"training is stopped but the checkpoint could not be "
                    f"persisted; the previous checkpoint is intact on disk and "
                    f"the pause can be retried. cause: {exc}") from exc
        return self.status()

    # ------------------------------------------------------------- train loop

    def _train_loop(self) -> None:
        try:
            cfg = self.cfg
            n = self.num_samples
            bs = cfg.batch_size
            batches_per_epoch = (n + bs - 1) // bs

            while self.epoch < cfg.epochs:
                epoch = self.epoch
                perm = epoch_permutation(cfg.seed, epoch, n)
                for b in range(self.next_batch, batches_per_epoch):
                    idx = perm[b * bs:(b + 1) * bs]
                    xb = self.X[idx]
                    yb = self.y[idx]
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    logits = self.model(xb)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    self.optimizer.step()
                    loss_value = loss.item()

                    # Test-only transient-fault hook.  It fires AFTER the
                    # optimizer step but BEFORE progress is recorded: the exact
                    # window where resuming naively would re-run the batch on
                    # top of already-updated parameters.
                    if _read_batch_failure_point(self.runs_dir) == (epoch, b):
                        raise RuntimeError(
                            f"injected transient training failure at "
                            f"epoch={epoch} batch={b} after optimizer.step() "
                            f"and before progress recording "
                            f"(loss={loss_value:.6f})")

                    if cfg.batch_sleep_seconds > 0:
                        time.sleep(cfg.batch_sleep_seconds)

                    with self._cond:
                        self.next_batch = b + 1
                        self.losses.append({
                            "step": len(self.losses) + 1,
                            "epoch": epoch,
                            "batch": b,
                            "loss": loss_value,
                        })
                        pause_now = self._pause_requested

                    if (b + 1) % cfg.checkpoint_every == 0 or b == batches_per_epoch - 1:
                        try:
                            self._save_checkpoint()
                        except CheckpointError:
                            pass  # recorded in status; retried at next boundary

                    if pause_now:
                        try:
                            self._save_checkpoint()
                        except CheckpointError:
                            pass  # pause() reports the failure to the caller
                        with self._cond:
                            self._resume_rng_state = torch.get_rng_state()
                            self._pause_requested = False
                            self.state = JobState.PAUSED
                            self._cond.notify_all()
                        return

                with self._cond:
                    self.epoch = epoch + 1
                    self.next_batch = 0

            with self._cond:
                self.completed = True
            try:
                self._save_checkpoint(force=True)
            except CheckpointError:
                with self._cond:
                    self.state = JobState.FAILED
                    self._cond.notify_all()
                return
            with self._cond:
                self.state = JobState.FINISHED
                self._cond.notify_all()
        except Exception as exc:  # noqa: BLE001 - never lose the error
            # The failing batch already consumed RNG draws and an optimizer
            # step, but its progress/loss was NOT recorded.  Leave the scene
            # exactly as-is and mark the job failed; resume must restore the
            # last successful checkpoint rather than reuse this scene.
            with self._cond:
                self.fail_epoch = self.epoch
                self.fail_batch = self.next_batch
                self.last_error = (
                    f"training failed at epoch={self.epoch} "
                    f"batch={self.next_batch}: {exc!r}; the batch progress was "
                    "not recorded. Resume restores the latest successful "
                    "checkpoint of this run and re-runs the failing batch from "
                    "that exact point.")
                self.state = JobState.FAILED
                self._cond.notify_all()

    # -------------------------------------------------------------- checkpoint

    def _save_checkpoint(self, force: bool = False) -> None:
        point = (self.epoch, self.next_batch)
        payload = {
            "run_id": self.run_id,
            "config": self.cfg.to_dict(),
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "completed": self.completed,
            "losses": self.losses,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "started_at": self.started_at,
            "saved_at": _now_iso(),
        }
        with self._save_lock:
            if not force and not self.checkpoint_dirty and self._saved_point == point:
                return  # this exact point is already persisted
            with self._cond:
                self.saving = True
            try:
                save_checkpoint(self.runs_dir, payload,
                                save_delay_seconds=self.cfg.checkpoint_sleep_seconds)
            except CheckpointError as exc:
                with self._cond:
                    self.saving = False
                    self.checkpoint_dirty = True
                    self.last_error = (f"checkpoint save failed at "
                                       f"epoch={point[0]} batch={point[1]}: {exc}")
                raise
            with self._cond:
                self.saving = False
                self.checkpoint_dirty = False
                self.last_error = None
                self._saved_point = point
                self.last_checkpoint_at = _now_iso()

    # ----------------------------------------------------------------- queries

    def status(self) -> dict:
        with self._cond:
            if self.cfg is not None:
                bpe = (self.num_samples + self.cfg.batch_size - 1) // self.cfg.batch_size
                total = self.cfg.epochs * bpe
                done = self.epoch * bpe + self.next_batch
            else:
                bpe = total = done = 0
            ckpt_file = checkpoint_path(self.runs_dir)
            return {
                "state": self.state.value,
                "run_id": self.run_id,
                "recovered_after_restart": self.recovered,
                "epoch": self.epoch,
                "next_batch": self.next_batch,
                "failed_at": (None if self.fail_epoch is None
                              else {"epoch": self.fail_epoch,
                                    "batch": self.fail_batch}),
                "resume_attempts": self.resume_attempts,
                "recoverable": (
                    self.state == JobState.FAILED
                    and ckpt_file.exists()
                ),
                "batches_per_epoch": bpe,
                "total_batches": total,
                "completed_batches": done,
                "progress": (done / total) if total else 0.0,
                "saving": self.saving,
                "checkpoint_dirty": self.checkpoint_dirty,
                "last_checkpoint_at": self.last_checkpoint_at,
                "last_error": self.last_error,
                "recovery_error": self.recovery_error,
                "started_at": self.started_at,
                "config": self.cfg.to_dict() if self.cfg else None,
                "dataset": {
                    "path": self.dataset_path,
                    "num_samples": self.num_samples,
                    "num_features": self.input_dim,
                    "num_classes": self.num_classes,
                },
            }

    def get_losses(self, since_step: int | None = None, limit: int | None = None) -> dict:
        with self._cond:
            losses = list(self.losses)
            total = len(losses)
        if since_step is not None:
            losses = [l for l in losses if l["step"] > since_step]
        if limit is not None:
            losses = losses[:limit]
        return {"count": total, "returned": len(losses), "losses": losses}

    def read_parameters(self) -> bytes:
        """Model parameters from the latest persisted checkpoint, as a .pt file."""
        ckpt = load_checkpoint(self.runs_dir)
        if ckpt is None:
            raise NotFoundError("no checkpoint exists yet; train first")
        buf = io.BytesIO()
        torch.save(ckpt["model_state_dict"], buf)
        return buf.getvalue()

    def checkpoint_file(self) -> Path:
        path = checkpoint_path(self.runs_dir)
        if not path.exists():
            raise NotFoundError("no checkpoint exists yet; train first")
        return path
