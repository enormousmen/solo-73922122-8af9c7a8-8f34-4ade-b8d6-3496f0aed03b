"""Atomic checkpoint persistence and strict loading.

A checkpoint is a single ``checkpoint.pt`` file inside the runs directory.
Saves are atomic on POSIX filesystems:

1. serialize to ``.checkpoint.pt.tmp.<uuid>`` in the same directory,
2. flush + fsync the file,
3. ``os.replace()`` it over ``checkpoint.pt`` (atomic rename),
4. fsync the directory so the rename itself is durable.

If the process is killed at any point, ``checkpoint.pt`` is therefore either
the complete previous checkpoint or the complete new one — never a torn file.
A failed save (e.g. disk full) leaves the previous checkpoint untouched so it
can be used for a retry or for recovery after a restart.

Loading never mutates or removes the file: an unreadable / corrupt checkpoint
raises :class:`CheckpointLoadError` and leaves the original bytes in place so
the failure reason stays queryable and the file can be inspected or retried.

TEST-ONLY HOOKS (used by scripts/verify_repro.py, never present in normal
operation; all driven by marker files in the runs directory):

* ``INJECT_SAVE_FAILURES``       - every save fails before touching anything.
* ``INJECT_LOAD_FAILURES``       - every checkpoint load raises
  :class:`CheckpointLoadError` (the file itself is kept).
* ``INJECT_LOAD_DELAY_SECONDS``  - sleep N seconds inside a load, to prove
  that a slow recovery never blocks status queries.
* ``INJECT_BATCH_FAILURE_AT``    - handled by the trainer (see trainer.py).
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import torch

CHECKPOINT_FILENAME = "checkpoint.pt"
INJECT_FAILURES_FILENAME = "INJECT_SAVE_FAILURES"
INJECT_LOAD_FAILURES_FILENAME = "INJECT_LOAD_FAILURES"
INJECT_LOAD_DELAY_FILENAME = "INJECT_LOAD_DELAY_SECONDS"


class CheckpointError(IOError):
    """Raised when a checkpoint could not be persisted."""


class CheckpointLoadError(IOError):
    """Raised when an existing checkpoint cannot be read/parsed.

    The original checkpoint file is deliberately left untouched, so callers
    can surface a queryable failure reason and retry later.
    """


def checkpoint_path(runs_dir) -> Path:
    return Path(runs_dir) / CHECKPOINT_FILENAME


def save_checkpoint(runs_dir, payload: dict, save_delay_seconds: float = 0.0) -> Path:
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)

    # --- test-only fault injection (see module docstring) ---
    if (runs / INJECT_FAILURES_FILENAME).exists():
        raise CheckpointError(
            f"checkpoint save failed: injected fault "
            f"('{INJECT_FAILURES_FILENAME}' file present in {runs})"
        )
    # ---------------------------------------------------------

    final = runs / CHECKPOINT_FILENAME
    tmp = runs / f".{CHECKPOINT_FILENAME}.tmp.{uuid.uuid4().hex}"
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        if save_delay_seconds > 0:
            # Simulates slow storage.  The window between "new file fully
            # written" and "atomic rename" is where a kill must still leave
            # the OLD checkpoint intact.
            time.sleep(save_delay_seconds)
        os.replace(tmp, final)
        _fsync_dir(runs)
    except CheckpointError:
        raise
    except Exception as exc:  # noqa: BLE001 - report any persistence failure
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise CheckpointError(f"failed to persist checkpoint to {final}: {exc!r}") from exc
    return final


def _injected_load_delay(runs: Path) -> None:
    delay_file = runs / INJECT_LOAD_DELAY_FILENAME
    if not delay_file.is_file():
        return
    try:
        delay = float(delay_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    if delay > 0:
        time.sleep(delay)


def load_checkpoint(runs_dir) -> dict | None:
    """Load the checkpoint if one exists; ``None`` otherwise.

    Only complete checkpoints can be observed thanks to the atomic rename.
    A present-but-unreadable file raises :class:`CheckpointLoadError`; the
    file is never deleted or truncated by this function.
    """
    runs = Path(runs_dir)
    final = checkpoint_path(runs)
    if not final.exists():
        return None

    # --- test-only fault injection (see module docstring) ---
    if (runs / INJECT_LOAD_FAILURES_FILENAME).exists():
        raise CheckpointLoadError(
            f"checkpoint load failed: injected fault "
            f"('{INJECT_LOAD_FAILURES_FILENAME}' file present in {runs}); "
            f"original checkpoint kept at {final}"
        )
    _injected_load_delay(runs)
    # ---------------------------------------------------------

    try:
        obj = torch.load(final, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a load error
        raise CheckpointLoadError(
            f"checkpoint at {final} is unreadable or corrupt: {exc!r}; "
            f"the original file was kept at {final} and can be retried"
        ) from exc
    if not isinstance(obj, dict):
        raise CheckpointLoadError(
            f"checkpoint at {final} has unexpected top-level type "
            f"{type(obj).__name__}; original file kept"
        )
    return obj


def cleanup_stale_tmp_files(runs_dir) -> list[str]:
    """Remove leftover temp files from saves interrupted by a kill."""
    removed = []
    runs = Path(runs_dir)
    if runs.is_dir():
        for tmp in runs.glob(f".{CHECKPOINT_FILENAME}.tmp.*"):
            try:
                tmp.unlink()
                removed.append(tmp.name)
            except OSError:
                pass
    return removed


def _fsync_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
