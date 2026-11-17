#!/usr/bin/env python3
"""End-to-end verification of bit-exact resumable training.

Spins up real uvicorn server processes and checks, against an uninterrupted
baseline (identical seed + config):

  A   uninterrupted baseline run (reference)
  B   pause + resume                  -> losses & final params bit-identical to A
  C   hard crash + restart + resume   -> bit-identical to A
  D   two concurrent resumes          -> exactly one 200, one 409; result == A
  E1  checkpoint save failure         -> pause reports 500, old checkpoint kept,
                                         retry succeeds, resume reproduces A
  E2  save failure, then hard crash   -> restart recovers the OLD checkpoint,
                                         resume still reproduces A
  F   slow checkpoint saves           -> status queries stay fast while saving
  G   hard crash in the middle of a save -> restart finds a complete (old or
                                         new) checkpoint, resume reproduces
                                         G-baseline
  H   service basics                  -> /, /docs, 404s before first checkpoint
  I1  transient failure INSIDE a batch (optimizer stepped, progress not
      recorded) -> job failed; resume restores the latest successful
      checkpoint of the same run and re-runs the batch -> bit-identical to A
  I2  failure before the first checkpoint -> resume is refused (404), no
      half-restored scene
  I3  failed job whose checkpoint cannot be loaded -> resume 503, original
      checkpoint kept, reason queryable; after the fault clears, retry works
  I4  slow recovery -> status stays responsive while state=recovering and a
      concurrent resume gets 409; the single trainer reproduces A
  I5  force=true starts a brand-new run that never mixes with the old run
  J   in-process unit checks: corrupt / tampered / incomplete checkpoints are
      rejected at startup and at failed-job resume, originals always kept

Exit code 0 iff every scenario passes.  Run from the repo root:

    python3 scripts/verify_repro.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = sys.executable

# Shared by scenarios A/B/C/D/E/I (bit-exact comparison target).
CFG_MAIN = {"epochs": 5, "batch_size": 32, "batch_sleep_seconds": 0.03,
            "checkpoint_every": 2, "checkpoint_sleep_seconds": 0.0}
# Scenario F: slow saves, short run.
CFG_SLOW_SAVE = {"epochs": 1, "batch_size": 32, "batch_sleep_seconds": 0.03,
                 "checkpoint_every": 1, "checkpoint_sleep_seconds": 1.0}
# Scenario G: kill mid-save, short run.
CFG_KILL_SAVE = {"epochs": 1, "batch_size": 32, "batch_sleep_seconds": 0.03,
                 "checkpoint_every": 1, "checkpoint_sleep_seconds": 0.5}

# Test-only marker files inside the runs directory (see app/checkpoint.py and
# app/trainer.py docstrings).
INJECT_BATCH_FAILURE = "INJECT_BATCH_FAILURE_AT"
INJECT_SAVE_FAILURES = "INJECT_SAVE_FAILURES"
INJECT_LOAD_FAILURES = "INJECT_LOAD_FAILURES"
INJECT_LOAD_DELAY = "INJECT_LOAD_DELAY_SECONDS"

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str = "") -> None:
    RESULTS.append((name, ok, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {note}" if note else ""), flush=True)


# --------------------------------------------------------------------- http

def api(port: int, method: str, path: str, body: dict | None = None,
        timeout: float = 120.0) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"detail": raw.decode(errors="replace")}


def api_bytes(port: int, path: str) -> bytes:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as resp:
        return resp.read()


# ------------------------------------------------------------------ servers

class Server:
    def __init__(self, runs_dir: Path, log_dir: Path, tag: str):
        self.runs_dir = runs_dir
        self.port = _free_port()
        self.tag = tag
        log = open(log_dir / f"server-{tag}.log", "wb")
        env = dict(os.environ)
        env["RUNS_DIR"] = str(runs_dir)
        env["CONFIG_PATH"] = str(ROOT / "config" / "default.json")
        env["ENABLE_TEST_CRASH_ENDPOINT"] = "1"
        self.proc = subprocess.Popen(
            [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self._log_path = log_dir / f"server-{tag}.log"
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server {self.tag} exited early; log:\n"
                    + self._log_path.read_text(errors="replace")[-3000:])
            try:
                code, _ = api(self.port, "GET", "/train/status", timeout=2)
                if code == 200:
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.3)
        raise RuntimeError(f"server {self.tag} did not become ready")

    def kill(self) -> None:
        """Sudden death via the test-only crash endpoint (os._exit).

        Equivalent to SIGKILL for recovery purposes: the process dies
        immediately, wherever it is (including mid-checkpoint-save), with no
        cleanup.  (Real SIGKILL cannot be used in every environment; Docker
        users can equally use `docker kill`, see README.)
        """
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/test/crash", data=b"", timeout=5)
        except Exception:
            pass  # the connection drops when the process dies
        self.proc.wait(timeout=10)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.kill()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ helpers

def wait_state(port: int, want: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and st["state"] == want:
            return st
        time.sleep(0.2)
    raise RuntimeError(f"state did not become {want!r} within {timeout}s")


def wait_until(port: int, pred, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and pred(st):
            return st
        time.sleep(0.05)
    raise RuntimeError("condition not met within timeout")


def start_run(port: int, cfg: dict) -> dict:
    code, st = api(port, "POST", "/train/start", {"force": True, "config": cfg})
    assert code == 200, f"start failed: {st}"
    return st


def run_to_finish(port: int, cfg: dict) -> tuple[list, bytes]:
    start_run(port, cfg)
    st = wait_state(port, "finished")
    assert st["completed_batches"] == st["total_batches"], st
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    return losses["losses"], params


def compare_to_baseline(name: str, ref_losses: list, ref_params: bytes,
                        got_losses: list, got_params: bytes) -> bool:
    ok = True
    if got_losses == ref_losses:
        print(f"    losses: {len(got_losses)} entries, exact match")
    else:
        ok = False
        n = min(len(got_losses), len(ref_losses))
        diff = next((i for i in range(n) if got_losses[i] != ref_losses[i]), None)
        print(f"    losses: MISMATCH (len {len(got_losses)} vs {len(ref_losses)}, "
              f"first diff at index {diff})")
        if diff is not None:
            print(f"      got: {got_losses[diff]}\n      ref: {ref_losses[diff]}")
    ref_sd = torch.load(io.BytesIO(ref_params), weights_only=True)
    got_sd = torch.load(io.BytesIO(got_params), weights_only=True)
    if ref_sd.keys() != got_sd.keys():
        ok = False
        print(f"    params: key mismatch {sorted(got_sd)} vs {sorted(ref_sd)}")
    elif all(torch.equal(got_sd[k], ref_sd[k]) for k in ref_sd):
        h_ref = hashlib.sha256(ref_params).hexdigest()[:16]
        h_got = hashlib.sha256(got_params).hexdigest()[:16]
        print(f"    params: all tensors bit-identical "
              f"(sha256 {h_got}{' ==' if h_got == h_ref else ' !='} {h_ref})")
    else:
        ok = False
        bad = [k for k in ref_sd if not torch.equal(got_sd[k], ref_sd[k])]
        print(f"    params: MISMATCH in tensors {bad}")
    record(name, ok)
    return ok


# ------------------------------------------------------------------ scenarios

def scenario_basics(port: int) -> None:
    print("\n== H: service basics (idle server) ==")
    code, _ = api(port, "GET", "/")
    record("H1 GET /", code == 200)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs",
                                    timeout=10) as resp:
            docs_ok = resp.status == 200 and b"swagger" in resp.read().lower()
    except Exception:
        docs_ok = False
    record("H2 GET /docs (Swagger UI)", docs_ok)
    code, _ = api(port, "GET", "/train/parameters")
    record("H3 parameters before any checkpoint -> 404", code == 404)
    code, _ = api(port, "POST", "/train/resume")
    record("H4 resume with no run -> 404", code == 404)
    code, _ = api(port, "POST", "/train/pause")
    record("H5 pause while idle -> 409", code == 409)


def scenario_pause_resume(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== B: pause + resume ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 12)
    t0 = time.monotonic()
    code, st = api(port, "POST", "/train/pause")
    dt = time.monotonic() - t0
    ok = (code == 200 and st["state"] == "paused" and not st["checkpoint_dirty"]
          and st["last_checkpoint_at"] is not None)
    record("B1 pause returns only after checkpoint persisted", ok,
           f"pause latency {dt:.2f}s, stopped at batch {st['completed_batches']}")
    code, st = api(port, "POST", "/train/resume")
    record("B2 resume accepted", code == 200 and st["state"] == "running")
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("B3 pause+resume == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)


def scenario_concurrent_resume(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== D: concurrent resume starts exactly one trainer ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 8)
    api(port, "POST", "/train/pause")
    codes: list[int] = []

    def fire():
        code, _ = api(port, "POST", "/train/resume")
        codes.append(code)

    threads = [threading.Thread(target=fire) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    record("D1 concurrent resumes -> one 200, one 409",
           sorted(codes) == [200, 409], f"got {sorted(codes)}")
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("D2 resumed run == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)


def scenario_save_failure_retry(port: int, runs_dir: Path,
                                ref: tuple[list, bytes]) -> None:
    print("\n== E1: checkpoint save failure keeps old checkpoint, retry works ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 6
               and s["last_checkpoint_at"] is not None)
    flag = runs_dir / "INJECT_SAVE_FAILURES"
    flag.write_text("1")                      # arm test-only fault injection
    time.sleep(0.3)                           # let any in-flight save finish
    ckpt = runs_dir / "checkpoint.pt"
    hash_before = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    code, body = api(port, "POST", "/train/pause")
    record("E1 pause with failing save -> 500", code == 500,
           str(body.get("detail", ""))[:100])
    code, st = api(port, "GET", "/train/status")
    record("E1 status shows dirty checkpoint + error",
           st["state"] == "paused" and st["checkpoint_dirty"]
           and "injected fault" in (st["last_error"] or ""))
    hash_after = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    record("E1 old checkpoint untouched", hash_before == hash_after)
    flag.unlink()                             # fix the "storage"
    code, st = api(port, "POST", "/train/pause")
    record("E1 retried pause -> 200, checkpoint persisted",
           code == 200 and not st["checkpoint_dirty"])
    code, _ = api(port, "POST", "/train/resume")
    record("E1 resume after retry", code == 200)
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("E1 run after save-failure == uninterrupted",
                        ref[0], ref[1], losses["losses"], params)


def scenario_slow_save_queries(port: int) -> None:
    print("\n== F: slow checkpoint saves do not block queries ==")
    start_run(port, CFG_SLOW_SAVE)
    wait_until(port, lambda s: s["saving"], timeout=30)
    lat = []
    for _ in range(3):
        t0 = time.monotonic()
        code, _ = api(port, "GET", "/train/status")
        lat.append(time.monotonic() - t0)
    code_l, _ = api(port, "GET", "/train/losses")
    worst = max(lat)
    record("F1 queries respond fast while a 1s save is in flight",
           worst < 1.0 and code == 200 and code_l == 200,
           f"worst status latency {worst*1000:.0f}ms")
    code, st = api(port, "POST", "/train/pause", timeout=60)
    record("F2 pause during slow saves", code == 200 and st["state"] == "paused")
    api(port, "POST", "/train/resume")
    st = wait_state(port, "finished", timeout=120)
    record("F3 slow-save run finishes", st["state"] == "finished")


def scenario_kill_restart(runs_dir: Path, log_dir: Path, server: Server,
                          ref: tuple[list, bytes]) -> Server:
    print("\n== C: hard crash + restart recovers from latest checkpoint ==")
    start_run(server.port, CFG_MAIN)
    wait_until(server.port, lambda s: s["completed_batches"] >= 10
               and s["last_checkpoint_at"] is not None)
    server.kill()
    record("C1 server crashed mid-run", True)
    server2 = Server(runs_dir, log_dir, "C-restarted")
    code, st = api(server2.port, "GET", "/train/status")
    record("C2 restart recovers paused state from checkpoint",
           st["state"] == "paused" and st["recovered_after_restart"]
           and 0 < st["completed_batches"] < st["total_batches"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("C3 resume after restart", code == 200)
    wait_state(server2.port, "finished")
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("C4 kill+restart == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)
    return server2


def scenario_save_failure_then_kill(runs_dir: Path, log_dir: Path, server: Server,
                                    ref: tuple[list, bytes]) -> Server:
    print("\n== E2: save failure, then hard crash -> old checkpoint used ==")
    start_run(server.port, CFG_MAIN)
    wait_until(server.port, lambda s: s["completed_batches"] >= 6
               and s["last_checkpoint_at"] is not None)
    flag = runs_dir / "INJECT_SAVE_FAILURES"
    flag.write_text("1")
    time.sleep(0.3)
    code, _ = api(server.port, "POST", "/train/pause")
    record("E2 pause with failing save -> 500", code == 500)
    server.kill()                              # die while dirty
    flag.unlink()
    server2 = Server(runs_dir, log_dir, "E2-restarted")
    code, st = api(server2.port, "GET", "/train/status")
    record("E2 restart falls back to old complete checkpoint",
           st["state"] == "paused" and st["recovered_after_restart"]
           and not st["checkpoint_dirty"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("E2 resume after restart", code == 200)
    wait_state(server2.port, "finished")
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("E2 save-fail+kill == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)
    return server2


def scenario_kill_mid_save(runs_dir: Path, log_dir: Path, server: Server) -> Server:
    print("\n== G: hard crash in the middle of a checkpoint save ==")
    print("  (baseline: identical config, uninterrupted)")
    ref_losses, ref_params = run_to_finish(server.port, CFG_KILL_SAVE)
    start_run(server.port, CFG_KILL_SAVE)
    # Crash while a save is in flight AND an earlier checkpoint of this run
    # already exists on disk (completed_batches >= 2 means the save for
    # batch 1 finished and the save for batch >= 2 is now running).
    wait_until(server.port,
               lambda s: s["saving"] and s["completed_batches"] >= 2, timeout=30)
    server.kill()                              # crash inside the save window
    record("G1 server crashed during save", True)
    server2 = Server(runs_dir, log_dir, "G-restarted")
    leftovers = [p.name for p in runs_dir.glob(".checkpoint.pt.tmp.*")]
    record("G2 no torn checkpoint / stale temp cleaned", leftovers == [],
           f"leftovers: {leftovers or 'none'}")
    code, st = api(server2.port, "GET", "/train/status")
    record("G3 restart recovers a complete checkpoint (old or new)",
           st["state"] == "paused" and st["recovered_after_restart"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("G3b resume after restart", code == 200)
    wait_state(server2.port, "finished", timeout=120)
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("G4 kill-mid-save == uninterrupted", ref_losses, ref_params,
                        losses["losses"], params)
    return server2


# --------------------------------------------------- failed-job recovery (I)

def scenario_failed_batch_then_resume(port: int, runs_dir: Path,
                                      ref: tuple[list, bytes]) -> None:
    """I1: failure inside a batch (params updated, progress unrecorded).

    The contaminated in-memory scene must be discarded; resume restores the
    latest successful checkpoint of the SAME run and re-runs the failing
    batch from that exact point -> bit-identical to the uninterrupted run.
    """
    print("\n== I1: in-batch transient failure -> restore checkpoint, retry ==")
    start_run(port, CFG_MAIN)
    # Fail at epoch 0 batch 5. With checkpoint_every=2, the latest successful
    # checkpoint covers batches 0..3 (next_batch=4); batches 4..5 must be
    # re-run from that point after resume.
    marker = runs_dir / INJECT_BATCH_FAILURE
    marker.write_text("0:5")
    st = wait_state(port, "failed")
    record("I1 job entered failed state after in-batch error",
           st["state"] == "failed" and st["failed_at"] == {"epoch": 0, "batch": 5}
           and "not recorded" in (st["last_error"] or ""),
           f"failed_at={st.get('failed_at')}")
    record("I1 failure is queryable and a checkpoint exists",
           st["recoverable"] is True and st["last_checkpoint_at"] is not None)
    # The failing batch's own progress was never recorded: the in-memory
    # position points AT the failing batch (next_batch=5), while the last
    # successful checkpoint on disk still covers only batches 0..3
    # (next_batch=4) — so resume rolls back and re-runs batches 4 and 5.
    code, st = api(port, "GET", "/train/status")
    on_disk = torch.load(runs_dir / "checkpoint.pt", map_location="cpu",
                         weights_only=True)
    record("I1 no phantom progress for the failing batch; disk point is older",
           st["completed_batches"] == 5 and on_disk["next_batch"] == 4
           and st["next_batch"] == 5,
           f"memory next_batch={st['next_batch']}, "
           f"disk next_batch={on_disk['next_batch']}")

    # Remove the transient fault, then resume: full restore from checkpoint.
    marker.unlink()
    code, st = api(port, "POST", "/train/resume")
    record("I1 resume accepted after failure", code == 200 and st["state"] == "running",
           f"code={code} detail={st.get('detail')}")
    # After the restore, the position must be the checkpoint position (4),
    # proving the contaminated scene was rolled back rather than reused.
    wait_until(port, lambda s: s["state"] == "running" and s["completed_batches"] >= 5)
    code, st = api(port, "GET", "/train/status")
    record("I1 resumed from the last successful checkpoint (batch re-run)",
           st["resume_attempts"] == 1 and st["failed_at"] is None
           and not st["checkpoint_dirty"])
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("I1 failure+resume == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)


def scenario_failure_before_first_checkpoint(port: int, runs_dir: Path) -> None:
    """I2: with no successful checkpoint, resume must be explicitly refused."""
    print("\n== I2: failure before first checkpoint -> resume refused ==")
    cfg = dict(CFG_MAIN, checkpoint_every=100)  # no checkpoint within epoch 0
    start_run(port, cfg)
    (runs_dir / INJECT_BATCH_FAILURE).write_text("0:2")
    st = wait_state(port, "failed")
    ckpt = runs_dir / "checkpoint.pt"
    record("I2 failed with no checkpoint on disk",
           st["state"] == "failed" and not ckpt.exists())
    code, body = api(port, "POST", "/train/resume")
    record("I2 resume refused with 404 and a clear reason",
           code == 404 and "no successful checkpoint" in body.get("detail", ""),
           str(body.get("detail", ""))[:110])
    code, st = api(port, "GET", "/train/status")
    record("I2 job stays failed (no half-restored scene)",
           st["state"] == "failed" and st["recoverable"] is False)
    # A forced fresh run is the only way out and must not reuse the old job.
    (runs_dir / INJECT_BATCH_FAILURE).unlink()
    code, st = api(port, "POST", "/train/start",
                   {"force": True, "config": CFG_MAIN})
    record("I2 force start accepted and discards the failed run",
           code == 200 and st["state"] == "running")
    wait_state(port, "finished")


def scenario_load_failure_keeps_checkpoint(port: int, runs_dir: Path,
                                           ref: tuple[list, bytes]) -> None:
    """I3: failed job whose checkpoint cannot be loaded.

    Resume must fail closed (503), keep the original checkpoint and expose a
    queryable reason; once the fault clears, a retry fully restores and
    reproduces the uninterrupted run.
    """
    print("\n== I3: unloadable checkpoint -> resume 503, original kept, retry ==")
    start_run(port, CFG_MAIN)
    (runs_dir / INJECT_BATCH_FAILURE).write_text("0:7")
    wait_state(port, "failed")
    (runs_dir / INJECT_BATCH_FAILURE).unlink()
    ckpt = runs_dir / "checkpoint.pt"
    hash_before = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    (runs_dir / INJECT_LOAD_FAILURES).write_text("1")
    code, body = api(port, "POST", "/train/resume")
    record("I3 resume with unloadable checkpoint -> 503",
           code == 503 and "kept" in body.get("detail", ""),
           str(body.get("detail", ""))[:110])
    hash_after = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    record("I3 original checkpoint byte-identical after failed resume",
           hash_before == hash_after)
    code, st = api(port, "GET", "/train/status")
    record("I3 job still failed, reason queryable, nothing started",
           st["state"] == "failed" and st["resume_attempts"] == 1
           and "could not be loaded" in (st["last_error"] or "")
           and st["recovery_error"] is not None)
    (runs_dir / INJECT_LOAD_FAILURES).unlink()   # storage repaired
    code, st = api(port, "POST", "/train/resume")
    record("I3 retry after fault clears -> 200",
           code == 200 and st["state"] == "running", f"code={code}")
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("I3 load-failure then retry == uninterrupted",
                        ref[0], ref[1], losses["losses"], params)


def scenario_slow_recovery_queries_and_single_trainer(
        port: int, runs_dir: Path, ref: tuple[list, bytes]) -> None:
    """I4: slow restore keeps queries live; concurrent resumes -> one trainer."""
    print("\n== I4: slow recovery -> queries live, one trainer only ==")
    start_run(port, CFG_MAIN)
    (runs_dir / INJECT_BATCH_FAILURE).write_text("0:3")
    wait_state(port, "failed")
    (runs_dir / INJECT_BATCH_FAILURE).unlink()
    (runs_dir / INJECT_LOAD_DELAY).write_text("1.5")
    # Fire the resume in a thread: the 1.5s restore happens inside this call.
    resume_codes: list[int] = []
    t_resume = threading.Thread(
        target=lambda: resume_codes.append(api(port, "POST", "/train/resume")[0]))
    t_resume.start()
    saw_recovering = False
    latencies = []
    second_code = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        code, st = api(port, "GET", "/train/status", timeout=5)
        latencies.append(time.monotonic() - t0)
        if st["state"] == "recovering":
            saw_recovering = True
            # A second resume while recovery is in flight must be rejected.
            if second_code is None:
                second_code, second_body = api(port, "POST", "/train/resume")
            break
        time.sleep(0.02)
    t_resume.join(timeout=30)
    record("I4 recovering state was observable during restore", saw_recovering)
    record("I4 status responds fast while recovery is in flight",
           bool(latencies) and max(latencies) < 1.0,
           f"worst latency {max(latencies)*1000:.0f}ms" if latencies else "no samples")
    record("I4 concurrent resume during recovery -> 409",
           second_code == 409, str(second_body.get("detail", ""))[:80])
    record("I4 original resume eventually accepted -> 200",
           resume_codes == [200], f"codes={resume_codes}")
    (runs_dir / INJECT_LOAD_DELAY).unlink()
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("I4 slow-recovery run == uninterrupted",
                        ref[0], ref[1], losses["losses"], params)


def scenario_force_new_run_isolation(port: int, runs_dir: Path,
                                     ref: tuple[list, bytes]) -> None:
    """I5: force=true starts a fresh run and never mixes with the old one."""
    print("\n== I5: forced new run never reuses the failed run's recovery ==")
    start_run(port, CFG_MAIN)
    (runs_dir / INJECT_BATCH_FAILURE).write_text("0:5")
    st = wait_state(port, "failed")
    old_run_id = st["run_id"]
    (runs_dir / INJECT_BATCH_FAILURE).unlink()
    code, st = api(port, "POST", "/train/start",
                   {"force": True, "config": CFG_MAIN})
    record("I5 force start replaces the failed run",
           code == 200 and st["run_id"] != old_run_id
           and st["resume_attempts"] == 0 and st["failed_at"] is None,
           f"old={old_run_id} new={st['run_id']}")
    record("I5 old checkpoint removed before the new run starts",
           not (runs_dir / "checkpoint.pt").exists())
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("I5 forced new run == uninterrupted",
                        ref[0], ref[1], losses["losses"], params)


# --------------------------------------------------------- unit checks (J)

def _unit_manager(runs_dir: Path):
    """Build an in-process JobManager with an absolute dataset path."""
    from app.config import TrainConfig
    from app.trainer import JobManager
    cfg = TrainConfig().merged({
        "dataset_path": str(ROOT / "data" / "classification.csv"),
        "epochs": 2, "batch_size": 32, "batch_sleep_seconds": 0.0,
        "checkpoint_every": 2,
    })
    return JobManager(runs_dir=str(runs_dir), default_config=cfg)


def scenario_unit_checkpoint_guards() -> None:
    print("\n== J: checkpoint guard unit checks (corrupt/tampered/incomplete) ==")
    from app.checkpoint import CheckpointLoadError
    from app.trainer import JobManager, ConflictError, NotFoundError

    work = Path(tempfile.mkdtemp(prefix="verify-unit-"))
    runs = work / "runs"
    runs.mkdir()
    mgr = _unit_manager(runs)
    mgr.start(force=True)
    deadline = time.monotonic() + 60
    while mgr.status()["state"] != "finished" and time.monotonic() < deadline:
        time.sleep(0.05)
    good_bytes = (runs / "checkpoint.pt").read_bytes()
    record("J0 unit harness produced a complete checkpoint", len(good_bytes) > 0)

    # J1: garbage file -> startup refuses, keeps the file, reason queryable.
    (runs / "checkpoint.pt").write_bytes(b"not-a-torch-file" * 8)
    mgr2 = _unit_manager(runs)
    st2 = mgr2.status()
    record("J1 corrupt checkpoint -> failed at startup, no half-restore",
           st2["state"] == "failed" and st2["recovery_error"] is not None
           and (runs / "checkpoint.pt").read_bytes() == b"not-a-torch-file" * 8)
    try:
        mgr2.resume()
        j1 = False
        j1_note = "no exception"
    except NotFoundError as exc:
        # No usable point exists (run_id unknown), so resume is refused and
        # the stored recovery reason is surfaced in the 404 detail.
        j1 = "resume is refused" in str(exc) and "unreadable or corrupt" in str(exc)
        j1_note = str(exc)[:90]
    record("J1 resume on corrupt checkpoint refused, reason surfaced",
           j1, j1_note)

    # J2: checkpoint from a DIFFERENT run_id -> failed-job resume refuses to mix.
    (runs / "checkpoint.pt").write_bytes(good_bytes)
    mgr3 = _unit_manager(runs)
    st3 = mgr3.status()
    record("J2 good checkpoint recovers a resumable state (paused/finished)",
           st3["state"] in ("paused", "finished") and st3["run_id"] is not None,
           f"state={st3['state']}")
    # Simulate a failed job whose in-memory run_id no longer matches disk.
    mgr3.state = type(mgr3.state).FAILED
    mgr3.run_id = "foreignrun999"
    try:
        mgr3.resume()
        j2 = False
    except ConflictError as exc:
        j2 = "different run" in str(exc)
    record("J2 foreign-run checkpoint -> 409, no mixing of recovery results", j2)
    st3b = mgr3.status()
    record("J2 job stays failed and original checkpoint is untouched",
           st3b["state"] == "failed"
           and (runs / "checkpoint.pt").read_bytes() == good_bytes
           and st3b["run_id"] == "foreignrun999")

    # J3: structurally incomplete checkpoint -> rejected, file kept.
    good = torch.load(io.BytesIO(good_bytes), map_location="cpu",
                      weights_only=True)
    del good["optimizer_state_dict"]
    buf = io.BytesIO()
    torch.save(good, buf)
    (runs / "checkpoint.pt").write_bytes(buf.getvalue())
    mgr4 = _unit_manager(runs)
    st4 = mgr4.status()
    record("J3 incomplete checkpoint -> failed at startup, reason queryable",
           st4["state"] == "failed"
           and "missing required fields" in (st4["recovery_error"] or ""))
    try:
        mgr4.resume()
        j3 = False
        j3_note = "no exception"
    except NotFoundError as exc:
        j3 = "no run or usable checkpoint" in str(exc)
        j3_note = str(exc)[:90]
    record("J3 resume on incomplete checkpoint refused", j3, j3_note)

    # J4: no checkpoint at all -> resume is refused (404), nothing started.
    (runs / "checkpoint.pt").unlink()
    mgr5 = _unit_manager(runs)
    try:
        mgr5.resume()
        j4 = False
    except NotFoundError as exc:
        j4 = "no run or usable checkpoint" in str(exc)
    record("J4 resume with no run/checkpoint -> 404 refused", j4)
    record("J4 idle scene untouched", mgr5.status()["state"] == "idle")
    shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------- main

def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="verify-repro-"))
    runs_dir = work / "runs"
    log_dir = work / "logs"
    log_dir.mkdir()
    print(f"workdir: {work}")
    server = Server(runs_dir, log_dir, "main")
    try:
        scenario_basics(server.port)

        print("\n== A: uninterrupted baseline ==")
        ref_losses, ref_params = run_to_finish(server.port, CFG_MAIN)
        record("A baseline finished", True,
               f"{len(ref_losses)} batches, final loss {ref_losses[-1]['loss']:.6f}")

        scenario_pause_resume(server.port, (ref_losses, ref_params))
        scenario_concurrent_resume(server.port, (ref_losses, ref_params))
        scenario_save_failure_retry(server.port, runs_dir, (ref_losses, ref_params))
        scenario_slow_save_queries(server.port)

        server = scenario_kill_restart(runs_dir, log_dir, server,
                                       (ref_losses, ref_params))
        server = scenario_save_failure_then_kill(runs_dir, log_dir, server,
                                                 (ref_losses, ref_params))
        server = scenario_kill_mid_save(runs_dir, log_dir, server)

        # Failed-job recovery semantics (the core fix).
        scenario_failed_batch_then_resume(server.port, runs_dir,
                                          (ref_losses, ref_params))
        scenario_failure_before_first_checkpoint(server.port, runs_dir)
        scenario_load_failure_keeps_checkpoint(server.port, runs_dir,
                                               (ref_losses, ref_params))
        scenario_slow_recovery_queries_and_single_trainer(
            server.port, runs_dir, (ref_losses, ref_params))
        scenario_force_new_run_isolation(server.port, runs_dir,
                                         (ref_losses, ref_params))
        scenario_unit_checkpoint_guards()
    finally:
        server.stop()

    print("\n================ SUMMARY ================")
    failed = 0
    for name, ok, note in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1
    print(f"  {len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    if failed:
        print(f"  server logs under {log_dir}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
