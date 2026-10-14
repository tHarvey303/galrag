"""Move results between machine-local scratch (``work_dir``) and the shared archive (``data_dir``).

Storage policy: ``/nexus`` (Nexus) is shared network storage, and heavy I/O there
(thousands of small files, a hot database written by several loops) slows the whole
platform and causes SQLite lock stalls.  Jobs therefore do their hot I/O on the
machine's local disk -- the live SQLite and the raw HTML/PDF cache live in
``GALRAG_WORK_DIR``, and on the GPU machine the LanceDB store lives in
``GALRAG_LANCEDB_DIR`` -- and only move finished results back to Nexus in bulk:
:func:`push` writes one consistent SQLite snapshot (``sqlite3`` backup API into a temp
file, then an atomic rename over ``data_dir/galrag.sqlite``) and rsyncs a local LanceDB
directory into ``data_dir/lancedb``.  :func:`pull` is the reverse and creates a local
working copy from the archive.  The MCP server only ever reads the archive.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import tarfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from galrag.config import Settings

log = logging.getLogger(__name__)


def _same(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def _size(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def backup_sqlite(src: Path, dst: Path) -> int:
    """Consistent copy of the SQLite database ``src`` to ``dst`` via the backup API.

    Writes ``dst + ".tmp"`` sequentially, then atomically renames it onto ``dst``; the
    live file and its -wal/-shm are never copied directly.  Returns the size in bytes."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    s = sqlite3.connect(src, timeout=600)   # read-only use; the backup holds a read snapshot
    try:
        d = sqlite3.connect(tmp)
        try:
            s.backup(d, pages=-1)
            d.execute("PRAGMA journal_mode=DELETE")   # self-contained file, no -wal next to it
        finally:
            d.close()
    finally:
        s.close()
    os.replace(tmp, dst)
    for suffix in ("-wal", "-shm"):                # stale sidecars of a previous file would corrupt it
        Path(str(dst) + suffix).unlink(missing_ok=True)
    return dst.stat().st_size


def copy_tree(src: Path, dst: Path) -> str:
    """Mirror directory ``src`` into ``dst`` (``rsync -a --delete``; copytree fallback)."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        subprocess.run([rsync, "-a", "--delete", f"{src}/", f"{dst}/"], check=True)
        return "rsync"
    log.warning("rsync not found: falling back to shutil.copytree (no --delete; stale files stay)")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return "copytree (rsync missing; no deletions)"


def _which(only: str | None) -> tuple[bool, bool]:
    if only in (None, ""):
        return True, True
    if only not in ("sqlite", "lancedb"):
        raise ValueError("only must be 'sqlite' or 'lancedb'")
    return only == "sqlite", only == "lancedb"


def push(settings: Settings, *, sqlite: bool = True, lancedb: bool = True, raw: bool = True) -> dict[str, Any]:
    """Copy local results to the archive in ``data_dir``.  Returns what was synced."""
    out: dict[str, Any] = {}
    if sqlite:
        src, dst = settings.sqlite_path, settings.archive_sqlite_path
        if _same(src, dst):
            out["sqlite"] = "skipped (no work_dir: the live SQLite is the archive)"
        else:
            t0 = time.time()
            n = backup_sqlite(src, dst)
            out["sqlite"] = {"from": str(src), "to": str(dst), "bytes": n, "seconds": round(time.time() - t0, 1)}
    if lancedb:
        src, dst = settings.lancedb_dir, settings.archive_lancedb_dir
        if _same(src, dst):
            out["lancedb"] = "skipped (LanceDB already lives in data_dir)"
        elif not Path(src).exists():
            out["lancedb"] = f"skipped ({src} does not exist)"
        else:
            t0 = time.time()
            how = copy_tree(src, dst)
            out["lancedb"] = {"from": str(src), "to": str(dst), "method": how, "bytes": _size(dst),
                              "seconds": round(time.time() - t0, 1)}
    if raw and settings.work_dir:
        try:
            out.update(archive_raw(settings))
        except Exception:
            log.exception("raw archive failed")
            out["raw"] = "failed (see log)"
    log.info("sync push: %s", out)
    return out


def pull(settings: Settings, *, sqlite: bool = True, lancedb: bool = True, force: bool = False) -> dict[str, Any]:
    """Create/refresh the local working copies from the archive in ``data_dir``.

    An existing local SQLite is only overwritten with ``force`` (it may hold work that
    has not been pushed yet)."""
    out: dict[str, Any] = {}
    if sqlite:
        src, dst = settings.archive_sqlite_path, settings.sqlite_path
        if _same(src, dst):
            out["sqlite"] = "skipped (no work_dir: the live SQLite is the archive)"
        elif dst.exists() and not force:
            raise FileExistsError(f"{dst} exists; use force=True (--force) to overwrite it with the archive")
        elif not src.exists():
            out["sqlite"] = f"skipped ({src} does not exist)"
        else:
            t0 = time.time()
            n = backup_sqlite(src, dst)
            out["sqlite"] = {"from": str(src), "to": str(dst), "bytes": n, "seconds": round(time.time() - t0, 1)}
    if lancedb:
        src, dst = settings.archive_lancedb_dir, settings.lancedb_dir
        if _same(src, dst):
            out["lancedb"] = "skipped (LanceDB already lives in data_dir)"
        elif not Path(src).exists():
            out["lancedb"] = f"skipped ({src} does not exist)"
        else:
            t0 = time.time()
            how = copy_tree(src, dst)
            out["lancedb"] = {"from": str(src), "to": str(dst), "method": how, "bytes": _size(dst),
                              "seconds": round(time.time() - t0, 1)}
    log.info("sync pull: %s", out)
    return out


def archive_raw(settings: Settings, *, source: Path | None = None, min_mb: float | None = None,
                min_age_s: float = 600.0, delete: bool = True) -> dict[str, Any]:
    """Move the raw HTML/PDF cache to Nexus **in bulk**: one tarball per call under
    ``settings.raw_archive_dir`` (``raw-<UTC stamp>.tar``), then delete the local files.

    Only files older than ``min_age_s`` are taken (a fetch may still be writing), and
    nothing happens below ``min_mb`` MB (default ``settings.raw_archive_mb``) unless
    ``min_mb=0``.  The raw cache is only needed to re-parse without re-fetching, so the
    tarballs are a cold archive; ``tar -xf`` them into ``raw_dir`` to restore.
    """
    src = Path(source or settings.raw_dir)
    thresh = settings.raw_archive_mb if min_mb is None else min_mb
    if not src.exists():
        return {"raw": f"skipped ({src} does not exist)"}
    now = time.time()
    files = [f for f in src.rglob("*") if f.is_file() and now - f.stat().st_mtime >= min_age_s]
    total = sum(f.stat().st_size for f in files)
    if not files or total < thresh * 1e6:
        return {"raw": f"skipped ({len(files)} files, {total / 1e6:.0f} MB < {thresh} MB)"}
    dest_dir = settings.raw_archive_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tmp = dest_dir / f"raw-{stamp}.tar.tmp"
    final = dest_dir / f"raw-{stamp}.tar"
    t0 = time.time()
    with tarfile.open(tmp, "w") as tar:  # files are already gzipped; no recompression
        for f in files:
            tar.add(f, arcname=str(f.relative_to(src)))
    with tarfile.open(tmp) as tar:  # verify before deleting anything
        names = set(tar.getnames())
    missing = [f for f in files if str(f.relative_to(src)) not in names]
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"raw archive verification failed: {len(missing)} files missing")
    os.replace(tmp, final)
    if delete:
        for f in files:
            f.unlink(missing_ok=True)
        for d in sorted((d for d in src.rglob("*") if d.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    out = {"raw": {"files": len(files), "bytes": total, "tar": str(final), "deleted_local": delete,
                   "seconds": round(time.time() - t0, 1)}}
    log.info("raw archive: %s", out)
    return out


def sync_push_every(settings: Settings, interval_s: float, *, sqlite: bool = True,
                    lancedb: bool = True) -> Callable[..., dict[str, Any] | None]:
    """Callable for the loops' ``on_progress`` hook: pushes when at least ``interval_s``
    seconds passed since the last push (call it with ``force=True`` for a final push).
    Push errors are logged, never raised."""
    last = time.time()

    def maybe_push(force: bool = False) -> dict[str, Any] | None:
        nonlocal last
        if not force and time.time() - last < interval_s:
            return None
        last = time.time()
        try:
            return push(settings, sqlite=sqlite, lancedb=lancedb)
        except Exception:
            log.exception("sync push failed")
            return None

    return maybe_push
