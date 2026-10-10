"""Saving finished work off the GPU VM as soon as it is done, so a preempted spot VM loses only the work in progress.

A store is a folder: a Hugging Face bucket prefix (``hf://buckets/NAMESPACE/BUCKET/PREFIX``) or a local directory
(tests, or a disk mounted on the VM).

- **Units:** work is saved in units. Each unit's data files are saved first and its marker file after, in a separate
  call, because bucket batches are not transactional.
- **What a marker means:** a marker in the store means its unit is complete. A resumed job, on any VM in any region,
  skips every unit whose marker is there.
- **Background saving:** ``Uploader`` saves units from a background thread, so the GPU never waits on the network.
- **Waiting:** ``flush`` waits for the uploads, e.g. before the job exits, or in the 60 s Nebius gives before it
  stops a spot VM.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass

HF_PREFIX = "hf://buckets/"


class Store:
    url: str

    def files(self) -> dict[str, int]:
        """Every saved file: path relative to the store -> size in bytes."""
        raise NotImplementedError

    def put(self, files: list[tuple[str, str]]) -> None:
        """Save ``(local path, path in the store)`` pairs; raises if any failed."""
        raise NotImplementedError

    def get(self, rel: str, local: str) -> None:
        raise NotImplementedError

    def delete(self, rels: list[str]) -> None:
        raise NotImplementedError


class LocalStore(Store):
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.url = self.root

    def files(self) -> dict[str, int]:
        out = {}
        for dirpath, _, names in os.walk(self.root):
            for name in names:
                if not name.endswith(".part"):
                    path = os.path.join(dirpath, name)
                    out[os.path.relpath(path, self.root).replace(os.sep, "/")] = os.path.getsize(path)
        return out

    def put(self, files: list[tuple[str, str]]) -> None:
        for local, rel in files:
            dest = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(local, dest + ".part")
            os.replace(dest + ".part", dest)  # a reader never sees half a file

    def get(self, rel: str, local: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        shutil.copyfile(os.path.join(self.root, rel), local)

    def delete(self, rels: list[str]) -> None:
        for rel in rels:
            os.remove(os.path.join(self.root, rel))


class HfBucketStore(Store):
    """A prefix in a Hugging Face Storage Bucket. The token comes from huggingface_hub's usual places (the
    launcher puts it in ``~/.cache/huggingface/token`` on the VM for the job's duration)."""

    def __init__(self, bucket_id: str, prefix: str = "", api=None) -> None:
        if api is None:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()  # one bar set per upload would flood the job's log
            api = HfApi()
        self.bucket_id, self.prefix, self.api = bucket_id, prefix.strip("/"), api
        self.url = f"{HF_PREFIX}{bucket_id}" + (f"/{self.prefix}" if self.prefix else "")

    def _remote(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def files(self) -> dict[str, int]:
        lead = f"{self.prefix}/" if self.prefix else ""
        return {f.path[len(lead):]: f.size
                for f in self.api.list_bucket_tree(self.bucket_id, prefix=lead or None, recursive=True)
                if getattr(f, "size", None) is not None and f.path.startswith(lead)}

    def put(self, files: list[tuple[str, str]]) -> None:
        self.api.batch_bucket_files(self.bucket_id, add=[(local, self._remote(rel)) for local, rel in files])

    def get(self, rel: str, local: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        self.api.download_bucket_files(self.bucket_id, [(self._remote(rel), local)], raise_on_missing_files=True)

    def delete(self, rels: list[str]) -> None:
        if rels:
            self.api.batch_bucket_files(self.bucket_id, delete=[self._remote(rel) for rel in rels])


def permanent(error: Exception) -> bool:
    """An upload error that retrying won't fix: HTTP 4xx except 408/429 (quota, permission, bad request), or a full
    or read-only disk."""
    import errno

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return True
    return isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EROFS, errno.EACCES)


def open_store(url: str) -> Store:
    """``hf://buckets/NAMESPACE/BUCKET[/PREFIX]`` or a local directory."""
    if url.startswith(HF_PREFIX):
        namespace, _, rest = url[len(HF_PREFIX):].partition("/")
        bucket, _, prefix = rest.partition("/")
        if not namespace or not bucket:
            raise ValueError(f"store {url!r}: want hf://buckets/NAMESPACE/BUCKET[/PREFIX]")
        return HfBucketStore(f"{namespace}/{bucket}", prefix)
    if "://" in url:
        raise ValueError(f"store {url!r}: want hf://buckets/NAMESPACE/BUCKET[/PREFIX] or a local directory")
    return LocalStore(url)


@dataclass
class _Unit:
    name: str
    data: list[tuple[str, str]]
    marker: tuple[str, str] | None
    nbytes: int


class Uploader:
    """Saves units to a store from a background thread, in the order they were submitted.

    - A unit's marker is saved only after all its data files were, never if they failed (after ``retries``
      attempts), so the store never claims an unfinished unit.
    - Failed units are listed in ``failed``; their files stay on the VM for the launcher's final upload.
    - ``rate()`` is the measured upload speed, used to estimate how long the backlog takes. Only uploads of at least
      ``RATE_MIN_BYTES`` count, because a small file measures latency, not bandwidth: a 1 KB index once made a 0.9 GB
      shard look like 12 days of uploading. It stays None until ``RATE_MEASURED_BYTES`` have been measured.
    - **Permanent errors** (a full bucket, missing permission: HTTP 4xx other than 408/429, or a full disk) are not
      retried. They set ``broken``, so the caller can stop producing work that can't be saved.
    """

    RATE_MIN_BYTES = 16 * 2**20
    RATE_MEASURED_BYTES = 100 * 2**20

    def __init__(self, store: Store, *, retries: int = 6, backoff: float = 5.0, log=print) -> None:
        self.store, self.retries, self.backoff, self.log = store, retries, backoff, log
        self.failed: list[str] = []
        self.saved: list[str] = []
        self.broken: str | None = None
        self._units: deque[_Unit] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self._bytes = 0
        self._seconds = 0.0
        self._thread = threading.Thread(target=self._run, name="uploader", daemon=True)
        self._thread.start()

    def submit(self, name: str, data: list[tuple[str, str]], marker: tuple[str, str] | None = None) -> None:
        files = [*data, *([marker] if marker else [])]
        unit = _Unit(name, list(data), marker, sum(os.path.getsize(local) for local, _ in files))
        with self._cv:
            if self._closed:
                raise RuntimeError("the uploader is closed")
            self._units.append(unit)
            self._cv.notify_all()

    def pending_bytes(self) -> int:
        with self._cv:
            return sum(u.nbytes for u in self._units)

    def rate(self) -> float | None:
        """Bytes per second over the large uploads so far, or None until enough has been measured."""
        with self._cv:
            if self._bytes < self.RATE_MEASURED_BYTES or self._seconds <= 0:
                return None
            return self._bytes / self._seconds

    def _put(self, files: list[tuple[str, str]], what: str) -> bool:
        if self.broken:
            return False
        for attempt in range(1, self.retries + 1):
            t = time.time()
            try:
                self.store.put(files)
            except Exception as e:  # noqa: BLE001 — network, auth, quota: retry, then give up on this unit
                self.log(f"upload of {what} failed ({attempt}/{self.retries}): {type(e).__name__}: {e}")
                if permanent(e):
                    self.broken = f"{type(e).__name__}: {e}"
                    self.log(f"the store refuses uploads ({self.broken}); not retrying")
                    return False
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
                continue
            size = sum(os.path.getsize(local) for local, _ in files)
            if size >= self.RATE_MIN_BYTES:  # small files measure latency, not bandwidth
                with self._cv:
                    self._bytes += size
                    self._seconds += time.time() - t
            return True
        return False

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._units and not self._closed:
                    self._cv.wait()
                if not self._units:
                    return
                unit = self._units[0]  # stays queued (and counted as pending) until it is done
            try:
                ok = self._put(unit.data, f"{unit.name} (data)") if unit.data else True
                if ok and unit.marker:
                    ok = self._put([unit.marker], f"{unit.name} (marker)")
            except Exception as e:  # noqa: BLE001 — never let the thread die with units queued (flush would hang)
                self.log(f"upload of {unit.name} crashed: {type(e).__name__}: {e}")
                ok = False
            with self._cv:
                self._units.popleft()
                (self.saved if ok else self.failed).append(unit.name)
                rate = self._bytes / self._seconds if self._seconds > 0 else float("nan")
                waiting = sum(u.nbytes for u in self._units)
                self._cv.notify_all()
            if ok:
                self.log(f"saved {unit.name} to {self.store.url} ({unit.nbytes / 1e6:.0f} MB; "
                         f"{rate / 1e6:.0f} MB/s so far; {waiting / 1e9:.1f} GB waiting)")

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until every submitted unit is saved or has failed. False if ``timeout`` ran out first."""
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            while self._units:
                if not self._thread.is_alive():
                    return False
                left = None if deadline is None else deadline - time.time()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(5.0 if left is None else min(left, 5.0))
        return True

    def close(self, timeout: float | None = None) -> bool:
        """``flush``, then stop the thread. False if units were still unsaved when ``timeout`` ran out."""
        done = self.flush(timeout)
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        return done
