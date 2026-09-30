"""Judge transports: standard requests, and the OpenAI Batch API.

Both produce the same `complete()` surface so the graph cannot tell them apart. The batch
adapter exists because the HLD commits to it for the judge, but it is worth being explicit
about the trade-off: `/v1/batches` costs about half as much and returns results on the
provider's schedule, which is *not* bounded at the low end. For a 60-item judging pass the
wait usually dominates the saving, so standard mode is the default for iteration and batch
is there for the larger sweeps.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

API_ROOT = "https://api.openai.com/v1"
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = (1.0, 3.0, 8.0)
BATCH_POLL_SECONDS = 20.0
BATCH_WAIT_SECONDS = 60 * 60 * 6
BATCH_SUFFIX = ".jsonl"


class BatchNotReady(RuntimeError):
    """The batch job is still running; call again later."""


@dataclass
class StandardJudgeClient:
    model: str
    api_key: str
    base_url: str | None = None

    def complete(self, messages: list[dict[str, str]], **options: object) -> str:
        return self._post_messages(messages)

    def _post_messages(self, messages: list[dict[str, str]]) -> str:
        endpoint = f"{(self.base_url or API_ROOT).rstrip('/')}/chat/completions"
        payload = json.dumps({"model": self.model, "messages": messages}).encode("utf-8")
        last: str | None = None
        for attempt in range(MAX_ATTEMPTS):
            request = urllib.request.Request(
                endpoint,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return (body["choices"][0]["message"]["content"]) or ""
            except urllib.error.HTTPError as exc:
                last = f"http_{exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}"
                if exc.code not in RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < len(BACKOFF_SECONDS):
                time.sleep(BACKOFF_SECONDS[attempt])
        raise RuntimeError(f"judge request failed: {last}")


@dataclass
class BatchJudgeClient:
    """Submit a JSONL batch job, then poll and collect.

    Deliberately not a drop-in for `complete()`: batching is a two-phase operation with
    real state on the provider's side, so pretending otherwise would hide the part that
    actually needs designing around.
    """

    model: str
    api_key: str
    base_url: str | None = None
    workdir: Path | None = None

    @property
    def root(self) -> str:
        return f"{(self.base_url or API_ROOT).rstrip('/')}"

    def write_requests(
        self, items: dict[str, list[dict[str, str]]], *, custom_id_prefix: str
    ) -> Path:
        """Write one request per line, tagged with a custom_id we can map back."""
        if self.workdir is None:
            raise ValueError("BatchJudgeClient needs a workdir")
        self.workdir.mkdir(parents=True, exist_ok=True)
        path = self.workdir / f"{custom_id_prefix}{BATCH_SUFFIX}"
        with path.open("w", encoding="utf-8") as handle:
            for key, messages in items.items():
                body = {
                    "custom_id": key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": self.model, "messages": messages},
                }
                handle.write(json.dumps(body) + "\n")
        return path

    def submit(self, path: Path) -> str:
        boundary = f"----harness{uuid.uuid4().hex}"
        payload = _multipart(boundary, path)
        request = urllib.request.Request(
            f"{self.root}/batches",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["id"])

    def status(self, batch_id: str) -> dict:
        request = urllib.request.Request(
            f"{self.root}/batches/{batch_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def collect(self, batch_id: str) -> dict[str, str]:
        """Fetch outputs. Raises BatchNotReady while the job is still running."""
        info = self.status(batch_id)
        state = info.get("status")
        if state != "completed":
            raise BatchNotReady(f"batch {batch_id} is {state}")

        outputs = info.get("output_file_id")
        if not outputs:
            raise BatchNotReady(f"batch {batch_id} completed without an output file")

        request = urllib.request.Request(
            f"{self.root}/files/{outputs}/content",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            text = response.read().decode("utf-8")

        results: dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            body = ((row.get("response") or {}).get("body")) or {}
            choices = body.get("choices") or [{}]
            results[row["custom_id"]] = (choices[0].get("message") or {}).get("content") or ""
        return results

    def run(
        self, items: dict[str, list[dict[str, str]]], *, wait_seconds: int = BATCH_WAIT_SECONDS
    ) -> dict[str, str]:
        """Submit, poll, and collect in one call. Long-running by design."""
        path = self.write_requests(items, custom_id_prefix="judge")
        batch_id = self.submit(path)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            try:
                return self.collect(batch_id)
            except BatchNotReady:
                time.sleep(BATCH_POLL_SECONDS)
        raise BatchNotReady(f"batch {batch_id} did not finish inside {wait_seconds}s")


def _multipart(boundary: str, path: Path) -> bytes:
    """Minimal multipart/form-data body for the batch upload."""
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="input_file"; '
        f'filename="{path.name}"\r\n'
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode()
    return prefix + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
