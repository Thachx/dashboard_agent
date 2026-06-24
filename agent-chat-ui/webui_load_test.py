import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer (got {value!r}).")


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a float (got {value!r}).")


def env_paths(name: str, fallback: list[str]) -> list[str]:
    value = os.getenv(name, "")
    if not value.strip():
        return fallback
    paths = [part.strip() for part in value.split("|") if part.strip()]
    return paths or fallback


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def hit_path(session: requests.Session, base_url: str, path: str) -> requests.Response:
    return session.get(f"{base_url}{path}", timeout=REQUEST_TIMEOUT_SECS)


def run_user(user_index: int) -> None:
    session = requests.Session()
    for request_index in range(REQUESTS_PER_USER):
        path = PATHS[(user_index + request_index) % len(PATHS)]
        if RANDOMIZE_PATHS:
            path = random.choice(PATHS)
        start = time.perf_counter()
        try:
            response = hit_path(session, BASE_URL, path)
            elapsed = time.perf_counter() - start
            if response.ok:
                record_success(elapsed)
            else:
                record_failure(
                    f"GET {path} status={response.status_code}",
                    elapsed,
                )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            record_failure(f"GET {path} error={exc}", elapsed)

        if REQUEST_PAUSE_SECS > 0:
            time.sleep(REQUEST_PAUSE_SECS)


def record_success(latency: float) -> None:
    with STATS_LOCK:
        STATS["ok"] += 1
        STATS["latencies"].append(latency)


def record_failure(error: str, latency: float | None = None) -> None:
    with STATS_LOCK:
        STATS["fail"] += 1
        STATS["errors"].append(error)
        if latency is not None:
            STATS["latencies"].append(latency)


BASE_URL = normalize_base_url(os.getenv("CHAT_UI_BASE_URL", "http://localhost:3000"))
REQUESTS_PER_USER = env_int("LOADTEST_REQUESTS_PER_USER", 5)
NUM_USERS = env_int("LOADTEST_USERS", 20)
MAX_WORKERS = env_int("LOADTEST_CONCURRENCY", min(NUM_USERS, 10))
REQUEST_TIMEOUT_SECS = env_float("LOADTEST_TIMEOUT_SECS", 30.0)
REQUEST_PAUSE_SECS = env_float("LOADTEST_PAUSE_SECS", 0.0)
RANDOMIZE_PATHS = os.getenv("LOADTEST_RANDOMIZE_PATHS", "true").lower() in (
    "1",
    "true",
    "yes",
)

DEFAULT_PATHS = [
    "/",
    "/api/health",
]
PATHS = env_paths("LOADTEST_PATHS", DEFAULT_PATHS)

STATS_LOCK = threading.Lock()
STATS = {"ok": 0, "fail": 0, "latencies": [], "errors": []}


def main() -> None:
    total_requests = NUM_USERS * REQUESTS_PER_USER
    print(
        "Starting web UI load test:",
        f"base_url={BASE_URL},",
        f"users={NUM_USERS},",
        f"requests_per_user={REQUESTS_PER_USER},",
        f"total_requests={total_requests},",
        f"concurrency={MAX_WORKERS}",
    )
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_user, idx) for idx in range(NUM_USERS)]
        for future in as_completed(futures):
            future.result()

    elapsed = time.perf_counter() - start
    with STATS_LOCK:
        latencies = list(STATS["latencies"])
        ok = STATS["ok"]
        fail = STATS["fail"]
        errors = list(STATS["errors"])

    rate = ok / elapsed if elapsed > 0 else 0.0
    print(f"Completed in {elapsed:.2f}s. ok={ok} fail={fail} req/s={rate:.2f}")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(
            "Latency (s):",
            f"min={min(latencies):.3f}",
            f"avg={avg_latency:.3f}",
            f"p50={percentile(latencies, 0.50):.3f}",
            f"p95={percentile(latencies, 0.95):.3f}",
            f"p99={percentile(latencies, 0.99):.3f}",
            f"max={max(latencies):.3f}",
        )
    if errors:
        print("Sample errors:")
        for error in errors[:5]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
