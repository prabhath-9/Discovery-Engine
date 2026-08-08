from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STATS_CSV = Path("bench/results_stats.csv")
DEFAULT_ENDPOINT = "/v1/feed"

TARGET_RPS = 10_000
WORKERS_PER_POD = 16
API_VCPU_PER_POD = 4
RETRIEVAL_VCPU_PER_POD = 4
COST_PER_VCPU_HR = 0.04
HEADROOM = 0.40
REDIS_PG_COST_HR = 1.20
QUERIES_PER_REQUEST = 4
ANN_LATENCY_MS = 2.0
GPU_HOURS_PER_DAY = 2.0
GPU_COST_PER_HR = 1.0
LLM_COST_PER_CALL = 0.0005
SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600


@dataclass(frozen=True, slots=True)
class CapacityResult:
    mean_latency_ms: float
    concurrent_in_flight: float
    throughput_per_pod_rps: float
    api_pods: int
    retrieval_pods: int
    cost_per_hour: float
    cost_per_1000: float
    cost_per_million: float
    llm_cost_per_million: float
    llm_multiplier: float


def read_mean_latency_ms(stats_csv: Path, endpoint_name: str = DEFAULT_ENDPOINT) -> float:
    with stats_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("Name") == endpoint_name:
            return float(row["Average Response Time"])
    for row in rows:
        if row.get("Name") in ("Aggregated", "Total"):
            return float(row["Average Response Time"])
    raise ValueError(f"no '{endpoint_name}' or aggregated row found in {stats_csv}")


def api_pods_needed(
    mean_latency_ms: float,
    target_rps: int = TARGET_RPS,
    workers_per_pod: int = WORKERS_PER_POD,
    headroom: float = HEADROOM,
) -> tuple[float, int]:
    mean_latency_s = mean_latency_ms / 1000.0
    throughput_per_pod = workers_per_pod / mean_latency_s
    raw_pods = target_rps / throughput_per_pod
    return throughput_per_pod, math.ceil(raw_pods * (1 + headroom))


def retrieval_pods_needed(
    target_rps: int = TARGET_RPS,
    queries_per_request: int = QUERIES_PER_REQUEST,
    ann_latency_ms: float = ANN_LATENCY_MS,
    vcpu_per_pod: int = RETRIEVAL_VCPU_PER_POD,
) -> int:
    cores_needed = target_rps * queries_per_request * (ann_latency_ms / 1000.0)
    return math.ceil(cores_needed / vcpu_per_pod)


def hourly_cost(
    api_pods: int,
    retrieval_pods: int,
    api_vcpu_per_pod: int = API_VCPU_PER_POD,
    retrieval_vcpu_per_pod: int = RETRIEVAL_VCPU_PER_POD,
    cost_per_vcpu_hr: float = COST_PER_VCPU_HR,
    redis_pg_cost_hr: float = REDIS_PG_COST_HR,
) -> float:
    return (
        api_pods * api_vcpu_per_pod * cost_per_vcpu_hr
        + retrieval_pods * retrieval_vcpu_per_pod * cost_per_vcpu_hr
        + redis_pg_cost_hr
    )


def online_cost_per_1000(cost_per_hour: float, target_rps: int = TARGET_RPS) -> float:
    requests_per_hour = target_rps * SECONDS_PER_HOUR
    return cost_per_hour / requests_per_hour * 1000


def offline_cost_per_1000(
    target_rps: int = TARGET_RPS,
    gpu_hours_per_day: float = GPU_HOURS_PER_DAY,
    gpu_cost_per_hr: float = GPU_COST_PER_HR,
) -> float:
    daily_requests = target_rps * SECONDS_PER_DAY
    daily_cost = gpu_hours_per_day * gpu_cost_per_hr
    return daily_cost / daily_requests * 1000


def llm_comparison(cost_per_1000: float, llm_cost_per_call: float = LLM_COST_PER_CALL) -> tuple[float, float]:
    llm_cost_per_1000 = llm_cost_per_call * 1000
    multiplier = llm_cost_per_1000 / cost_per_1000 if cost_per_1000 else float("inf")
    return llm_cost_per_1000, multiplier


def compute(
    mean_latency_ms: float,
    target_rps: int = TARGET_RPS,
    workers_per_pod: int = WORKERS_PER_POD,
    headroom: float = HEADROOM,
) -> CapacityResult:
    throughput_per_pod, api_pods = api_pods_needed(mean_latency_ms, target_rps, workers_per_pod, headroom)
    retrieval_pods = retrieval_pods_needed(target_rps)
    cost_hr = hourly_cost(api_pods, retrieval_pods)
    cost_1000 = online_cost_per_1000(cost_hr, target_rps) + offline_cost_per_1000(target_rps)
    llm_cost_1000, multiplier = llm_comparison(cost_1000)

    return CapacityResult(
        mean_latency_ms=mean_latency_ms,
        concurrent_in_flight=target_rps * mean_latency_ms / 1000.0,
        throughput_per_pod_rps=throughput_per_pod,
        api_pods=api_pods,
        retrieval_pods=retrieval_pods,
        cost_per_hour=cost_hr,
        cost_per_1000=cost_1000,
        cost_per_million=cost_1000 * 1000,
        llm_cost_per_million=llm_cost_1000 * 1000,
        llm_multiplier=multiplier,
    )


def format_report(result: CapacityResult, target_rps: int = TARGET_RPS) -> str:
    return (
        f"Capacity math for {target_rps:,} req/s (mean latency: {result.mean_latency_ms:.1f}ms measured)\n"
        f"  Concurrent requests in flight:      {result.concurrent_in_flight:.0f}\n"
        f"  Throughput per API pod:             {result.throughput_per_pod_rps:.0f} req/s\n"
        f"  API pods needed (+{int(HEADROOM * 100)}% headroom):    {result.api_pods}\n"
        f"  Retrieval pods needed:               {result.retrieval_pods}\n"
        "\n"
        f"Cost per hour:                        ${result.cost_per_hour:.2f}\n"
        f"Cost per 1,000 recommendations:       ${result.cost_per_1000:.6f}\n"
        f"Cost per 1,000,000 recommendations:   ${result.cost_per_million:.2f}\n"
        "\n"
        f"One-LLM-call-per-request would cost:  ${result.llm_cost_per_million:.2f} per million "
        f"— {result.llm_multiplier:,.0f}x more expensive\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capacity + cost model from architecture.md §10")
    parser.add_argument("--mean-latency-ms", type=float, default=None, help="Override measured mean latency")
    parser.add_argument("--stats-csv", type=Path, default=DEFAULT_STATS_CSV, help="Locust --csv stats file")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT)
    parser.add_argument("--target-rps", type=int, default=TARGET_RPS)
    args = parser.parse_args()

    mean_latency_ms = args.mean_latency_ms
    if mean_latency_ms is None:
        mean_latency_ms = read_mean_latency_ms(args.stats_csv, args.endpoint)

    result = compute(mean_latency_ms, target_rps=args.target_rps)
    print(format_report(result, target_rps=args.target_rps))


if __name__ == "__main__":
    main()
