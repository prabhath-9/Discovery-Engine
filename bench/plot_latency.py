from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAW_LATENCIES_PATH = Path("bench/raw_latencies.csv")
OUTPUT_PATH = Path("bench/latency.png")
LATENCY_BUDGET_MS = 80.0

# Validated categorical slot 1 (blue) and the status "critical" red, matching
# the palette used in ui/app.py.
BAR_COLOR = "#2a78d6"
BUDGET_LINE_COLOR = "#d03b3b"


def read_latencies(path: Path, name: str = "/v1/feed") -> list[float]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [float(row["response_time_ms"]) for row in reader if row["name"] == name and row["success"] == "True"]


def build_histogram(
    latencies_ms: list[float],
    budget_ms: float = LATENCY_BUDGET_MS,
    output_path: Path = OUTPUT_PATH,
    title: str = "/v1/feed latency under load",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies_ms, bins=40, color=BAR_COLOR, edgecolor="white")
    ax.axvline(budget_ms, color=BUDGET_LINE_COLOR, linestyle="--", linewidth=2, label=f"{budget_ms:.0f}ms p99 budget")
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("requests")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    latencies = read_latencies(RAW_LATENCIES_PATH)
    if not latencies:
        raise SystemExit(f"no samples found in {RAW_LATENCIES_PATH}")
    path = build_histogram(latencies)
    print(f"wrote {path} from {len(latencies)} samples")


if __name__ == "__main__":
    main()
