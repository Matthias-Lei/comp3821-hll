"""Summarise the benchmark JSON files in data/results/ into tables for the report.

Run from the repository root after python/run_benchmarks.py:

    python3 python/results_analyser.py

For every stream, sketch and memory budget it combines the hash-seed trials into bias, RMSE,
their 95% confidence intervals and the number of saturated trials, checks the theory where a
formula exists, and writes data/analysis/summary.csv and data/analysis/summary.md.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "data" / "results"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "analysis"
CONFIDENCE = 0.95
MISSING = "—"


@dataclass(frozen=True)
class Trial:
    """One benchmark run: one stream, sketch, memory budget and hash seed."""
    stream: str
    sketch: str
    bits: int
    seed: int
    true_n: int
    bits_used: int
    estimate: float | None  # None when the sketch saturated and gave no estimate
    command: str


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class Summary:
    """All trials of one stream, sketch and memory budget: one row of the results table."""
    stream: str
    sketch: str
    bits: int
    bits_used: int
    true_n: int
    trials: int
    saturated: int
    bias: float | None
    bias_ci: Interval | None
    rmse: float | None
    rmse_ci: Interval | None
    predicted_bias: float | None
    predicted_rmse: float | None
    command: str

    # The theory describes every trial, so it is only checked when no trial saturated:
    # the unsaturated trials alone are a biased sample. None means "not checked".
    @property
    def bias_confirmed(self) -> bool | None:
        if self.saturated or self.bias_ci is None or self.predicted_bias is None:
            return None
        return self.bias_ci.contains(self.predicted_bias)

    @property
    def rmse_confirmed(self) -> bool | None:
        if self.saturated or self.rmse_ci is None or self.predicted_rmse is None:
            return None
        return self.rmse_ci.contains(self.predicted_rmse)

def linear_counting_theory(n: float, m: float) -> tuple[float, float]:
    """Issue #6 / Whang et al. (1990), with t = n / m:
    predicted bias (e^t - t - 1) / 2 and predicted RMSE sqrt(m (e^t - t - 1))."""
    t = n / m
    try:
        excess = math.expm1(t) - t  # expm1 keeps e^t - 1 accurate when t is small
    except OverflowError:
        return math.inf, math.inf
    return excess / 2, math.sqrt(m * excess)


# Predicted (bias, RMSE) for a stream with n distinct values and an m-bit budget, by sketch name.
THEORY: dict[str, Callable[[float, float], tuple[float, float]]] = {
    "LinearCounting": linear_counting_theory,
}

def sketch_name(benchmark_name: str) -> str:
    """BM_LinearCounting/bits:8192/seed:7/iterations:1 -> LinearCounting"""
    return benchmark_name.split("/")[0].removeprefix("BM_")


def load_trials(path: Path) -> list[Trial]:
    """Reads one results file (one stream) and returns one Trial per benchmark run in it."""
    with open(path) as file:
        data = json.load(file)

    trials = []
    for entry in data["benchmarks"]:
        if entry.get("run_type") == "aggregate":
            continue
        try:
            trials.append(Trial(
                stream=path.stem,
                sketch=sketch_name(entry["name"]),
                bits=int(entry["bits"]),
                seed=int(entry["seed"]),
                true_n=int(entry["true_n"]),
                bits_used=int(entry["bits_used"]),
                estimate=None if entry["saturated"] else float(entry["estimate"]),
                command=data.get("command", ""),
            ))
        except KeyError as missing:
            raise ValueError(f"{path.name}: {entry.get('name')} has no {missing} counter; rerun the current benchmark") from None
    return trials


def load_all_trials(results_dir: Path) -> list[Trial]:
    """Reads every .json file in the results directory."""
    trials = []
    for path in sorted(results_dir.glob("*.json")):
        trials.extend(load_trials(path))
    return trials

def group_trials(trials: list[Trial]) -> dict[tuple[str, str, int], list[Trial]]:
    """Groups the trials by (stream, sketch, memory budget). Each group holds that budget's T seeds."""
    groups: dict[tuple[str, str, int], list[Trial]] = defaultdict(list)
    for trial in trials:
        groups[(trial.stream, trial.sketch, trial.bits)].append(trial)
    return dict(groups)


def mean_with_interval(values: list[float]) -> tuple[float, Interval | None]:
    """Mean and its 95% interval, mean +- t * s / sqrt(T), with s the sample standard deviation.

    Proposal: "calculate the mean and sample standard deviation, the standard errors of which
    reduce as 1/sqrt(T)". There is no interval for fewer than two values.
    """
    k = len(values)
    mean = sum(values) / k
    if k < 2:
        return mean, None
    s = math.sqrt(sum((x - mean) ** 2 for x in values) / (k - 1))
    half_width = stats.t.ppf((1 + CONFIDENCE) / 2, k - 1) * s / math.sqrt(k)
    return mean, Interval(mean - half_width, mean + half_width)


def summarise(trials: list[Trial]) -> Summary:
    """Combines one group's trials. Each error is estimate - true_n, over the unsaturated trials.

    Bias is the mean error and RMSE the root of the mean squared error. The RMSE interval is the
    interval of the mean squared error (lower end clamped at 0), square-rooted at both ends.
    """
    first = trials[0]
    if any(trial.true_n != first.true_n for trial in trials):
        raise ValueError(f"{first.stream} {first.sketch} {first.bits}: trials disagree on true_n")

    errors = [trial.estimate - trial.true_n for trial in trials if trial.estimate is not None]
    bias = bias_ci = rmse = rmse_ci = None
    if errors:
        bias, bias_ci = mean_with_interval(errors)
        mse, mse_ci = mean_with_interval([error ** 2 for error in errors])
        rmse = math.sqrt(mse)
        if mse_ci is not None:
            rmse_ci = Interval(math.sqrt(max(mse_ci.low, 0.0)), math.sqrt(mse_ci.high))

    theory = THEORY.get(first.sketch)
    predicted_bias, predicted_rmse = theory(first.true_n, first.bits) if theory else (None, None)

    return Summary(
        stream=first.stream,
        sketch=first.sketch,
        bits=first.bits,
        bits_used=first.bits_used,
        true_n=first.true_n,
        trials=len(trials),
        saturated=len(trials) - len(errors),
        bias=bias,
        bias_ci=bias_ci,
        rmse=rmse,
        rmse_ci=rmse_ci,
        predicted_bias=predicted_bias,
        predicted_rmse=predicted_rmse,
        command=first.command,
    )


def summarise_all(trials: list[Trial]) -> list[Summary]:
    groups = group_trials(trials)
    return [summarise(groups[key]) for key in sorted(groups)]


CSV_FIELDS = [
    "stream", "sketch", "bits", "bits_used", "true_n", "trials", "saturated",
    "bias", "bias_low", "bias_high", "rmse", "rmse_low", "rmse_high",
    "predicted_bias", "predicted_rmse", "bias_confirmed", "rmse_confirmed", "command",
]


def csv_row(summary: Summary) -> dict:
    return {
        "stream": summary.stream,
        "sketch": summary.sketch,
        "bits": summary.bits,
        "bits_used": summary.bits_used,
        "true_n": summary.true_n,
        "trials": summary.trials,
        "saturated": summary.saturated,
        "bias": summary.bias,
        "bias_low": summary.bias_ci.low if summary.bias_ci else None,
        "bias_high": summary.bias_ci.high if summary.bias_ci else None,
        "rmse": summary.rmse,
        "rmse_low": summary.rmse_ci.low if summary.rmse_ci else None,
        "rmse_high": summary.rmse_ci.high if summary.rmse_ci else None,
        "predicted_bias": summary.predicted_bias,
        "predicted_rmse": summary.predicted_rmse,
        "bias_confirmed": summary.bias_confirmed,
        "rmse_confirmed": summary.rmse_confirmed,
        "command": summary.command,
    }


def write_csv(summaries: list[Summary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_row(summary) for summary in summaries)


def number(value: float | None) -> str:
    return MISSING if value is None else f"{value:.2f}"


def interval(value: Interval | None) -> str:
    return MISSING if value is None else f"[{value.low:.2f}, {value.high:.2f}]"


def mark(ok: bool | None) -> str:
    return MISSING if ok is None else ("✓" if ok else "✗")


MARKDOWN_HEADER = (
    "| Sketch | Budget (bits) | Bits used | n | Bias | Bias 95% CI | RMSE | RMSE 95% CI "
    "| Saturated | Predicted bias | Predicted RMSE | Bias in CI | RMSE in CI | Command |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def markdown_row(summary: Summary) -> str:
    cells = [
        summary.sketch,
        str(summary.bits),
        str(summary.bits_used),
        str(summary.true_n),
        number(summary.bias),
        interval(summary.bias_ci),
        number(summary.rmse),
        interval(summary.rmse_ci),
        f"{summary.saturated} of {summary.trials}",
        number(summary.predicted_bias),
        number(summary.predicted_rmse),
        mark(summary.bias_confirmed),
        mark(summary.rmse_confirmed),
        f"`{summary.command}`" if summary.command else MISSING,
    ]
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def write_markdown(summaries: list[Summary], path: Path) -> None:
    """One table per stream, one row per sketch and memory budget."""
    by_stream: dict[str, list[Summary]] = defaultdict(list)
    for summary in summaries:
        by_stream[summary.stream].append(summary)

    sections = []
    for stream, rows in by_stream.items():
        sections.append("\n".join([f"### {stream}", "", MARKDOWN_HEADER, *(markdown_row(row) for row in rows)]))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR, help="directory of benchmark JSON files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for summary.csv and summary.md")
    args = parser.parse_args()

    trials = load_all_trials(args.results)
    if not trials:
        print(f"no results in {args.results} (run python/run_benchmarks.py first)", file=sys.stderr)
        return 1

    summaries = summarise_all(trials)
    write_csv(summaries, args.output / "summary.csv")
    write_markdown(summaries, args.output / "summary.md")
    print(f"summarised {len(trials)} trials into {len(summaries)} rows in {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())