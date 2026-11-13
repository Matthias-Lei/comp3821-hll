"""Summarise the benchmark JSON files in data/results/ into tables for the report.

Run from the repository root after python/run_benchmarks.py:

    python3 python/results_analyser.py

For every stream, sketch and memory budget it combines the hash-seed trials into what the
proposal asks for: bias, sample standard deviation and RMSE, their 95% confidence intervals,
relative errors, the storage actually used, the number of saturated trials and the time per
pass. It compares each sketch with its theoretical prediction within the range of n where the
formula applies, and reports how well merged sketches of two disjoint parts of a stream do.
It writes data/analysis/summary.csv and data/analysis/summary.md.
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
MERGE_SUFFIX = "Merge"  # bench.cpp names merge benchmarks BM_<Sketch>Merge
TIME_UNITS_IN_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}


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
    time_ms: float | None = None  # one pass over the stream
    # Merge benchmarks only: distinct values in each part, and whether the merged estimate
    # equals that of a sketch built directly from the whole stream.
    merge_left_n: int | None = None
    merge_right_n: int | None = None
    merged_equals_direct: bool | None = None


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
    sd: float | None
    rmse: float | None
    rmse_ci: Interval | None
    predicted_bias: float | None
    predicted_rmse: float | None
    theory_applies: bool
    mean_time_ms: float | None
    command: str
    merge_left_n: int | None = None
    merge_right_n: int | None = None
    merged_equals_direct: int | None = None  # number of trials

    @property
    def relative_bias(self) -> float | None:
        return None if self.bias is None or self.true_n == 0 else self.bias / self.true_n

    @property
    def relative_rmse(self) -> float | None:
        return None if self.rmse is None or self.true_n == 0 else self.rmse / self.true_n

    # A prediction is only checked where its formula applies, and when no trial saturated: the
    # theory describes every trial, so the unsaturated ones alone are a biased sample.
    # None means "not checked".
    def _checkable(self) -> bool:
        return self.theory_applies and not self.saturated

    @property
    def bias_confirmed(self) -> bool | None:
        if not self._checkable() or self.bias_ci is None or self.predicted_bias is None:
            return None
        return self.bias_ci.contains(self.predicted_bias)

    @property
    def rmse_confirmed(self) -> bool | None:
        if not self._checkable() or self.rmse_ci is None or self.predicted_rmse is None:
            return None
        return self.rmse_ci.contains(self.predicted_rmse)


# --- Theory ---------------------------------------------------------------------------------
# Each formula takes n distinct values and an m-bit budget and returns (predicted bias,
# predicted RMSE). The constants mirror the C++ sketches, so keep them in step with the code.

BLOOM_HASHES = 3  # k_ in cpp/src/bloom_filter.cpp
HLL_REGISTER_BITS = 6  # HyperLogLog::kRegisterBits
HLL_SMALL_RANGE = 2.5  # HyperLogLog switches to Linear Counting when the estimate is <= 2.5 r
KMV_BITS_PER_VALUE = 64  # proposal: KMV stores up to B / 64 hash values


def linear_counting_theory(n: float, m: float) -> tuple[float, float]:
    """Issue #6 / Whang et al. (1990), with t = n / m:
    predicted bias (e^t - t - 1) / 2 and predicted RMSE sqrt(m (e^t - t - 1))."""
    t = n / m
    try:
        excess = math.expm1(t) - t  # expm1 keeps e^t - 1 accurate when t is small
    except OverflowError:
        return math.inf, math.inf
    return excess / 2, math.sqrt(m * excess)


def bloom_filter_theory(n: float, m: float) -> tuple[float, float]:
    """Linear Counting generalised to k bits per value (the estimator -(m/k) ln V): k times the
    estimate is a Linear Counting estimate of the k n bits set, so both the bias and the RMSE
    are those of Linear Counting with k n values, divided by k. Treats each value's k bits as
    independent."""
    bias, rmse = linear_counting_theory(BLOOM_HASHES * n, m)
    return bias / BLOOM_HASHES, rmse / BLOOM_HASHES


def hyperloglog_registers(m: float) -> int:
    """The register count the C++ sketch picks: the largest power of two that fits m bits."""
    registers = 1
    while registers * 2 <= m // HLL_REGISTER_BITS:
        registers *= 2
    return registers


def hyperloglog_theory(n: float, m: float) -> tuple[float, float]:
    """Flajolet et al. (2007), the published error formula the proposal adopts for HLL. With r
    registers, small cardinalities use Linear Counting on the registers; above that the
    estimate is unbiased with relative standard error 1.04 / sqrt(r)."""
    registers = hyperloglog_registers(m)
    if n <= HLL_SMALL_RANGE * registers:
        return linear_counting_theory(n, registers)
    return 0.0, 1.04 * n / math.sqrt(registers)


def hyperloglog_applies(n: float, m: float) -> bool:
    """The sketch switches on its raw estimate, not on n, so near 2.5 r some seeds use one
    estimator and some the other; and raw HyperLogLog stays biased up to about 5 r (Heule et al.
    2013). Neither formula holds from 2 r (20% below the switch, several standard errors of the
    raw estimate at r >= 1024) to 5 r."""
    registers = hyperloglog_registers(m)
    return n <= 0.8 * HLL_SMALL_RANGE * registers or n >= 5 * registers


def kmv_theory(n: float, m: float) -> tuple[float, float]:
    """Bar-Yossef et al. (2002), in the unbiased form (k - 1) / U_(k) with k = B / 64: exact
    for n < k, otherwise unbiased with variance n (n - k + 1) / (k - 2) (Beyer et al. 2007)."""
    k = int(m) // KMV_BITS_PER_VALUE
    if n < k:
        return 0.0, 0.0
    if k < 3:
        return 0.0, math.inf
    return 0.0, math.sqrt(n * (n - k + 1) / (k - 2))


# Predicted (bias, RMSE) by sketch name: the benchmark function's name without "BM_".
# KMinVals is the name on Tin's KMV branch, so it is ready when that is merged.
THEORY: dict[str, Callable[[float, float], tuple[float, float]]] = {
    "LinearCounting": linear_counting_theory,
    "BloomFilter": bloom_filter_theory,
    "HyperLogLog": hyperloglog_theory,
    "KMinVals": kmv_theory,
}

# Where a formula only holds for some n: proposal, "within the range of n where that
# estimator's formula applies". Sketches not listed apply everywhere they have not saturated.
THEORY_APPLIES: dict[str, Callable[[float, float], bool]] = {
    "HyperLogLog": hyperloglog_applies,
}


def base_sketch(sketch: str) -> str:
    """LinearCountingMerge -> LinearCounting. A merged sketch of the whole stream is built from
    the same values as a direct one, so it has the same predicted error."""
    return sketch.removesuffix(MERGE_SUFFIX)


def predict(sketch: str, n: float, m: float) -> tuple[float | None, float | None, bool]:
    """Predicted (bias, RMSE) and whether the formula applies at this n."""
    name = base_sketch(sketch)
    theory = THEORY.get(name)
    if theory is None:
        return None, None, False
    applies = THEORY_APPLIES.get(name, lambda n, m: True)(n, m)
    bias, rmse = theory(n, m)
    return bias, rmse, applies


# --- Loading --------------------------------------------------------------------------------

def sketch_name(benchmark_name: str) -> str:
    """BM_LinearCounting/bits:8192/seed:7/iterations:1 -> LinearCounting"""
    return benchmark_name.split("/")[0].removeprefix("BM_")


def time_in_ms(entry: dict) -> float | None:
    if "real_time" not in entry:
        return None
    return float(entry["real_time"]) * TIME_UNITS_IN_MS[entry.get("time_unit", "ns")]


def optional_int(entry: dict, key: str) -> int | None:
    return int(entry[key]) if key in entry else None


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
                time_ms=time_in_ms(entry),
                merge_left_n=optional_int(entry, "merge_left_n"),
                merge_right_n=optional_int(entry, "merge_right_n"),
                merged_equals_direct=bool(entry["merged_equals_direct"]) if "merged_equals_direct" in entry else None,
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


# --- Statistics -----------------------------------------------------------------------------

def group_trials(trials: list[Trial]) -> dict[tuple[str, str, int], list[Trial]]:
    """Groups the trials by (stream, sketch, memory budget). Each group holds that budget's T seeds."""
    groups: dict[tuple[str, str, int], list[Trial]] = defaultdict(list)
    for trial in trials:
        groups[(trial.stream, trial.sketch, trial.bits)].append(trial)
    return dict(groups)


def sample_sd(values: list[float]) -> float | None:
    """Sample standard deviation (dividing by T - 1); none for fewer than two values."""
    k = len(values)
    if k < 2:
        return None
    mean = sum(values) / k
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (k - 1))


def mean_with_interval(values: list[float]) -> tuple[float, Interval | None]:
    """Mean and its 95% interval, mean +- t * s / sqrt(T), with s the sample standard deviation.

    Proposal: "calculate the mean and sample standard deviation, the standard errors of which
    reduce as 1/sqrt(T)". There is no interval for fewer than two values.
    """
    k = len(values)
    mean = sum(values) / k
    s = sample_sd(values)
    if s is None:
        return mean, None
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

    predicted_bias, predicted_rmse, applies = predict(first.sketch, first.true_n, first.bits)
    times = [trial.time_ms for trial in trials if trial.time_ms is not None]
    merged = [trial.merged_equals_direct for trial in trials if trial.merged_equals_direct is not None]

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
        sd=sample_sd(errors),
        rmse=rmse,
        rmse_ci=rmse_ci,
        predicted_bias=predicted_bias,
        predicted_rmse=predicted_rmse,
        theory_applies=applies,
        mean_time_ms=sum(times) / len(times) if times else None,
        command=first.command,
        merge_left_n=first.merge_left_n,
        merge_right_n=first.merge_right_n,
        merged_equals_direct=sum(merged) if merged else None,
    )


def summarise_all(trials: list[Trial]) -> list[Summary]:
    groups = group_trials(trials)
    return [summarise(groups[key]) for key in sorted(groups)]


# --- Output ---------------------------------------------------------------------------------

CSV_FIELDS = [
    "stream", "sketch", "bits", "bits_used", "true_n", "trials", "saturated",
    "bias", "bias_low", "bias_high", "sd", "rmse", "rmse_low", "rmse_high",
    "relative_bias", "relative_rmse",
    "predicted_bias", "predicted_rmse", "theory_applies", "bias_confirmed", "rmse_confirmed",
    "mean_time_ms", "merge_left_n", "merge_right_n", "merged_equals_direct", "command",
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
        "sd": summary.sd,
        "rmse": summary.rmse,
        "rmse_low": summary.rmse_ci.low if summary.rmse_ci else None,
        "rmse_high": summary.rmse_ci.high if summary.rmse_ci else None,
        "relative_bias": summary.relative_bias,
        "relative_rmse": summary.relative_rmse,
        "predicted_bias": summary.predicted_bias,
        "predicted_rmse": summary.predicted_rmse,
        "theory_applies": summary.theory_applies,
        "bias_confirmed": summary.bias_confirmed,
        "rmse_confirmed": summary.rmse_confirmed,
        "mean_time_ms": summary.mean_time_ms,
        "merge_left_n": summary.merge_left_n,
        "merge_right_n": summary.merge_right_n,
        "merged_equals_direct": summary.merged_equals_direct,
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


def percent(value: float | None) -> str:
    return MISSING if value is None else f"{100 * value:.2f}%"


def interval(value: Interval | None) -> str:
    return MISSING if value is None else f"[{value.low:.2f}, {value.high:.2f}]"


def mark(ok: bool | None) -> str:
    return MISSING if ok is None else ("✓" if ok else "✗")


def table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


MARKDOWN_HEADER = (
    "| Sketch | Budget (bits) | Bits used | n | Bias | Bias 95% CI | SD | RMSE | RMSE 95% CI "
    "| Relative RMSE | Saturated | Predicted bias | Predicted RMSE | Bias in CI | RMSE in CI "
    "| Time per pass (ms) | Command |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)

MERGE_HEADER = (
    "| Sketch | Budget (bits) | d₁ | d₂ | n = d₁ + d₂ | Merged = direct | Saturated | Bias | RMSE | Relative RMSE |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def markdown_row(summary: Summary) -> str:
    return table_row([
        summary.sketch,
        str(summary.bits),
        str(summary.bits_used),
        str(summary.true_n),
        number(summary.bias),
        interval(summary.bias_ci),
        number(summary.sd),
        number(summary.rmse),
        interval(summary.rmse_ci),
        percent(summary.relative_rmse),
        f"{summary.saturated} of {summary.trials}",
        number(summary.predicted_bias),
        number(summary.predicted_rmse),
        mark(summary.bias_confirmed),
        mark(summary.rmse_confirmed),
        number(summary.mean_time_ms),
        f"`{summary.command}`" if summary.command else MISSING,
    ])


def merge_row(summary: Summary) -> str:
    return table_row([
        base_sketch(summary.sketch),
        str(summary.bits),
        str(summary.merge_left_n),
        str(summary.merge_right_n),
        str(summary.true_n),
        f"{summary.merged_equals_direct} of {summary.trials}",
        f"{summary.saturated} of {summary.trials}",
        number(summary.bias),
        number(summary.rmse),
        percent(summary.relative_rmse),
    ])


def write_markdown(summaries: list[Summary], path: Path) -> None:
    """One table per stream with a row per sketch and memory budget, then the stream's
    mergeability results when there are any."""
    by_stream: dict[str, list[Summary]] = defaultdict(list)
    for summary in summaries:
        by_stream[summary.stream].append(summary)

    sections = []
    for stream, rows in by_stream.items():
        lines = [f"### {stream}", "", MARKDOWN_HEADER, *(markdown_row(row) for row in rows)]
        merges = [row for row in rows if row.merge_left_n is not None]
        if merges:
            lines += ["", "Merging two disjoint parts of the stream:", "", MERGE_HEADER, *(merge_row(row) for row in merges)]
        sections.append("\n".join(lines))

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
