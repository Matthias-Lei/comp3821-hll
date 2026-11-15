"""Draw the figures and key numbers for the report's Linear Counting section.

Run from the repository root after results_analyser.py:

    python3 python/plot_linear_counting.py --summary data/analysis-all/summary.csv

The summary holds every sketch on create_analysis_streams.py's streams (the team's observations
and distribution generators) and the data repo's SQL query streams; this script keeps Linear
Counting and its merge.

Figures (PDF for the report, PNG for a quick look) in data/analysis-lc/figures/:

1. lc_rmse_vs_theory: measured RMSE against Whang's sqrt(m(e^t - t - 1)), one panel per
   budget, on the observations streams and the real SQL query results.
2. lc_bias: measured bias in units of the predicted RMSE.
3. lc_saturation: share of the 64 seeds that saturate (every bit set) against n, with the
   predicted probability exp(-m e^(-n/m)).
4. lc_by_distribution: measured RMSE / predicted RMSE for every distribution of repeats the
   team's generator offers, and for the SQL streams.

lc_key_numbers.md lists the numbers the text quotes: merge exactness, how many predictions
fall inside the measured 95% interval, and where saturation starts.

Where the results are kept
--------------------------
data/ is in .gitignore, so the results of the full run are in the data repo
(https://github.com/Matthias-Lei/comp3821-hll-data), on the branch sid-linear-counting-results,
under results/:

    results/
      README.md                 how the run was made, and the headline numbers
      raw/all-results.tar.gz    the Google Benchmark JSON, one file per stream (905 files,
                                80 MB packed, 1.2 GB unpacked)
      analysis/summary.csv      results_analyser.py's output: 28,960 rows, one per stream,
                                sketch and budget
      analysis/summary.md       the same table as Markdown
      linear-counting/          this script's figures (PDF and PNG) and lc_key_numbers.md

The run covered 5 sketches (LinearCounting, BloomFilter, KMinVals, HyperLogLog and
HyperLogLogPlusPlus, plus the Merge benchmarks of LC, HLL and HLL++), 905 streams (the data
repo's 200 streams in data/streams/ and 705 from create_analysis_streams.py), 4 budgets (8192,
32768, 65536 and 131072 bits) and 64 hash seeds: 1,853,440 trials.

Each summary.csv row is 64 trials of one sketch at one budget on one stream. Its columns:

    stream, sketch, bits, bits_used    the stream's file name, the benchmark name without BM_,
                                       the budget and the bits the sketch actually used
    true_n, trials, saturated          the exact distinct count, the number of seeds, and how
                                       many seeds gave no estimate (a full bitmap)
    bias, bias_low, bias_high          mean(estimate - n) and its 95% confidence interval
    sd, rmse, rmse_low, rmse_high      the spread, the RMSE and the RMSE's 95% interval
    relative_bias, relative_rmse       bias and RMSE divided by n
    predicted_bias, predicted_rmse     the sketch's theory at this n and budget
    theory_applies                     whether n is in the range where the theory is judged
    bias_confirmed, rmse_confirmed     whether the prediction lies inside the measured interval
    mean_time_ms                       time per pass over the stream
    merge_left_n, merge_right_n,       Merge rows only: the distinct counts of the two parts
    merged_equals_direct               and how many seeds gave the same estimate as one
                                       sketch over the whole stream
    command                            the benchmark command that made the JSON

To redraw the figures from the data repo, copy analysis/summary.csv here and run:

    python3 python/plot_linear_counting.py --summary path/to/summary.csv
"""
import argparse
import sys
from pathlib import Path

from plot_sketches import (  # noqa: F401 (re-exported for the tests and the report's text)
    BASE_DIR,
    DEFAULT_SUMMARY,
    KIND_STYLE,
    LINEAR_COUNTING,
    SQL_LABEL,
    distribution_of,
    draw_sketch,
    saturation_n,
    saturation_probability,
    stream_kind,
)
import plot_sketches

DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "analysis-lc" / "figures"
SKETCH = LINEAR_COUNTING.name
MERGE_SKETCH = LINEAR_COUNTING.merge

# The figures are plot_sketches.py's, drawn for Linear Counting; these keep the names the
# report's section and its tests use.


def load_rows(path: Path, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """summary.csv rows of Linear Counting and its merge, each with its stream kind added.
    With kinds, only streams of those kinds."""
    return plot_sketches.load_rows(path, (SKETCH, MERGE_SKETCH), kinds)


def predicted_rmse_curve(m: int, n_low: float, n_high: float) -> tuple[list[float], list[float]]:
    """Predicted RMSE from n_low up to n_high, stopping at m ln m where the formula stops applying."""
    return plot_sketches.predicted_rmse_curve(LINEAR_COUNTING, m, n_low, n_high)


def rmse_ratios(rows: list[dict]) -> dict[str, list[float]]:
    return plot_sketches.rmse_ratios(rows, LINEAR_COUNTING)


def key_numbers(rows: list[dict]) -> str:
    return plot_sketches.key_numbers(rows, LINEAR_COUNTING)


def plot_rmse_vs_theory(rows: list[dict], stem: Path) -> list[Path]:
    return plot_sketches.plot_rmse_vs_theory(rows, LINEAR_COUNTING, stem)


def plot_bias(rows: list[dict], stem: Path) -> list[Path]:
    return plot_sketches.plot_bias(rows, LINEAR_COUNTING, stem)


def plot_saturation(rows: list[dict], stem: Path) -> list[Path]:
    return plot_sketches.plot_saturation(rows, LINEAR_COUNTING, stem)


def plot_by_distribution(rows: list[dict], stem: Path) -> list[Path]:
    return plot_sketches.plot_by_distribution(rows, LINEAR_COUNTING, stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="summary.csv from results_analyser.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for the figures")
    args = parser.parse_args()

    if not args.summary.is_file():
        print(f"no summary at {args.summary} (run python/results_analyser.py first)", file=sys.stderr)
        return 1
    for path in draw_sketch(load_rows(args.summary, tuple(KIND_STYLE)), LINEAR_COUNTING, args.output):
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
