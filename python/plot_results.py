"""Draw the report's two main figures from data/analysis/summary.csv.

Run from the repository root after python/results_analyser.py:

    python3 python/plot_results.py
    python3 python/plot_results.py --stream uniform_observations_seed7427369

1. rmse_vs_theory_<sketch>: measured RMSE against the predicted RMSE, one panel per memory
   budget, for every sketch with a formula in results_analyser.THEORY. The proposal compares
   the "observed RMSE of each sketch against the predicted theoretical guarantee" and counts a
   prediction as confirmed when it falls inside the 95% interval of the measured RMSE.
2. sketch_comparison: relative RMSE of every sketch at each memory budget on one stream, the
   proposal's question of which sketch is most accurate under the same memory budget.

Each figure is saved as PDF (for the report) and PNG (for a quick look) in
data/analysis/figures/.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only: WSL and CI have no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from results_analyser import THEORY  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = BASE_DIR / "data" / "analysis" / "summary.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "analysis" / "figures"
BUDGETS = (8192, 32768, 65536, 131072)
BITS_PER_KIB = 8192

# Categorical slots 1-3 of the dataviz reference palette (light mode), validated for
# colour-blind separation as a set. Each sketch keeps its colour and marker in every figure,
# and the marker shape keeps the sketches apart in greyscale print.
SKETCH_STYLE = {
    "LinearCounting": {"color": "#2a78d6", "marker": "o", "label": "Linear Counting"},
    "BloomFilter": {"color": "#eb6834", "marker": "s", "label": "Bloom filter"},
    "HyperLogLog": {"color": "#1baf7a", "marker": "^", "label": "HyperLogLog"},
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

REPORT_STYLE = {
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": INK,
}

INT_FIELDS = ("bits", "bits_used", "true_n", "trials", "saturated")
FLOAT_FIELDS = ("bias", "bias_low", "bias_high", "rmse", "rmse_low", "rmse_high", "predicted_bias", "predicted_rmse")
BOOL_FIELDS = ("bias_confirmed", "rmse_confirmed")


# --- Loading --------------------------------------------------------------------------------

def parse_row(row: dict[str, str]) -> dict:
    """Converts one summary.csv row from text. Empty cells become None."""
    parsed: dict = dict(row)
    for field in INT_FIELDS:
        parsed[field] = int(row[field])
    for field in FLOAT_FIELDS:
        parsed[field] = float(row[field]) if row[field] else None
    for field in BOOL_FIELDS:
        parsed[field] = {"True": True, "False": False}.get(row[field])
    return parsed


def load_summary(path: Path) -> list[dict]:
    with open(path, newline="") as file:
        return [parse_row(row) for row in csv.DictReader(file)]


# --- Figure 1: measured RMSE against theory -------------------------------------------------

def theory_panels(rows: list[dict], sketch: str) -> dict[int, list[dict]]:
    """The sketch's rows that can be judged against theory, by memory budget and sorted by n.

    Groups with a saturated trial are left out: the formula does not apply once the sketch
    fails, which is also when results_analyser stops checking it.
    """
    panels: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["sketch"] == sketch and row["saturated"] == 0 and row["rmse_low"] is not None and row["rmse"] > 0:
            panels[row["bits"]].append(row)
    return {bits: sorted(group, key=lambda row: row["true_n"]) for bits, group in sorted(panels.items())}


def log_error_bars(values: list[float], lows: list[float], highs: list[float]) -> list[list[float]]:
    """Error-bar lengths for a log axis. A log axis cannot show 0, so an interval whose low end
    is 0 (or below) is drawn down to a tenth of the value instead."""
    floors = [low if low > 0 else value / 10 for value, low in zip(values, lows)]
    return [[value - floor for value, floor in zip(values, floors)], [high - value for value, high in zip(values, highs)]]


def theory_curve(sketch: str, m: int, n_low: float, n_high: float, points: int = 200) -> tuple[list[float], list[float]]:
    """Predicted RMSE at log-spaced n from n_low to n_high."""
    low, high = math.log10(n_low), math.log10(n_high)
    ns = [10 ** (low + (high - low) * i / (points - 1)) for i in range(points)]
    return ns, [THEORY[sketch](n, m)[1] for n in ns]


def plot_rmse_vs_theory(rows: list[dict], sketch: str, stem: Path) -> list[Path]:
    """One panel per budget: measured RMSE with its 95% interval, and the predicted RMSE curve.

    A filled marker means the prediction lies inside the measured interval (confirmed), a hollow
    marker that it does not. Intervals reaching 0 are drawn down to a tenth of the value.
    """
    panels = theory_panels(rows, sketch)
    if not panels:
        return []
    style = SKETCH_STYLE.get(sketch, {"color": SKETCH_STYLE["LinearCounting"]["color"], "marker": "o", "label": sketch})
    all_n = [row["true_n"] for group in panels.values() for row in group]

    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
        handles: dict[str, object] = {}
        for ax, bits in zip(axes.flat, BUDGETS):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{bits // BITS_PER_KIB} KiB (m = {bits:,} bits)", loc="left")

            ns, curve = theory_curve(sketch, bits, min(all_n), max(all_n))
            (line,) = ax.plot(ns, curve, color=INK_SECONDARY, linewidth=1.5)
            handles["predicted RMSE"] = line

            for confirmed, label in ((True, "measured, prediction inside the 95% CI"), (False, "measured, prediction outside the 95% CI")):
                points = [row for row in panels.get(bits, []) if row["rmse_confirmed"] is confirmed]
                if not points:
                    continue
                rmse = [row["rmse"] for row in points]
                handles[label] = ax.errorbar(
                    [row["true_n"] for row in points],
                    rmse,
                    yerr=log_error_bars(rmse, [row["rmse_low"] for row in points], [row["rmse_high"] for row in points]),
                    fmt=style["marker"],
                    markersize=6,
                    markeredgewidth=1.5,
                    color=style["color"],
                    markerfacecolor=style["color"] if confirmed else SURFACE,
                    elinewidth=1,
                    capsize=0,
                )

        fig.suptitle(f"{style['label']}: measured RMSE against theory (64 hash seeds per point)", x=0.06, ha="left", color=INK, fontsize=11)
        fig.supxlabel("true number of distinct values, n", color=INK_SECONDARY)
        fig.supylabel("RMSE of the estimate", color=INK_SECONDARY)
        fig.legend(list(handles.values()), list(handles.keys()), loc="upper left", ncol=len(handles), bbox_to_anchor=(0.05, 0.955))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save(fig, stem)


# --- Figure 2: sketches compared at equal memory --------------------------------------------

def pick_comparison_stream(rows: list[dict]) -> str | None:
    """The stream with the most distinct values on which every sketch gave an unsaturated
    RMSE at every budget, so no point on the comparison is missing or conditional."""
    sketches = {row["sketch"] for row in rows if row["sketch"] in SKETCH_STYLE}
    needed = {(row["sketch"], row["bits"]) for row in rows if row["sketch"] in sketches}
    complete: dict[str, set[tuple[str, int]]] = defaultdict(set)
    true_n: dict[str, int] = {}
    for row in rows:
        if row["sketch"] in sketches and row["saturated"] == 0 and row["rmse"] is not None:
            complete[row["stream"]].add((row["sketch"], row["bits"]))
            true_n[row["stream"]] = row["true_n"]
    candidates = [stream for stream, have in complete.items() if have >= needed]
    return max(candidates, key=lambda stream: (true_n[stream], stream)) if candidates else None


def comparison_series(rows: list[dict], stream: str) -> dict[str, list[dict]]:
    """Each sketch's rows for one stream, sorted by budget, in the fixed sketch order."""
    series: dict[str, list[dict]] = {}
    for sketch in SKETCH_STYLE:
        points = sorted((row for row in rows if row["stream"] == stream and row["sketch"] == sketch and row["rmse"] is not None), key=lambda row: row["bits"])
        if points:
            series[sketch] = points
    return series


def spread_log_positions(values: list[float], min_ratio: float = 1.25) -> list[float]:
    """Nudges label heights on a log axis apart so neighbours differ by at least min_ratio,
    keeping their order. Returns the heights in the input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] * min_ratio)
    return placed


def relative_percent(value: float, true_n: int) -> float:
    return 100 * value / true_n


def percent_tick(value: float, _position: int) -> str:
    """Labels log-axis ticks as percentages, only at the 1, 2, 3 and 5 steps of each decade."""
    if value <= 0:
        return ""
    step = value / 10 ** math.floor(math.log10(value))
    return f"{value:g}%" if round(step) in (1, 2, 3, 5) and math.isclose(step, round(step), rel_tol=1e-6) else ""


def plot_sketch_comparison(rows: list[dict], stream: str, stem: Path) -> list[Path]:
    """Relative RMSE (% of n) of each sketch at the four memory budgets on one stream."""
    series = comparison_series(rows, stream)
    if not series:
        return []
    n = next(iter(series.values()))[0]["true_n"]

    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        ends = []
        for sketch, points in series.items():
            style = SKETCH_STYLE[sketch]
            x = [row["bits"] / BITS_PER_KIB for row in points]
            y = [relative_percent(row["rmse"], n) for row in points]
            lows = [relative_percent(row["rmse_low"] if row["rmse_low"] is not None else row["rmse"], n) for row in points]
            highs = [relative_percent(row["rmse_high"] if row["rmse_high"] is not None else row["rmse"], n) for row in points]
            ax.errorbar(x, y, yerr=log_error_bars(y, lows, highs), color=style["color"], marker=style["marker"], markersize=7,
                        linewidth=2, elinewidth=1, capsize=0, label=style["label"])
            ends.append((style["label"], x[-1], y[-1]))

        # Direct labels at the right-hand end of each line, in text ink, nudged apart where they would overlap.
        for (label, x_end, y_end), y_label in zip(ends, spread_log_positions([y for _, _, y in ends])):
            ax.annotate(label, xy=(x_end, y_end), xytext=(x_end * 1.18, y_label), va="center", color=INK,
                        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8})

        budgets_kib = [bits / BITS_PER_KIB for bits in BUDGETS]
        ax.set_xticks(budgets_kib, [f"{kib:g} KiB" for kib in budgets_kib])
        ax.xaxis.set_minor_locator(NullLocator())
        percent = FuncFormatter(percent_tick)
        ax.yaxis.set_major_formatter(percent)
        ax.yaxis.set_minor_formatter(percent)
        ax.set_xlim(budgets_kib[0] / 1.25, budgets_kib[-1] * 2.2)
        ax.set_xlabel("memory budget")
        ax.set_ylabel("relative RMSE (% of n), 64 hash seeds, 95% CI")
        ax.set_title(f"Accuracy at equal memory: {stream} (n = {n:,})", loc="left")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return save(fig, stem)


# --- Output ---------------------------------------------------------------------------------

def save(fig: plt.Figure, stem: Path) -> list[Path]:
    """Saves the figure as stem.pdf and stem.png, then closes it."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="summary.csv from results_analyser.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for the figures")
    parser.add_argument("--stream", help="stream for the sketch comparison (default: the largest complete one)")
    args = parser.parse_args()

    if not args.summary.is_file():
        print(f"no summary at {args.summary} (run python/results_analyser.py first)", file=sys.stderr)
        return 1
    rows = load_summary(args.summary)
    if args.stream and args.stream not in {row["stream"] for row in rows}:
        print(f"stream {args.stream!r} is not in {args.summary}", file=sys.stderr)
        return 1

    written = []
    for sketch in THEORY:
        written += plot_rmse_vs_theory(rows, sketch, args.output / f"rmse_vs_theory_{sketch}")

    stream = args.stream or pick_comparison_stream(rows)
    if stream is None:
        print("skipped sketch_comparison: no stream has unsaturated results for every sketch and budget", file=sys.stderr)
    else:
        written += plot_sketch_comparison(rows, stream, args.output / "sketch_comparison")

    for path in written:
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
