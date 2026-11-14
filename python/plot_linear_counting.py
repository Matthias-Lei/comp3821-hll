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
"""
import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only: WSL and CI have no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plot_results import (  # noqa: E402
    AXIS,
    BITS_PER_KIB,
    BUDGETS,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    REPORT_STYLE,
    SURFACE,
    load_summary,
    log_error_bars,
    save,
)
from results_analyser import linear_counting_theory  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = BASE_DIR / "data" / "analysis-all" / "summary.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "analysis-lc" / "figures"
SKETCH = "LinearCounting"
MERGE_SKETCH = "LinearCountingMerge"
MIN_JUDGED_N = 100  # below this the 64 errors are mostly 0 or 1 and the interval says little
SEEDS = 64
# If the theory holds, an RMSE measured over 64 seeds is within about 1.96 / sqrt(2 * 64) of the
# true RMSE 95% of the time (the RMSE of normal errors has relative standard error 1 / sqrt(2T)).
RATIO_BAND = 1.96 / math.sqrt(2 * SEEDS)
SQL_LABEL = "real SQL"

LC_STYLE = {**REPORT_STYLE, "axes.grid": False}

# Categorical slots 1-3 of the dataviz palette, checked with its validator (all checks pass;
# aqua is under 3:1 against the surface, so every series also has its own marker and a legend
# entry). Each kind of stream keeps its colour and marker in every figure.
KIND_STYLE = {
    "observations": {"color": "#2a78d6", "marker": "o", "label": "synthetic, team observations generator"},
    "sql": {"color": "#eb6834", "marker": "^", "label": "real SQL query results"},
    "nsample": {"color": "#1baf7a", "marker": "s", "label": "synthetic, team distribution generator"},
}
# Budgets are ordered, so the saturation figure uses one blue ramp, light to dark (validated
# as an ordinal ramp).
BUDGET_COLOURS = dict(zip(BUDGETS, ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")))
SATURATION_DASH = (0, (4, 3))
SATURATION_WINDOW = (0.5, 2.0)  # the saturation figure covers n from 0.5 to 2 times m ln m
REGION_FILL = "#f0efea"  # a step off the surface: marks a region without competing with data


# --- Loading --------------------------------------------------------------------------------

def stream_kind(stream: str) -> str | None:
    """Which kind of stream a name is: one of KIND_STYLE, or None for any other stream.

    Names follow create_test_streams.py: <distribution>_observations_seed<s>_m<m> and
    <distribution>_seed<s>_nsample<n>; the data repo's SQL streams contain _query_.
    """
    if "_query_" in stream:
        return "sql"
    if "_observations_" in stream:
        return "observations"
    if "_nsample" in stream:
        return "nsample"
    return None


def distribution_of(stream: str) -> str | None:
    """The distribution of repeats in an observations stream, SQL_LABEL for SQL, else None."""
    kind = stream_kind(stream)
    if kind == "sql":
        return SQL_LABEL
    if kind == "observations":
        return stream.split("_observations_")[0]
    return None


def optional_int(text: str | None) -> int | None:
    return int(text) if text else None


def load_rows(path: Path, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """summary.csv rows of Linear Counting and its merge, each with its stream kind added.
    With kinds, only streams of those kinds."""
    rows = []
    for row in load_summary(path):
        kind = stream_kind(row["stream"])
        if row["sketch"] in (SKETCH, MERGE_SKETCH) and (kinds is None or kind in kinds):
            rows.append({**row, "kind": kind, "merged_equals_direct": optional_int(row.get("merged_equals_direct"))})
    return rows


def judged(rows: list[dict], kinds: tuple[str, ...]) -> dict[int, list[dict]]:
    """Unsaturated Linear Counting rows of the given kinds with an RMSE interval, by budget, sorted by n."""
    panels: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["sketch"] == SKETCH and row["kind"] in kinds and row["saturated"] == 0 and row["rmse_low"] is not None and row["rmse"] > 0:
            panels[row["bits"]].append(row)
    return {bits: sorted(group, key=lambda row: row["true_n"]) for bits, group in panels.items()}


# --- Theory ---------------------------------------------------------------------------------

def saturation_n(m: int) -> float:
    """n = m ln m, around where every bit is expected to be set (the coupon collector's bound)."""
    return m * math.log(m)


def saturation_probability(n: float, m: int) -> float:
    """Chance that all m bits are set after n distinct values: (1 - e^(-n/m))^m ~ exp(-m e^(-n/m))."""
    return math.exp(-m * math.exp(-n / m))


def log_space(low: float, high: float, points: int = 200) -> list[float]:
    a, b = math.log10(low), math.log10(high)
    return [10 ** (a + (b - a) * i / (points - 1)) for i in range(points)]


def predicted_rmse_curve(m: int, n_low: float, n_high: float) -> tuple[list[float], list[float]]:
    """Predicted RMSE from n_low up to n_high, stopping at m ln m where the formula stops applying."""
    ns = log_space(n_low, min(n_high, saturation_n(m)))
    return ns, [linear_counting_theory(n, m)[1] for n in ns]


# --- Figures --------------------------------------------------------------------------------

def budget_axes(title: str, ylabel: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, bits in zip(axes.flat, BUDGETS):
        ax.set_title(f"{bits // BITS_PER_KIB} KiB (m = {bits:,} bits)", loc="left")
        ax.set_xscale("log")
    fig.suptitle(title, x=0.06, ha="left", color=INK, fontsize=11)
    fig.supxlabel("true number of distinct values, n", color=INK_SECONDARY)
    fig.supylabel(ylabel, color=INK_SECONDARY)
    return fig, list(axes.flat)


def mark_regions(ax: plt.Axes, m: int) -> None:
    """Shades n < MIN_JUDGED_N (too few collisions to judge) and dashes n = m ln m (saturation)."""
    ax.axvspan(0.5, MIN_JUDGED_N, color=REGION_FILL, linewidth=0, zorder=0)
    ax.axvline(saturation_n(m), color=INK_MUTED, linewidth=1, linestyle=SATURATION_DASH)


def region_handles() -> dict[str, object]:
    return {
        f"shaded: n < {MIN_JUDGED_N}, too few collisions to judge": Patch(color=REGION_FILL),
        "dashed: n = m ln m (saturation)": Line2D([], [], color=INK_MUTED, linewidth=1, linestyle=SATURATION_DASH),
    }


def finish(fig: plt.Figure, handles: dict[str, object], stem: Path) -> list[Path]:
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper left", ncol=2, bbox_to_anchor=(0.05, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return save(fig, stem)


def plot_rmse_vs_theory(rows: list[dict], stem: Path) -> list[Path]:
    """Measured RMSE (95% interval) on observations and SQL streams, with the predicted RMSE.

    Filled markers: the prediction is inside the measured interval. Hollow: it is not.
    The dashed line is n = m ln m; groups with a saturated seed are left out.
    """
    panels = judged(rows, ("observations", "sql"))
    if not panels:
        return []
    all_n = [row["true_n"] for group in panels.values() for row in group]
    with plt.rc_context(LC_STYLE):
        fig, axes = budget_axes("Linear Counting: measured RMSE against theory (64 hash seeds per point)", "RMSE of the estimate")
        handles: dict[str, object] = {}
        for ax, bits in zip(axes, BUDGETS):
            ax.set_yscale("log")
            (line,) = ax.plot(*predicted_rmse_curve(bits, min(all_n), max(all_n)), color=INK_SECONDARY, linewidth=1.5)
            handles["predicted RMSE, sqrt(m(e^t - t - 1))"] = line
            mark_regions(ax, bits)
            for kind in ("observations", "sql"):
                style = KIND_STYLE[kind]
                for confirmed in (True, False):
                    points = [row for row in panels.get(bits, []) if row["kind"] == kind and row["rmse_confirmed"] is confirmed]
                    if not points:
                        continue
                    rmse = [row["rmse"] for row in points]
                    bars = ax.errorbar(
                        [row["true_n"] for row in points], rmse,
                        yerr=log_error_bars(rmse, [row["rmse_low"] for row in points], [row["rmse_high"] for row in points]),
                        fmt=style["marker"], markersize=6, markeredgewidth=1.5, color=style["color"],
                        markerfacecolor=style["color"] if confirmed else SURFACE, elinewidth=1, capsize=0, alpha=0.9,
                    )
                    if confirmed:
                        handles[style["label"]] = bars
        handles["hollow: prediction outside the 95% CI"] = Line2D(
            [], [], linestyle="none", marker="o", markersize=6, markeredgewidth=1.5, color=INK_MUTED, markerfacecolor=SURFACE)
        return finish(fig, {**handles, **region_handles()}, stem)


def plot_bias(rows: list[dict], stem: Path) -> list[Path]:
    """Measured bias divided by the predicted RMSE, with its 95% interval scaled the same way."""
    panels = judged(rows, ("observations", "sql"))
    if not panels:
        return []
    with plt.rc_context(LC_STYLE):
        fig, axes = budget_axes("Linear Counting: bias in units of the predicted RMSE", "bias / predicted RMSE")
        handles: dict[str, object] = {}
        for ax, bits in zip(axes, BUDGETS):
            ax.axhline(0, color=AXIS, linewidth=1)
            mark_regions(ax, bits)
            for kind in ("observations", "sql"):
                style = KIND_STYLE[kind]
                points = [row for row in panels.get(bits, []) if row["kind"] == kind and row["bias_low"] is not None]
                if not points:
                    continue
                scale = [row["predicted_rmse"] for row in points]
                bias = [row["bias"] / s for row, s in zip(points, scale)]
                low = [(row["bias"] - row["bias_low"]) / s for row, s in zip(points, scale)]
                high = [(row["bias_high"] - row["bias"]) / s for row, s in zip(points, scale)]
                handles[style["label"]] = ax.errorbar(
                    [row["true_n"] for row in points], bias, yerr=[low, high], fmt=style["marker"], markersize=6,
                    color=style["color"], elinewidth=1, capsize=0, alpha=0.9,
                )
        return finish(fig, {**handles, **region_handles()}, stem)


def plot_saturation(rows: list[dict], stem: Path) -> list[Path]:
    """Share of seeds that saturate against n, one ramp colour per budget, with the prediction."""
    observations = [row for row in rows if row["sketch"] == SKETCH and row["kind"] == "observations"]
    near = {
        bits: sorted((row for row in observations if row["bits"] == bits and SATURATION_WINDOW[0] <= row["true_n"] / saturation_n(bits) <= SATURATION_WINDOW[1]),
                     key=lambda row: row["true_n"])
        for bits in BUDGETS
    }
    if not any(near.values()):
        return []
    with plt.rc_context(LC_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        for bits in BUDGETS:
            colour = BUDGET_COLOURS[bits]
            ns = log_space(SATURATION_WINDOW[0] * saturation_n(bits), SATURATION_WINDOW[1] * saturation_n(bits))
            ax.plot(ns, [saturation_probability(n, bits) for n in ns], color=colour, linewidth=1.5)
            ax.plot([row["true_n"] for row in near[bits]], [row["saturated"] / row["trials"] for row in near[bits]],
                    "o", color=colour, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1)
            centre = saturation_n(bits)
            ax.annotate(f"{bits // BITS_PER_KIB} KiB", (centre, saturation_probability(centre, bits)),
                        xytext=(-8, 6), textcoords="offset points", ha="right", color=INK_SECONDARY)
        ax.set_title("Linear Counting: share of the 64 hash seeds where every bit is set", loc="left", color=INK, fontsize=11, pad=30)
        ax.set_xlabel("true number of distinct values, n")
        ax.set_ylabel("share of seeds saturated")
        ax.legend([Line2D([], [], color=INK_SECONDARY, linewidth=1.5), Line2D([], [], linestyle="none", marker="o", color=INK_SECONDARY)],
                  ["predicted, exp(-m e^(-n/m))", "measured, observations streams"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        fig.tight_layout()
        return save(fig, stem)


def rmse_ratios(rows: list[dict]) -> dict[str, list[float]]:
    """Measured RMSE / predicted RMSE of every judged group with n >= MIN_JUDGED_N, by distribution
    of repeats (alphabetical), with the SQL streams last."""
    ratios: dict[str, list[float]] = defaultdict(list)
    for group in judged(rows, ("observations", "sql")).values():
        for row in group:
            if row["true_n"] >= MIN_JUDGED_N:
                ratios[distribution_of(row["stream"])].append(row["rmse"] / row["predicted_rmse"])
    order = sorted(name for name in ratios if name != SQL_LABEL) + ([SQL_LABEL] if SQL_LABEL in ratios else [])
    return {name: ratios[name] for name in order}


def plot_by_distribution(rows: list[dict], stem: Path) -> list[Path]:
    """One row per distribution of repeats: each dot is one stream and budget's measured RMSE
    divided by the predicted RMSE. The shaded band is where 95% of dots fall if the theory holds."""
    ratios = rmse_ratios(rows)
    if not ratios:
        return []
    with plt.rc_context(LC_STYLE):
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(ratios) + 1.8))
        ax.axvspan(1 - RATIO_BAND, 1 + RATIO_BAND, color=REGION_FILL, linewidth=0, zorder=0)
        ax.axvline(1, color=INK_SECONDARY, linewidth=1)
        for y, (name, values) in enumerate(ratios.items()):
            style = KIND_STYLE["sql" if name == SQL_LABEL else "observations"]
            ax.plot(values, [y] * len(values), style["marker"], color=style["color"], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, alpha=0.75)
        ax.set_yticks(range(len(ratios)), [f"{name} ({len(values)})" for name, values in ratios.items()])
        ax.invert_yaxis()
        ax.set_xlabel("measured RMSE / predicted RMSE (1 = exactly as predicted)")
        ax.set_title(f"Linear Counting on every distribution of repeats (n >= {MIN_JUDGED_N}, all budgets)",
                     loc="left", color=INK, fontsize=11, pad=24)
        ax.legend([Patch(color=REGION_FILL), Line2D([], [], color=INK_SECONDARY, linewidth=1)],
                  ["shaded: where 95% of dots fall if the theory holds", "measured = predicted"],
                  loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        fig.tight_layout()
        return save(fig, stem)


# --- Key numbers ----------------------------------------------------------------------------

def judged_for_key_numbers(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["sketch"] == SKETCH and row["true_n"] >= MIN_JUDGED_N and row["rmse_confirmed"] is not None]


def confirmation_line(rows: list[dict], label: str) -> str:
    rmse = sum(bool(row["rmse_confirmed"]) for row in rows)
    bias = sum(bool(row["bias_confirmed"]) for row in rows)
    return f"| {label} | {rmse} of {len(rows)} | {bias} of {len(rows)} |"


def first_saturated_n(rows: list[dict], bits: int) -> int | None:
    saturated = [row["true_n"] for row in rows if row["sketch"] == SKETCH and row["bits"] == bits and row["saturated"] > 0]
    return min(saturated) if saturated else None


def key_numbers(rows: list[dict]) -> str:
    merges = [row for row in rows if row["sketch"] == MERGE_SKETCH and row["merged_equals_direct"] is not None]
    judged_rows = judged_for_key_numbers(rows)
    lines = [
        "# Linear Counting: key numbers",
        "",
        f"- **Merge:** the OR of the two halves' bitmaps equals the bitmap of the whole stream in "
        f"**{sum(row['merged_equals_direct'] for row in merges):,} of {sum(row['trials'] for row in merges):,}** trials.",
    ]
    lines += ["", f"## Predictions inside the measured 95% interval (n >= {MIN_JUDGED_N}, no seed saturated)", "",
              "| Streams, budget | RMSE | Bias |", "|---|---|---|"]
    for kind, style in KIND_STYLE.items():
        for bits in BUDGETS:
            group = [row for row in judged_rows if row["kind"] == kind and row["bits"] == bits]
            lines.append(confirmation_line(group, f"{style['label']}, {bits // BITS_PER_KIB} KiB"))
    ratios = rmse_ratios(rows)
    lines += ["", "## By distribution of repeats (all budgets)", "",
              "| Distribution | RMSE | Bias | Median measured / predicted RMSE |", "|---|---|---|---|"]
    for name, values in ratios.items():
        group = [row for row in judged_rows if distribution_of(row["stream"]) == name]
        lines.append(f"{confirmation_line(group, name)} {statistics.median(values):.3f} |")
    lines += ["", "## Saturation", "", "| Budget | m ln m | Smallest n with a saturated seed |", "|---|---|---|"]
    for bits in BUDGETS:
        first = first_saturated_n(rows, bits)
        lines.append(f"| {bits // BITS_PER_KIB} KiB | {saturation_n(bits):,.0f} | {f'{first:,}' if first else 'none'} |")
    return "\n".join(lines) + "\n"


# --- Command line ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="summary.csv from results_analyser.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for the figures")
    args = parser.parse_args()

    if not args.summary.is_file():
        print(f"no summary at {args.summary} (run python/results_analyser.py first)", file=sys.stderr)
        return 1
    rows = load_rows(args.summary, tuple(KIND_STYLE))

    written = [
        *plot_rmse_vs_theory(rows, args.output / "lc_rmse_vs_theory"),
        *plot_bias(rows, args.output / "lc_bias"),
        *plot_saturation(rows, args.output / "lc_saturation"),
        *plot_by_distribution(rows, args.output / "lc_by_distribution"),
    ]
    numbers = args.output / "lc_key_numbers.md"
    numbers.parent.mkdir(parents=True, exist_ok=True)
    numbers.write_text(key_numbers(rows), encoding="utf-8")
    for path in written + [numbers]:
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
