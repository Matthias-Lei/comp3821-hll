"""Draw each sketch's own figures and key numbers: Linear Counting, the Bloom filter, K-Minimum
Values, HyperLogLog and HyperLogLog++.

Run from the repository root after results_analyser.py:

    python3 python/plot_sketches.py --summary data/analysis-all/summary.csv
    python3 python/plot_sketches.py --sketch HyperLogLog --sketch KMinVals

For every sketch, in data/analysis-all/figures/<short name>/ (PDF for the report, PNG for a
quick look), where the short names are lc, bloom, kmv, hll and hllpp:

1. <short>_rmse_vs_theory: measured RMSE against the sketch's predicted RMSE
   (results_analyser.THEORY), one panel per budget, on the observations streams and the real
   SQL query results. Filled: the prediction is inside the measured 95% interval. Hollow: it is
   not. Grey: the formula is not judged at that n (results_analyser.THEORY_APPLIES).
2. <short>_bias: measured bias in units of the predicted RMSE.
3. <short>_by_distribution: measured RMSE / predicted RMSE for every distribution of repeats
   the team's generator offers, and for the SQL streams.
4. <short>_saturation, for the sketches whose bitmap can fill up (Linear Counting and the
   Bloom filter): share of the 64 seeds with every bit set, with the predicted probability.

<short>_key_numbers.md lists the numbers the text quotes: merge exactness, how many predictions
fall inside the measured 95% interval, and, where they apply, exactness below k and saturation.

plot_comparison.py draws the sketches against each other. plot_linear_counting.py's docstring
says where the full run's summary.csv is kept and what its columns mean.
"""
import argparse
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only: WSL and CI have no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

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
from results_analyser import (  # noqa: E402
    BLOOM_HASHES,
    HLL_SMALL_RANGE,
    KMV_BITS_PER_VALUE,
    THEORY,
    THEORY_APPLIES,
    hyperloglog_registers,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = BASE_DIR / "data" / "analysis-all" / "summary.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "analysis-all" / "figures"
MIN_JUDGED_N = 100  # below this the 64 errors are mostly 0 or 1 and the interval says little
SEEDS = 64
# If the theory holds, an RMSE measured over 64 seeds is within about 1.96 / sqrt(2 * 64) of the
# true RMSE 95% of the time (the RMSE of normal errors has relative standard error 1 / sqrt(2T)).
RATIO_BAND = 1.96 / math.sqrt(2 * SEEDS)
SQL_LABEL = "real SQL"

FIGURE_STYLE = {**REPORT_STYLE, "axes.grid": False}

# Categorical slots 1-3 of the dataviz palette, checked with its validator (all checks pass;
# aqua is under 3:1 against the surface, so every series also has its own marker and a legend
# entry). Each kind of stream keeps its colour and marker in every figure.
KIND_STYLE = {
    "observations": {"color": "#2a78d6", "marker": "o", "label": "synthetic, team observations generator"},
    "sql": {"color": "#eb6834", "marker": "^", "label": "real SQL query results"},
    "nsample": {"color": "#1baf7a", "marker": "s", "label": "synthetic, team distribution generator"},
}
SHOWN_KINDS = ("observations", "sql")  # the kinds the per-sketch figures draw
# Budgets are ordered, so figures coloured by budget use one blue ramp, light to dark
# (validated as an ordinal ramp).
BUDGET_COLOURS = dict(zip(BUDGETS, ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")))
MARK_DASH = (0, (4, 3))
# Relative RMSE axes run from 0.01% to 100% of n. HyperLogLog++'s sparse mode is near exact (down
# to 1e-5 %), which would squash the range where the sketches differ, so anything under the
# floor goes on the bottom edge with the exact results, and anything saturated on the top edge.
PERCENT_FLOOR, PERCENT_CEILING = 0.01, 100.0
EDGE_STEP = 0.035  # each series' row of edge markers sits this far (axes fraction) from the last
SATURATION_WINDOW = (0.5, 2.0)  # the saturation figure covers n from 0.5 to 2 times its saturation n
REGION_FILL = "#f0efea"  # a step off the surface: marks a region without competing with data


# --- The sketches ---------------------------------------------------------------------------

def saturation_n(m: int, hashes: int = 1) -> float:
    """n where every bit of an m-bit bitmap is expected to be set, setting `hashes` bits per value:
    k n = m ln m (the coupon collector's bound)."""
    return m * math.log(m) / hashes


def saturation_probability(n: float, m: int, hashes: int = 1) -> float:
    """Chance that all m bits are set after n distinct values set `hashes` bits each:
    (1 - e^(-k n/m))^m ~ exp(-m e^(-k n/m))."""
    return math.exp(-m * math.exp(-hashes * n / m))


def kmv_k(m: int) -> int:
    """KMV keeps k = B / 64 hash values."""
    return m // KMV_BITS_PER_VALUE


@dataclass(frozen=True)
class Marker:
    """A dashed vertical line at n = at(m) on every budget panel."""
    at: Callable[[int], float]
    label: str


@dataclass(frozen=True)
class SketchSpec:
    """What the figures need to know about one sketch.

    name and merge are the benchmark names without BM_. hashes is the number of bits each value
    sets, for sketches whose bitmap can fill up (None otherwise). exact_below(m) is the n below
    which the sketch must be exact.
    """
    name: str
    label: str
    short: str
    theory_label: str
    merge: str | None = None
    merge_phrase: str = "the merged sketch gives the same estimate as one built over the whole stream"
    hashes: int | None = None
    marker: Marker | None = None
    shade_label: str = f"shaded: n < {MIN_JUDGED_N}, too few collisions to judge"
    exact_below: Callable[[int], int] | None = None


LINEAR_COUNTING = SketchSpec(
    "LinearCounting", "Linear Counting", "lc", "sqrt(m(e^t - t - 1))",
    merge="LinearCountingMerge",
    merge_phrase="the OR of the two halves' bitmaps equals the bitmap of the whole stream",
    hashes=1, marker=Marker(saturation_n, "dashed: n = m ln m (saturation)"),
)
BLOOM_FILTER = SketchSpec(
    "BloomFilter", "Bloom filter", "bloom", f"sqrt(m(e^kt - kt - 1)) / k, k = {BLOOM_HASHES}",
    hashes=BLOOM_HASHES,
    marker=Marker(lambda m: saturation_n(m, BLOOM_HASHES), "dashed: n = (m/k) ln m (saturation)"),
)
K_MINIMUM_VALUES = SketchSpec(
    "KMinVals", "K-Minimum Values", "kmv", "sqrt(n(n - k + 1) / (k - 2)), k = B/64",
    marker=Marker(kmv_k, "dashed: n = k = B/64 (exact below)"), exact_below=kmv_k,
)
HYPERLOGLOG = SketchSpec(
    "HyperLogLog", "HyperLogLog", "hll", "1.04 n / sqrt(r) (LC on the r registers below 2.5 r)",
    merge="HyperLogLogMerge",
    merge_phrase="the register-wise maximum of the two halves gives the same estimate as the whole stream",
    marker=Marker(lambda m: HLL_SMALL_RANGE * hyperloglog_registers(m), "dashed: n = 2.5 r (estimator switch)"),
    shade_label=f"shaded: not judged (n < {MIN_JUDGED_N}, or 2 r to 5 r)",
)
HYPERLOGLOG_PLUS_PLUS = SketchSpec(
    "HyperLogLogPlusPlus", "HyperLogLog++", "hllpp", "LC over 2^25 (sparse), 1.04 n / sqrt(r) (dense)",
    merge="HyperLogLogPlusPlusMerge",
    marker=Marker(lambda m: 5 * hyperloglog_registers(m), "dashed: n = 5 r (judged from here)"),
    shade_label="shaded: not judged (n < 5 r: sparse, bias correction)",
)
SPECS = {spec.name: spec for spec in (LINEAR_COUNTING, BLOOM_FILTER, K_MINIMUM_VALUES, HYPERLOGLOG, HYPERLOGLOG_PLUS_PLUS)}


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


def optional_float(text: str | None) -> float | None:
    return float(text) if text else None


def load_rows(path: Path, sketches: tuple[str, ...] | None = None, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """summary.csv rows, each with its stream kind added. With sketches or kinds, only rows of
    those sketches (benchmark names without BM_) or on streams of those kinds."""
    rows = []
    for row in load_summary(path):
        kind = stream_kind(row["stream"])
        if (sketches is None or row["sketch"] in sketches) and (kinds is None or kind in kinds):
            rows.append({
                **row, "kind": kind,
                "merged_equals_direct": optional_int(row.get("merged_equals_direct")),
                "mean_time_ms": optional_float(row.get("mean_time_ms")),
            })
    return rows


def by_n(rows: list[dict], sketch: str, bits: int) -> list[dict]:
    """The sketch's rows at one budget on the observations streams, one per n (n > 0), sorted.
    Every observations stream with the same n gives the same result, so the first is kept."""
    found: dict[int, dict] = {}
    for row in rows:
        if row["sketch"] == sketch and row["bits"] == bits and row["kind"] == "observations" and row["true_n"] > 0:
            found.setdefault(row["true_n"], row)
    return [found[n] for n in sorted(found)]


def state(row: dict) -> str:
    """'saturated' (no estimate), 'exact' (RMSE 0) or 'measured'."""
    if row["saturated"] > 0 or row["rmse"] is None:
        return "saturated"
    return "exact" if row["rmse"] == 0 else "measured"


def percent_of_n(value: float | None, row: dict | None) -> float | None:
    return None if value is None or row is None else 100 * value / row["true_n"]


def edge(row: dict | None) -> str | None:
    """Where a row goes on a relative RMSE axis: 'top' (saturated, or over PERCENT_CEILING),
    'bottom' (exact or under PERCENT_FLOOR), or None (drawn as a point; also None for no row)."""
    if row is None:
        return None
    if state(row) == "saturated" or percent_of_n(row["rmse"], row) >= PERCENT_CEILING:
        return "top"
    if state(row) == "exact" or percent_of_n(row["rmse"], row) < PERCENT_FLOOR:
        return "bottom"
    return None


def judged(rows: list[dict], spec: SketchSpec, kinds: tuple[str, ...]) -> dict[int, list[dict]]:
    """Unsaturated rows of the sketch on the given kinds with n > 0, an RMSE interval and RMSE > 0,
    by budget, sorted by n. (A log axis cannot show 0: an empty stream, or KMV exact below k.)"""
    panels: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if (row["sketch"] == spec.name and row["kind"] in kinds and row["true_n"] > 0 and row["saturated"] == 0
                and row["rmse_low"] is not None and row["rmse"] > 0):
            panels[row["bits"]].append(row)
    return {bits: sorted(group, key=lambda row: row["true_n"]) for bits, group in panels.items()}


# --- Theory ---------------------------------------------------------------------------------

def log_space(low: float, high: float, points: int = 200) -> list[float]:
    a, b = math.log10(low), math.log10(high)
    return [10 ** (a + (b - a) * i / (points - 1)) for i in range(points)]


def predicted_rmse_curve(spec: SketchSpec, m: int, n_low: float, n_high: float) -> tuple[list[float], list[float]]:
    """Predicted RMSE from n_low to n_high, stopping where the bitmap saturates, and leaving out
    n where the prediction is 0 (exact) or infinite."""
    if spec.hashes:
        n_high = min(n_high, saturation_n(m, spec.hashes))
    if n_high <= n_low:
        return [], []
    points = [(n, THEORY[spec.name](n, m)[1]) for n in log_space(n_low, n_high)]
    kept = [(n, rmse) for n, rmse in points if 0 < rmse < math.inf]
    return [n for n, _ in kept], [rmse for _, rmse in kept]


def unjudged_ranges(spec: SketchSpec, m: int, n_low: float, n_high: float) -> list[tuple[float, float]]:
    """The ranges of n in [n_low, n_high] where results_analyser does not judge the formula."""
    applies = THEORY_APPLIES.get(spec.name)
    if applies is None:
        return []
    ranges, start = [], None
    ns = log_space(n_low, n_high, 400)
    for n in ns:
        if not applies(n, m):
            start = n if start is None else start
        elif start is not None:
            ranges.append((start, n))
            start = None
    if start is not None:
        ranges.append((start, ns[-1]))
    return ranges


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


def mark_regions(ax: plt.Axes, spec: SketchSpec, m: int, n_high: float) -> None:
    """Shades n < MIN_JUDGED_N and where the formula is not judged, and dashes the sketch's marker."""
    ax.axvspan(0.5, MIN_JUDGED_N, color=REGION_FILL, linewidth=0, zorder=0)
    for low, high in unjudged_ranges(spec, m, MIN_JUDGED_N, n_high):
        ax.axvspan(low, high, color=REGION_FILL, linewidth=0, zorder=0)
    if spec.marker:
        ax.axvline(spec.marker.at(m), color=INK_MUTED, linewidth=1, linestyle=MARK_DASH)


def region_handles(spec: SketchSpec) -> dict[str, object]:
    handles: dict[str, object] = {spec.shade_label: Patch(color=REGION_FILL)}
    if spec.marker:
        handles[spec.marker.label] = Line2D([], [], color=INK_MUTED, linewidth=1, linestyle=MARK_DASH)
    return handles


def finish(fig: plt.Figure, handles: dict[str, object], stem: Path) -> list[Path]:
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper left", ncol=2, bbox_to_anchor=(0.05, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return save(fig, stem)


NOT_JUDGED_LABEL = "grey: the formula is not judged at this n"


def plot_rmse_vs_theory(rows: list[dict], spec: SketchSpec, stem: Path) -> list[Path]:
    """Measured RMSE (95% interval) on observations and SQL streams, with the predicted RMSE.

    Filled markers: the prediction is inside the measured interval. Hollow: it is not. Grey:
    the formula is not judged at that n. Groups with a saturated seed are left out.
    """
    panels = judged(rows, spec, SHOWN_KINDS)
    if not panels:
        return []
    all_n = [row["true_n"] for group in panels.values() for row in group]
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = budget_axes(f"{spec.label}: measured RMSE against theory (64 hash seeds per point)", "RMSE of the estimate")
        handles: dict[str, object] = {}
        for ax, bits in zip(axes, BUDGETS):
            ax.set_yscale("log")
            ns, curve = predicted_rmse_curve(spec, bits, min(all_n), max(all_n))
            if ns:
                (line,) = ax.plot(ns, curve, color=INK_SECONDARY, linewidth=1.5)
                handles[f"predicted RMSE, {spec.theory_label}"] = line
            mark_regions(ax, spec, bits, 2 * max(all_n))
            for kind in SHOWN_KINDS:
                style = KIND_STYLE[kind]
                for confirmed in (True, False, None):
                    points = [row for row in panels.get(bits, []) if row["kind"] == kind and row["rmse_confirmed"] is confirmed]
                    if not points:
                        continue
                    colour = INK_MUTED if confirmed is None else style["color"]
                    rmse = [row["rmse"] for row in points]
                    bars = ax.errorbar(
                        [row["true_n"] for row in points], rmse,
                        yerr=log_error_bars(rmse, [row["rmse_low"] for row in points], [row["rmse_high"] for row in points]),
                        fmt=style["marker"], markersize=6, markeredgewidth=1.5, color=colour,
                        markerfacecolor=SURFACE if confirmed is False else colour, elinewidth=1, capsize=0, alpha=0.9,
                    )
                    if confirmed:
                        handles[style["label"]] = bars
                    elif confirmed is False:  # a kind with no filled point still gets its legend entry
                        handles.setdefault(style["label"], Line2D([], [], linestyle="none", marker=style["marker"],
                                                                  markersize=6, color=style["color"]))
                    else:
                        handles[NOT_JUDGED_LABEL] = Line2D([], [], linestyle="none", marker="o", markersize=6, color=INK_MUTED)
        handles["hollow: prediction outside the 95% CI"] = Line2D(
            [], [], linestyle="none", marker="o", markersize=6, markeredgewidth=1.5, color=INK_MUTED, markerfacecolor=SURFACE)
        return finish(fig, {**handles, **region_handles(spec)}, stem)


def has_prediction(row: dict) -> bool:
    return row["predicted_rmse"] is not None and 0 < row["predicted_rmse"] < math.inf


def plot_bias(rows: list[dict], spec: SketchSpec, stem: Path) -> list[Path]:
    """Measured bias divided by the predicted RMSE, with its 95% interval scaled the same way."""
    panels = judged(rows, spec, SHOWN_KINDS)
    if not panels:
        return []
    all_n = [row["true_n"] for group in panels.values() for row in group]
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = budget_axes(f"{spec.label}: bias in units of the predicted RMSE", "bias / predicted RMSE")
        handles: dict[str, object] = {}
        for ax, bits in zip(axes, BUDGETS):
            ax.axhline(0, color=AXIS, linewidth=1)
            mark_regions(ax, spec, bits, 2 * max(all_n))
            for kind in SHOWN_KINDS:
                style = KIND_STYLE[kind]
                points = [row for row in panels.get(bits, []) if row["kind"] == kind and row["bias_low"] is not None and has_prediction(row)]
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
        return finish(fig, {**handles, **region_handles(spec)}, stem)


def decade_percent(value: float, _position: int) -> str:
    """Labels log-axis ticks as percentages at powers of ten only."""
    return f"{value:g}%" if value > 0 and math.isclose(math.log10(value), round(math.log10(value)), abs_tol=1e-9) else ""


def percent_axis(ax: plt.Axes) -> None:
    """A log axis of relative RMSE from PERCENT_FLOOR to PERCENT_CEILING, labelled by decade."""
    ax.set_yscale("log")
    ax.set_ylim(PERCENT_FLOOR, PERCENT_CEILING)
    ax.yaxis.set_major_formatter(FuncFormatter(decade_percent))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def absolute_edge(row: dict) -> str | None:
    """Where a row goes on an absolute RMSE axis: 'top' (saturated), 'bottom' (RMSE exactly 0,
    which a log axis cannot show), or None (drawn as a point)."""
    return {"saturated": "top", "exact": "bottom"}.get(state(row))


def plot_rmse_by_budget(rows: list[dict], spec: SketchSpec, stem: Path, relative: bool) -> list[Path]:
    """RMSE (relative: % of n; otherwise absolute) against the cardinality n with every budget on one
    axis: the measured points (95% CI, one observations stream per n) and the predicted curve in
    the budget's colour. Saturated groups sit on the top edge, exact ones on the bottom edge."""
    series = {bits: by_n(rows, spec.name, bits) for bits in BUDGETS}
    if not any(series.values()):
        return []
    all_n = [row["true_n"] for points in series.values() for row in points]
    where_of = edge if relative else absolute_edge
    scale = (lambda value, row: percent_of_n(value, row)) if relative else (lambda value, row: value)
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax.set_xscale("log")
        if relative:
            percent_axis(ax)
        else:
            ax.set_yscale("log")
        ax.axvspan(min(all_n) / 1.5, MIN_JUDGED_N, color=REGION_FILL, linewidth=0, zorder=0)
        handles: dict[str, object] = {}
        transform = ax.get_xaxis_transform()
        for index, bits in enumerate(BUDGETS):
            colour = BUDGET_COLOURS[bits]
            ns, curve = predicted_rmse_curve(spec, bits, min(all_n), max(all_n))
            ax.plot(ns, [scale(rmse, {"true_n": n}) for n, rmse in zip(ns, curve)], color=colour, linewidth=1.2,
                    linestyle=MARK_DASH)
            points = [row for row in series[bits] if where_of(row) is None]
            values = [scale(row["rmse"], row) for row in points]
            ax.errorbar([row["true_n"] for row in points], values,
                        yerr=log_error_bars(values, [scale(row["rmse_low"], row) for row in points],
                                            [scale(row["rmse_high"], row) for row in points]),
                        fmt="o-", color=colour, markersize=4.5, linewidth=1.2, elinewidth=0.8, capsize=0,
                        markeredgecolor=SURFACE, markeredgewidth=0.6)
            for where, y in (("top", 1 - EDGE_STEP * (index + 0.5)), ("bottom", EDGE_STEP * (index + 0.5))):
                xs = [row["true_n"] for row in series[bits] if where_of(row) == where]
                if xs:
                    ax.plot(xs, [y] * len(xs), "o", color=colour, markersize=5, markerfacecolor=SURFACE,
                            markeredgewidth=1.2, transform=transform, clip_on=False)
            handles[f"{bits // BITS_PER_KIB} KiB (m = {bits:,} bits)"] = Line2D([], [], color=colour, marker="o", linewidth=1.2)
        hollow = {"linestyle": "none", "marker": "o", "markersize": 6, "color": INK_MUTED, "markerfacecolor": SURFACE}
        handles[f"dashed: predicted, {spec.theory_label}"] = Line2D([], [], color=INK_MUTED, linewidth=1.2, linestyle=MARK_DASH)
        handles["hollow, top edge: a seed saturated (no estimate)"] = Line2D([], [], **hollow)
        bottom = f"RMSE 0 or under {PERCENT_FLOOR:g}% of n" if relative else "RMSE exactly 0"
        handles[f"hollow, bottom edge: {bottom}"] = Line2D([], [], **hollow)
        handles[f"shaded: n < {MIN_JUDGED_N}, not judged"] = Patch(color=REGION_FILL)
        what = "relative RMSE" if relative else "RMSE"
        ax.set_xlabel("cardinality: true number of distinct values, n")
        ax.set_ylabel("relative RMSE, % of n (95% CI)" if relative else "RMSE of the estimate (95% CI)")
        ax.set_title(f"{spec.label}: {what} against cardinality at each memory budget (64 hash seeds per point)",
                     loc="left", color=INK, fontsize=11, pad=72)
        ax.legend(list(handles.values()), list(handles.keys()), loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        fig.tight_layout()
        return save(fig, stem)


def plot_relative_rmse_by_budget(rows: list[dict], spec: SketchSpec, stem: Path) -> list[Path]:
    return plot_rmse_by_budget(rows, spec, stem, relative=True)


def plot_memory_used(rows: list[dict], spec: SketchSpec, stem: Path) -> list[Path]:
    """Bits of sketch state actually used against n at each budget (one observations stream per n),
    with each budget as a dotted line. Only for sketches whose memory varies with n (HyperLogLog++,
    whose sparse list grows until it switches to dense registers)."""
    series = {bits: by_n(rows, spec.name, bits) for bits in BUDGETS}
    if len({row["bits_used"] for points in series.values() for row in points}) <= len(BUDGETS):
        return []
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=64)
        handles: dict[str, object] = {}
        for bits in BUDGETS:
            colour = BUDGET_COLOURS[bits]
            points = series[bits]
            ax.plot([row["true_n"] for row in points], [row["bits_used"] for row in points], "o-", color=colour,
                    markersize=4.5, linewidth=1.2, markeredgecolor=SURFACE, markeredgewidth=0.6)
            ax.axhline(bits, color=colour, linewidth=1, linestyle=(0, (1, 2)))
            handles[f"{bits // BITS_PER_KIB} KiB budget (dotted: {bits:,} bits)"] = Line2D([], [], color=colour, marker="o", linewidth=1.2)
        ax.set_xlabel("cardinality: true number of distinct values, n")
        ax.set_ylabel("bits of sketch state used")
        ax.set_title(f"{spec.label}: memory actually used against cardinality", loc="left", color=INK, fontsize=11, pad=36)
        ax.legend(list(handles.values()), list(handles.keys()), loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        fig.tight_layout()
        return save(fig, stem)


def saturation_label(hashes: int) -> str:
    return "predicted, exp(-m e^(-n/m))" if hashes == 1 else f"predicted, exp(-m e^(-{hashes}n/m))"


def plot_saturation(rows: list[dict], spec: SketchSpec, stem: Path) -> list[Path]:
    """Share of seeds that saturate against n, one ramp colour per budget, with the prediction.
    Only for sketches whose bitmap can fill up."""
    if not spec.hashes:
        return []
    observations = [row for row in rows if row["sketch"] == spec.name and row["kind"] == "observations"]
    near = {
        bits: sorted((row for row in observations if row["bits"] == bits
                      and SATURATION_WINDOW[0] <= row["true_n"] / saturation_n(bits, spec.hashes) <= SATURATION_WINDOW[1]),
                     key=lambda row: row["true_n"])
        for bits in BUDGETS
    }
    if not any(near.values()):
        return []
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        for bits in BUDGETS:
            colour = BUDGET_COLOURS[bits]
            centre = saturation_n(bits, spec.hashes)
            ns = log_space(SATURATION_WINDOW[0] * centre, SATURATION_WINDOW[1] * centre)
            ax.plot(ns, [saturation_probability(n, bits, spec.hashes) for n in ns], color=colour, linewidth=1.5)
            ax.plot([row["true_n"] for row in near[bits]], [row["saturated"] / row["trials"] for row in near[bits]],
                    "o", color=colour, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1)
            ax.annotate(f"{bits // BITS_PER_KIB} KiB", (centre, saturation_probability(centre, bits, spec.hashes)),
                        xytext=(-8, 6), textcoords="offset points", ha="right", color=INK_SECONDARY)
        ax.set_title(f"{spec.label}: share of the 64 hash seeds where every bit is set", loc="left", color=INK, fontsize=11, pad=30)
        ax.set_xlabel("true number of distinct values, n")
        ax.set_ylabel("share of seeds saturated")
        ax.legend([Line2D([], [], color=INK_SECONDARY, linewidth=1.5), Line2D([], [], linestyle="none", marker="o", color=INK_SECONDARY)],
                  [saturation_label(spec.hashes), "measured, observations streams"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        fig.tight_layout()
        return save(fig, stem)


def rmse_ratios(rows: list[dict], spec: SketchSpec) -> dict[str, list[float]]:
    """Measured RMSE / predicted RMSE of every judged group with n >= MIN_JUDGED_N, by distribution
    of repeats (alphabetical), with the SQL streams last."""
    ratios: dict[str, list[float]] = defaultdict(list)
    for group in judged(rows, spec, SHOWN_KINDS).values():
        for row in group:
            if row["true_n"] >= MIN_JUDGED_N and row["rmse_confirmed"] is not None and has_prediction(row):
                ratios[distribution_of(row["stream"])].append(row["rmse"] / row["predicted_rmse"])
    order = sorted(name for name in ratios if name != SQL_LABEL) + ([SQL_LABEL] if SQL_LABEL in ratios else [])
    return {name: ratios[name] for name in order}


def plot_by_distribution(rows: list[dict], spec: SketchSpec, stem: Path) -> list[Path]:
    """One row per distribution of repeats: each dot is one stream and budget's measured RMSE
    divided by the predicted RMSE. The shaded band is where 95% of dots fall if the theory holds."""
    ratios = rmse_ratios(rows, spec)
    if not ratios:
        return []
    with plt.rc_context(FIGURE_STYLE):
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
        ax.set_title(f"{spec.label} on every distribution of repeats (n >= {MIN_JUDGED_N}, all budgets)",
                     loc="left", color=INK, fontsize=11, pad=24)
        ax.legend([Patch(color=REGION_FILL), Line2D([], [], color=INK_SECONDARY, linewidth=1)],
                  ["shaded: where 95% of dots fall if the theory holds", "measured = predicted"],
                  loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        fig.tight_layout()
        return save(fig, stem)


# --- Key numbers ----------------------------------------------------------------------------

def judged_for_key_numbers(rows: list[dict], spec: SketchSpec) -> list[dict]:
    return [row for row in rows if row["sketch"] == spec.name and row["true_n"] >= MIN_JUDGED_N and row["rmse_confirmed"] is not None]


def confirmation_line(rows: list[dict], label: str) -> str:
    rmse = sum(bool(row["rmse_confirmed"]) for row in rows)
    bias = sum(bool(row["bias_confirmed"]) for row in rows)
    return f"| {label} | {rmse} of {len(rows)} | {bias} of {len(rows)} |"


def first_saturated_n(rows: list[dict], sketch: str, bits: int) -> int | None:
    saturated = [row["true_n"] for row in rows if row["sketch"] == sketch and row["bits"] == bits and row["saturated"] > 0]
    return min(saturated) if saturated else None


def merge_lines(rows: list[dict], spec: SketchSpec) -> list[str]:
    if not spec.merge:
        return ["- **Merge:** not benchmarked for this sketch."]
    merges = [row for row in rows if row["sketch"] == spec.merge and row["merged_equals_direct"] is not None]
    equal = sum(row["merged_equals_direct"] for row in merges)
    return [f"- **Merge:** {spec.merge_phrase} in **{equal:,} of {sum(row['trials'] for row in merges):,}** trials."]


def exact_lines(rows: list[dict], spec: SketchSpec) -> list[str]:
    if not spec.exact_below:
        return []
    small = [row for row in rows if row["sketch"] == spec.name and 0 < row["true_n"] < spec.exact_below(row["bits"])]
    exact = sum(row["rmse"] == 0 for row in small)
    return [f"- **Exact below k:** every seed gave exactly n in **{exact} of {len(small)}** stream and budget "
            f"groups with n < k."]


def key_numbers(rows: list[dict], spec: SketchSpec) -> str:
    judged_rows = judged_for_key_numbers(rows, spec)
    lines = [f"# {spec.label}: key numbers", "", *merge_lines(rows, spec), *exact_lines(rows, spec)]
    lines += ["", f"## Predictions inside the measured 95% interval (n >= {MIN_JUDGED_N}, where the theory is judged)", "",
              "| Streams, budget | RMSE | Bias |", "|---|---|---|"]
    for kind, style in KIND_STYLE.items():
        for bits in BUDGETS:
            group = [row for row in judged_rows if row["kind"] == kind and row["bits"] == bits]
            lines.append(confirmation_line(group, f"{style['label']}, {bits // BITS_PER_KIB} KiB"))
    ratios = rmse_ratios(rows, spec)
    lines += ["", "## By distribution of repeats (all budgets)", "",
              "| Distribution | RMSE | Bias | Median measured / predicted RMSE |", "|---|---|---|---|"]
    for name, values in ratios.items():
        group = [row for row in judged_rows if distribution_of(row["stream"]) == name]
        lines.append(f"{confirmation_line(group, name)} {statistics.median(values):.3f} |")
    if spec.hashes:
        lines += ["", "## Saturation", "", "| Budget | Predicted saturation n | Smallest n with a saturated seed |", "|---|---|---|"]
        for bits in BUDGETS:
            first = first_saturated_n(rows, spec.name, bits)
            lines.append(f"| {bits // BITS_PER_KIB} KiB | {saturation_n(bits, spec.hashes):,.0f} | {f'{first:,}' if first else 'none'} |")
    return "\n".join(lines) + "\n"


# --- Command line ---------------------------------------------------------------------------

def draw_sketch(rows: list[dict], spec: SketchSpec, output: Path) -> list[Path]:
    """Every figure and the key numbers of one sketch, in output/, named <short>_<figure>."""
    written = [
        *plot_rmse_vs_theory(rows, spec, output / f"{spec.short}_rmse_vs_theory"),
        *plot_rmse_by_budget(rows, spec, output / f"{spec.short}_rmse_by_budget", relative=False),
        *plot_relative_rmse_by_budget(rows, spec, output / f"{spec.short}_relative_rmse_by_budget"),
        *plot_memory_used(rows, spec, output / f"{spec.short}_memory_used"),
        *plot_bias(rows, spec, output / f"{spec.short}_bias"),
        *plot_saturation(rows, spec, output / f"{spec.short}_saturation"),
        *plot_by_distribution(rows, spec, output / f"{spec.short}_by_distribution"),
    ]
    numbers = output / f"{spec.short}_key_numbers.md"
    numbers.parent.mkdir(parents=True, exist_ok=True)
    numbers.write_text(key_numbers(rows, spec), encoding="utf-8")
    return written + [numbers]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="summary.csv from results_analyser.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for the figures (one folder per sketch)")
    parser.add_argument("--sketch", action="append", choices=list(SPECS), help="draw only this sketch (repeatable; default: all)")
    args = parser.parse_args()

    if not args.summary.is_file():
        print(f"no summary at {args.summary} (run python/results_analyser.py first)", file=sys.stderr)
        return 1
    specs = [SPECS[name] for name in (args.sketch or SPECS)]
    names = tuple(name for spec in specs for name in (spec.name, spec.merge) if name)
    rows = load_rows(args.summary, names, tuple(KIND_STYLE))
    for spec in specs:
        for path in draw_sketch(rows, spec, args.output / spec.short):
            print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
