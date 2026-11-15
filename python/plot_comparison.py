"""Draw the five sketches against each other for the report and poster.

Run from the repository root after results_analyser.py:

    python3 python/plot_comparison.py --summary data/analysis-all/summary.csv

Every observations stream with m distinct values holds the same values 0..m-1 whatever its
distribution of repeats, so every sketch gives identical results on them; the figures over n use
one of them per n. Figures (PDF for the report, PNG for a quick look) in
data/analysis-all/figures/comparison/:

1. relative_rmse_vs_n: relative RMSE (% of n) of every sketch against n, one panel per budget.
   Hollow markers on the top edge: a seed saturated, so there is no estimate. On the bottom
   edge: an RMSE of exactly 0.
2. relative_bias_vs_n: the same for the bias.
3. equal_memory: relative RMSE against the budget at n = 1,000, 10,000, 100,000 and 1,000,000,
   the proposal's question of which sketch is most accurate under the same memory.
4. theory_confirmed: share of groups (n >= 100, where the theory is judged) whose measured 95%
   interval contains the predicted RMSE, by sketch and budget.
5. theory_ratio: measured / predicted RMSE of every judged group, by sketch and kind of stream.
6. merge: share of seeds whose merged sketch of two disjoint halves gives the same estimate as
   one built over the whole stream, against n.
7. time_per_value: nanoseconds per value, by sketch and budget, on the distribution
   generator's streams of 1,000,000 or more values.
8. time_vs_length: nanoseconds per value against the number of values in the stream, one panel
   per budget (start-up cost on short streams, the steady rate on long ones).
9. speed_vs_accuracy: each sketch at each budget as one point, nanoseconds per value against
   relative RMSE at n = 100,000.
10. rmse_vs_n_16kib: absolute RMSE of every sketch against n on one axis at the largest budget,
    16 KiB. Hollow markers on the top edge: a seed saturated; on the bottom edge: RMSE exactly 0.

comparison_key_numbers.md has the tables the text quotes: memory used, relative RMSE at fixed n,
theory confirmed, saturation, merge exactness and speed.
"""
import argparse
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only: WSL and CI have no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from plot_results import AXIS, BITS_PER_KIB, BUDGETS, INK, INK_MUTED, INK_SECONDARY, SURFACE, save  # noqa: E402
from plot_sketches import (  # noqa: E402, F401 (by_n, state and the percent limits are used by the tests)
    BUDGET_COLOURS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUMMARY,
    EDGE_STEP,
    FIGURE_STYLE,
    KIND_STYLE,
    MARK_DASH,
    MIN_JUDGED_N,
    PERCENT_CEILING,
    PERCENT_FLOOR,
    RATIO_BAND,
    REGION_FILL,
    SPECS,
    absolute_edge,
    by_n,
    edge,
    first_saturated_n,
    has_prediction,
    load_rows,
    percent_axis,
    percent_of_n,
    state,
)
from results_analyser import MERGE_SUFFIX, hyperloglog_registers  # noqa: E402

N_TARGETS = (1_000, 10_000, 100_000, 1_000_000)  # n of the equal-memory panels and tables
MIN_TIMED_VALUES = 1_000_000  # long enough that the time per value is not start-up cost
SPEED_ACCURACY_N = 100_000  # the n whose relative RMSE the speed-against-accuracy figure uses
NSAMPLE = re.compile(r"_nsample(\d+)")
HLLPP_DENSE_SWITCH = 0.2  # HyperLogLog++ leaves sparse mode near 0.2 r (see results_analyser)
LARGEST_BUDGET = BUDGETS[-1]  # 16 KiB: the budget of the one-axis absolute RMSE figure

# Five series on one plot fail the dataviz palette's all-pairs check past three hues, so hue
# marks the family (categorical slots 1-3, validated all-pairs: bitmaps blue, order statistic
# orange, registers aqua) and marker and dash tell the members of a family apart. Aqua is under
# 3:1 against the surface, so every series also has a legend entry and its own marker.
BITMAP, ORDER_STATISTIC, REGISTERS = "#2a78d6", "#eb6834", "#1baf7a"
SKETCH_STYLE = {
    "LinearCounting": {"color": BITMAP, "marker": "o", "linestyle": "-", "label": "Linear Counting"},
    "BloomFilter": {"color": BITMAP, "marker": "s", "linestyle": "--", "label": "Bloom filter"},
    "KMinVals": {"color": ORDER_STATISTIC, "marker": "D", "linestyle": "-", "label": "K-Minimum Values"},
    "HyperLogLog": {"color": REGISTERS, "marker": "^", "linestyle": "-", "label": "HyperLogLog"},
    "HyperLogLogPlusPlus": {"color": REGISTERS, "marker": "v", "linestyle": "--", "label": "HyperLogLog++"},
}
SKETCHES = tuple(SKETCH_STYLE)
RATIO_TICKS = (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2)
TIME_DODGE = 0.06  # sketches at the same budget sit 2^(±0.06 i) apart so their ranges do not overlap
TWIN, TWIN_OF = "HyperLogLogPlusPlus", "HyperLogLog"  # HLL++ is exactly HLL once dense and past 5 r
TWIN_DODGE = 1.1  # where HLL++'s value equals HLL's, its point sits this factor to the right
TWIN_LABEL = "HyperLogLog++ = HyperLogLog here (drawn just to the right)"
EDGE_TEXT_ROOM = 0.2  # share of the log x range the edge text "saturated" needs beside a marker
SATURATED_LABEL = f"hollow, top edge: a seed saturated (no estimate), or RMSE over {PERCENT_CEILING:g}% of n"
EXACT_LABEL = f"hollow, bottom edge: RMSE 0 or under {PERCENT_FLOOR:g}% of n (near exact)"


# --- Selecting ------------------------------------------------------------------------------

def at_n(rows: list[dict], sketch: str, bits: int, n: int) -> dict | None:
    return next((row for row in by_n(rows, sketch, bits) if row["true_n"] == n), None)


def values_per_stream(stream: str) -> int | None:
    """The number of values in a distribution generator stream, from its name (..._nsample<N>...)."""
    found = NSAMPLE.search(stream)
    return int(found.group(1)) if found else None


def judged_groups(rows: list[dict], sketch: str, bits: int | None = None) -> list[dict]:
    """The sketch's groups that count towards theory confirmed: n >= MIN_JUDGED_N, judged."""
    return [row for row in rows if row["sketch"] == sketch and row["true_n"] >= MIN_JUDGED_N
            and row["rmse_confirmed"] is not None and (bits is None or row["bits"] == bits)]


def times_by_length(rows: list[dict], sketch: str, bits: int) -> dict[int, list[float]]:
    """Nanoseconds per value on every distribution generator stream, by the stream's number of values."""
    by_length: dict[int, list[float]] = {}
    for row in rows:
        length = values_per_stream(row["stream"]) if row["kind"] == "nsample" else None
        if row["sketch"] == sketch and row["bits"] == bits and length and row["mean_time_ms"]:
            by_length.setdefault(length, []).append(row["mean_time_ms"] * 1e6 / length)
    return dict(sorted(by_length.items()))


def nanoseconds_per_value(rows: list[dict], sketch: str, bits: int) -> list[float]:
    """Time per value on every distribution generator stream of at least MIN_TIMED_VALUES values."""
    return [t for length, times in times_by_length(rows, sketch, bits).items() if length >= MIN_TIMED_VALUES for t in times]


def merge_shares(rows: list[dict], sketch: str, bits: int) -> list[tuple[int, float]]:
    """(n, % of seeds whose merge equals the direct sketch) for the sketch's merge benchmark."""
    return sorted((row["true_n"], 100 * row["merged_equals_direct"] / row["trials"]) for row in rows
                  if row["sketch"] == sketch + MERGE_SUFFIX and row["bits"] == bits
                  and row["merged_equals_direct"] is not None and row["true_n"] > 0)


def measured_percents(rows: list[dict | None]) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """RMSE, interval low and high as % of n for the rows drawn as points (None for the others)."""
    kept = [row if row and edge(row) is None else None for row in rows]
    return ([percent_of_n(row["rmse"], row) if row else None for row in kept],
            [percent_of_n(row["rmse_low"], row) if row else None for row in kept],
            [percent_of_n(row["rmse_high"], row) if row else None for row in kept])


def absolute_values(rows: list[dict]) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """RMSE, interval low and high in distinct values for the rows drawn as points (None for the
    rows that go on an edge: saturated or exactly 0)."""
    kept = [row if absolute_edge(row) is None else None for row in rows]
    return ([row["rmse"] if row else None for row in kept],
            [row["rmse_low"] if row else None for row in kept],
            [row["rmse_high"] if row else None for row in kept])


# --- Drawing helpers ------------------------------------------------------------------------

def sketch_handles(sketches: tuple[str, ...] = SKETCHES) -> dict[str, object]:
    handles: dict[str, object] = {}
    for sketch in sketches:
        style = SKETCH_STYLE[sketch]
        handles[style["label"]] = Line2D([], [], color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                                         linewidth=1.5, markersize=6)
    return handles


def edge_handles(exact: bool = True) -> dict[str, object]:
    hollow = {"linestyle": "none", "marker": "o", "markersize": 6, "color": INK_MUTED, "markerfacecolor": SURFACE}
    handles: dict[str, object] = {SATURATED_LABEL: Line2D([], [], **hollow)}
    if exact:
        handles[EXACT_LABEL] = Line2D([], [], **hollow)
    return handles


def draw_edges(ax: plt.Axes, sketch: str, saturated_x: list[float], exact_x: list[float], index: int) -> None:
    """Hollow markers on the top edge (saturated) and bottom edge (exact), one row per sketch.
    Call label_top_edges once the axis is drawn to name the top-edge rows."""
    style = SKETCH_STYLE[sketch]
    transform = ax.get_xaxis_transform()
    for xs, y in ((saturated_x, 1 - EDGE_STEP * (index + 0.5)), (exact_x, EDGE_STEP * (index + 0.5))):
        if xs:
            ax.plot(xs, [y] * len(xs), linestyle="none", marker=style["marker"], markersize=5, color=style["color"],
                    markerfacecolor=SURFACE, markeredgewidth=1.2, transform=transform, clip_on=False)


def top_edge_text(rows: list[dict | None]) -> str:
    """What the top-edge markers of these rows mean: saturated, off the scale, or both."""
    saturated = {state(row) == "saturated" for row in rows if row is not None}
    return {frozenset({True}): "saturated", frozenset({False}): "off scale"}.get(frozenset(saturated), "saturated / off scale")


def edge_text_place(xs: list[float], low: float, high: float, log: bool = True) -> tuple[float, str]:
    """Where the text for a row of top-edge markers at xs goes on an x axis from low to high (log or
    linear): right of the last marker, else left of the first, else in the widest gap between
    markers. Returns (x, horizontal alignment)."""
    to_axis, from_axis = (math.log, math.exp) if log else (float, float)
    points = sorted(to_axis(x) for x in xs)
    start, end = to_axis(low), to_axis(high)
    room = EDGE_TEXT_ROOM * (end - start)
    if points[-1] + room <= end:
        return max(xs), "left"
    if points[0] - room >= start:
        return min(xs), "right"
    gaps = [(b - a, (a + b) / 2) for a, b in zip(points, points[1:])] or [(0.0, points[0])]
    return from_axis(max(gaps)[1]), "center"


def label_top_edges(ax: plt.Axes, rows: list[tuple[int, list[float], str]]) -> None:
    """Name each sketch's row of top-edge markers ("saturated" or "off scale") in muted ink beside
    the markers. rows holds (sketch index, marker xs, text); call after everything is drawn."""
    low, high = ax.get_xlim()
    transform = ax.get_xaxis_transform()
    for index, xs, text in rows:
        if not xs:
            continue
        x, align = edge_text_place(xs, low, high, log=ax.get_xscale() == "log")
        offset = {"left": 5, "right": -5, "center": 0}[align]
        ax.annotate(text, xy=(x, 1 - EDGE_STEP * (index + 0.5)), xycoords=transform, xytext=(offset, 0),
                    textcoords="offset points", ha=align, va="center", fontsize=6.5, color=INK_MUTED,
                    annotation_clip=False)


def same_value(row: dict | None, twin: dict | None) -> bool:
    """True when two rows show the same thing: equal RMSE (exact equality, as HLL++ = HLL past 5 r)
    or both saturated."""
    if row is None or twin is None:
        return False
    if state(row) == "saturated" or state(twin) == "saturated":
        return state(row) == state(twin)
    return row["rmse"] == twin["rmse"]


def dodged(sketch: str, xs: list[float], rows: list[dict | None], twins: list[dict | None]) -> tuple[list[float], bool]:
    """xs, with HyperLogLog++'s points moved right by TWIN_DODGE where they equal HyperLogLog's
    (twins lines up with rows), so the two series do not hide each other. Also says whether any moved."""
    if sketch != TWIN:
        return list(xs), False
    moved = [x * TWIN_DODGE if same_value(row, twin) else x for x, row, twin in zip(xs, rows, twins)]
    return moved, moved != list(xs)


def twin_handles(any_dodged: bool) -> dict[str, object]:
    """The legend note for dodged HyperLogLog++ points, when any were dodged."""
    if not any_dodged:
        return {}
    style = SKETCH_STYLE[TWIN]
    return {TWIN_LABEL: Line2D([], [], color=style["color"], marker=style["marker"], linestyle="none", markersize=6)}


def twins_by_n(rows: list[dict], points: list[dict], bits: int) -> list[dict | None]:
    """HyperLogLog's row at each point's n (None where it has none)."""
    hll = {row["true_n"]: row for row in by_n(rows, TWIN_OF, bits)}
    return [hll.get(row["true_n"]) for row in points]


def draw_series(ax: plt.Axes, sketch: str, xs: list[float], values: tuple[list, list, list]) -> None:
    """A line through the measured points (broken where a value is None) with their 95% intervals.
    On a log axis an interval reaching 0 is drawn down to a tenth of the value."""
    style = SKETCH_STYLE[sketch]
    ys, lows, highs = values
    ax.plot(xs, [math.nan if y is None else y for y in ys], color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markersize=4.5, linewidth=1.5, markeredgecolor=SURFACE, markeredgewidth=0.6)
    kept = [(x, y, low, high) for x, y, low, high in zip(xs, ys, lows, highs) if None not in (y, low, high)]
    if kept:
        down = [max(y - max(low, y / 10), 0) for _, y, low, _ in kept]
        up = [high - y for _, y, _, high in kept]
        ax.errorbar([k[0] for k in kept], [k[1] for k in kept], yerr=[down, up], fmt="none", ecolor=style["color"],
                    elinewidth=0.8, capsize=0, alpha=0.6)


def edge_rows(xs: list[float], rows: list[dict | None], where: str) -> list[float]:
    return [x for x, row in zip(xs, rows) if edge(row) == where]


def budget_grid(title: str, ylabel: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), sharex=True, sharey=True)
    for ax, bits in zip(axes.flat, BUDGETS):
        ax.set_title(f"{bits // BITS_PER_KIB} KiB ({bits:,} bits)", loc="left")
        ax.set_xscale("log")
    fig.suptitle(title, x=0.06, ha="left", color=INK, fontsize=11)
    fig.supxlabel("true number of distinct values, n", color=INK_SECONDARY)
    fig.supylabel(ylabel, color=INK_SECONDARY)
    return fig, list(axes.flat)


def finish(fig: plt.Figure, handles: dict[str, object], stem: Path, ncol: int = 3, top: float = 0.87) -> list[Path]:
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper left", ncol=ncol, bbox_to_anchor=(0.05, 0.955))
    fig.tight_layout(rect=(0, 0, 1, top))
    return save(fig, stem)


def has_observations(rows: list[dict]) -> bool:
    return any(by_n(rows, sketch, bits) for sketch in SKETCHES for bits in BUDGETS)


# --- Figures --------------------------------------------------------------------------------

def plot_relative_rmse_vs_n(rows: list[dict], stem: Path) -> list[Path]:
    """Relative RMSE against n, every sketch, one panel per budget."""
    if not has_observations(rows):
        return []
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = budget_grid("Relative RMSE at equal memory: every sketch on the same streams (64 hash seeds per point)",
                                "relative RMSE, % of n (95% CI)")
        any_dodged = False
        for ax, bits in zip(axes, BUDGETS):
            percent_axis(ax)
            labels = []
            for index, sketch in enumerate(SKETCHES):
                points = by_n(rows, sketch, bits)
                xs, moved = dodged(sketch, [row["true_n"] for row in points], points, twins_by_n(rows, points, bits))
                any_dodged |= moved
                draw_series(ax, sketch, xs, measured_percents(points))
                top = edge_rows(xs, points, "top")
                draw_edges(ax, sketch, top, edge_rows(xs, points, "bottom"), index)
                labels.append((index, top, top_edge_text([row for row in points if edge(row) == "top"])))
            label_top_edges(ax, labels)
        return finish(fig, {**sketch_handles(), **edge_handles(), **twin_handles(any_dodged)}, stem, top=0.88)


def plot_relative_bias_vs_n(rows: list[dict], stem: Path) -> list[Path]:
    """Relative bias against n, every sketch, one panel per budget, on a symmetric log axis."""
    if not has_observations(rows):
        return []
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = budget_grid("Relative bias at equal memory: mean(estimate - n) / n (64 hash seeds per point)",
                                "relative bias, % of n")
        for ax, bits in zip(axes, BUDGETS):
            ax.set_yscale("symlog", linthresh=0.1)
            ax.axhline(0, color=AXIS, linewidth=1)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
            for index, sketch in enumerate(SKETCHES):
                style = SKETCH_STYLE[sketch]
                points = by_n(rows, sketch, bits)
                bias = [percent_of_n(row["bias"], row) if state(row) != "saturated" else None for row in points]
                ax.plot([row["true_n"] for row in points], [math.nan if b is None else b for b in bias],
                        color=style["color"], linestyle=style["linestyle"], marker=style["marker"], markersize=4.5,
                        linewidth=1.5, markeredgecolor=SURFACE, markeredgewidth=0.6)
                draw_edges(ax, sketch, [row["true_n"] for row in points if state(row) == "saturated"], [], index)
        return finish(fig, {**sketch_handles(), **edge_handles(exact=False)}, stem)


def plot_equal_memory(rows: list[dict], stem: Path) -> list[Path]:
    """Relative RMSE against the budget, one panel per n in N_TARGETS."""
    targets = [n for n in N_TARGETS if any(at_n(rows, s, b, n) for s in SKETCHES for b in BUDGETS)]
    if not targets:
        return []
    kib = [bits / BITS_PER_KIB for bits in BUDGETS]
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, len(targets), figsize=(2.6 * len(targets) + 1.2, 4.6), sharey=True, squeeze=False)
        any_dodged = False
        for ax, n in zip(axes[0], targets):
            ax.set_xscale("log", base=2)
            percent_axis(ax)
            ax.set_xticks(kib, [f"{k:g}" for k in kib])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(kib[0] / 1.4, kib[-1] * 1.4)
            ax.set_title(f"n = {n:,}", loc="left")
            labels = []
            for index, sketch in enumerate(SKETCHES):
                found = [at_n(rows, sketch, bits, n) for bits in BUDGETS]
                xs, moved = dodged(sketch, kib, found, [at_n(rows, TWIN_OF, bits, n) for bits in BUDGETS])
                any_dodged |= moved
                draw_series(ax, sketch, xs, measured_percents(found))
                top = edge_rows(xs, found, "top")
                draw_edges(ax, sketch, top, edge_rows(xs, found, "bottom"), index)
                labels.append((index, top, top_edge_text([row for row in found if edge(row) == "top"])))
            label_top_edges(ax, labels)
        axes[0][0].set_ylabel("relative RMSE, % of n (95% CI)")
        fig.supxlabel("memory budget (KiB)", color=INK_SECONDARY)
        fig.suptitle("Which sketch is most accurate under the same memory? (64 hash seeds per point)",
                     x=0.06, ha="left", color=INK, fontsize=11)
        return finish(fig, {**sketch_handles(), **edge_handles(), **twin_handles(any_dodged)}, stem, ncol=3, top=0.83)


def plot_rmse_vs_n_at_budget(rows: list[dict], stem: Path, bits: int = LARGEST_BUDGET) -> list[Path]:
    """Absolute RMSE against n for every sketch on one axis at one budget (95% CI, one
    observations stream per n). Saturated groups sit on the top edge, exact ones on the bottom."""
    series = {sketch: by_n(rows, sketch, bits) for sketch in SKETCHES}
    if not any(series.values()):
        return []
    hollow = {"linestyle": "none", "marker": "o", "markersize": 6, "color": INK_MUTED, "markerfacecolor": SURFACE}
    edges = {"hollow, top edge: a seed saturated (no estimate)": Line2D([], [], **hollow),
             "hollow, bottom edge: RMSE exactly 0 (every seed exact)": Line2D([], [], **hollow)}
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        ax.set_xscale("log")
        ax.set_yscale("log")
        any_dodged, labels = False, []
        for index, (sketch, points) in enumerate(series.items()):
            xs, moved = dodged(sketch, [row["true_n"] for row in points], points, twins_by_n(rows, points, bits))
            any_dodged |= moved
            draw_series(ax, sketch, xs, absolute_values(points))
            top = [x for x, row in zip(xs, points) if absolute_edge(row) == "top"]
            draw_edges(ax, sketch, top, [x for x, row in zip(xs, points) if absolute_edge(row) == "bottom"], index)
            labels.append((index, top, "saturated"))
        label_top_edges(ax, labels)
        edges.update(twin_handles(any_dodged))
        ax.set_xlabel("true number of distinct values, n")
        ax.set_ylabel("RMSE of the estimate, distinct values (95% CI)")
        fig.suptitle(f"RMSE against cardinality, every sketch at {bits // BITS_PER_KIB} KiB ({bits:,} bits; "
                     "64 hash seeds per point)", x=0.06, ha="left", color=INK, fontsize=11)
        return finish(fig, {**sketch_handles(), **edges}, stem, ncol=3, top=0.9)


def confirmed_share(group: list[dict]) -> float:
    """% of the groups whose prediction is confirmed; NaN (no bar) when there are none."""
    return 100 * sum(bool(row["rmse_confirmed"]) for row in group) / len(group) if group else math.nan


def one_per_observations_n(rows: list[dict]) -> list[dict]:
    """The rows with each observations stream's (sketch, budget, n) kept once: those streams hold
    identical values, so their copies repeat one result."""
    seen: set[tuple[str, int, int]] = set()
    kept = []
    for row in rows:
        key = (row["sketch"], row["bits"], row["true_n"])
        if row["kind"] == "observations":
            if key in seen:
                continue
            seen.add(key)
        kept.append(row)
    return kept


def plot_theory_confirmed(rows: list[dict], stem: Path) -> list[Path]:
    """Share of judged groups whose measured 95% interval holds the predicted RMSE, by sketch and budget."""
    sketches = [s for s in SKETCHES if judged_groups(rows, s)]
    if not sketches:
        return []
    width = 0.2
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for offset, bits in enumerate(BUDGETS):
            xs = [i + (offset - 1.5) * width for i in range(len(sketches))]
            ax.bar(xs, [confirmed_share(judged_groups(rows, s, bits)) for s in sketches], width=width,
                   color=BUDGET_COLOURS[bits], edgecolor=SURFACE, linewidth=2, label=f"{bits // BITS_PER_KIB} KiB")
        for i, sketch in enumerate(sketches):
            group = judged_groups(rows, sketch)
            ax.annotate(f"{confirmed_share(group):.0f}% of {len(group):,}", (i, 100), xytext=(0, 4),
                        textcoords="offset points", ha="center", color=INK_SECONDARY)
        ax.set_xticks(range(len(sketches)), [SKETCH_STYLE[s]["label"] for s in sketches])
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_ylabel("predicted RMSE inside the measured 95% CI")
        ax.set_title(f"How often the theory is confirmed (stream and budget groups, n >= {MIN_JUDGED_N}, where judged)",
                     loc="left", color=INK, fontsize=11, pad=28)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
        fig.tight_layout()
        return save(fig, stem)


def plot_theory_ratio(rows: list[dict], stem: Path) -> list[Path]:
    """Measured / predicted RMSE of every judged group, one row per sketch, coloured by kind of stream."""
    sketches = [s for s in SKETCHES if judged_groups(rows, s)]
    if not sketches:
        return []
    offsets = {"observations": -0.22, "nsample": 0.0, "sql": 0.22}
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.85 * len(sketches) + 1.8))
        ax.set_xscale("log")
        ax.axvspan(1 - RATIO_BAND, 1 + RATIO_BAND, color=REGION_FILL, linewidth=0, zorder=0)
        ax.axvline(1, color=INK_SECONDARY, linewidth=1)
        for y, sketch in enumerate(sketches):
            for kind, offset in offsets.items():
                ratios = [row["rmse"] / row["predicted_rmse"] for row in judged_groups(rows, sketch)
                          if row["kind"] == kind and row["rmse"] and has_prediction(row)]
                style = KIND_STYLE[kind]
                ax.plot(ratios, [y + offset] * len(ratios), style["marker"], color=style["color"], markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=0.6, alpha=0.6)
        ax.set_yticks(range(len(sketches)), [SKETCH_STYLE[s]["label"] for s in sketches])
        ax.invert_yaxis()
        ax.set_xticks(RATIO_TICKS, [f"{tick:g}" for tick in RATIO_TICKS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("measured RMSE / predicted RMSE (1 = exactly as predicted)")
        ax.set_title(f"Measured against predicted error, every judged group (n >= {MIN_JUDGED_N}; exact groups, RMSE 0, not drawn)",
                     loc="left", color=INK, fontsize=11, pad=40)
        handles: dict[str, object] = {
            KIND_STYLE[k]["label"]: Line2D([], [], linestyle="none", marker=KIND_STYLE[k]["marker"], color=KIND_STYLE[k]["color"])
            for k in offsets
        }
        handles["shaded: where 95% of groups fall if the theory holds"] = Patch(color=REGION_FILL)
        ax.legend(list(handles.values()), list(handles.keys()), loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        fig.tight_layout()
        return save(fig, stem)


def plot_merge(rows: list[dict], stem: Path) -> list[Path]:
    """Share of seeds whose merge equals the direct sketch, against n, every sketch with a merge benchmark."""
    merged = tuple(s for s in SKETCHES if any(merge_shares(rows, s, bits) for bits in BUDGETS))
    if not merged:
        return []
    lowest = min(share for s in merged for bits in BUDGETS for _, share in merge_shares(rows, s, bits))
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = budget_grid("Merging two disjoint halves: share of seeds with the same estimate as one sketch of the whole stream",
                                "seeds where merged = direct")
        for ax, bits in zip(axes, BUDGETS):
            ax.set_ylim(min(lowest, 99) - 1, 100.6)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
            if "HyperLogLogPlusPlus" in merged:
                ax.axvline(HLLPP_DENSE_SWITCH * hyperloglog_registers(bits), color=INK_MUTED, linewidth=1, linestyle=MARK_DASH)
            for sketch in merged:
                style = SKETCH_STYLE[sketch]
                points = sorted(merge_shares(rows, sketch, bits), key=lambda point: -point[1])  # failures drawn last, on top
                ax.plot([n for n, _ in points], [share for _, share in points], linestyle="none", marker=style["marker"],
                        markersize=5, color=style["color"], markeredgecolor=SURFACE, markeredgewidth=0.6, alpha=0.8)
        handles = sketch_handles(merged)
        if "HyperLogLogPlusPlus" in merged:
            handles["dashed: n = 0.2 r, where HyperLogLog++ leaves sparse mode"] = Line2D(
                [], [], color=INK_MUTED, linewidth=1, linestyle=MARK_DASH)
        missing = [SKETCH_STYLE[s]["label"] for s in SKETCHES if s not in merged]
        if missing:
            handles[f"no merge benchmark: {', '.join(missing)}"] = Line2D([], [], linestyle="none")
        return finish(fig, handles, stem, ncol=2)


def quartiles(groups: dict[float, list[float]]) -> tuple[list[float], list[float], list[float], list[float]]:
    """(x, median, median - Q1, Q3 - median) for every x with at least two values."""
    xs, median, down, up = [], [], [], []
    for x, values in groups.items():
        if len(values) < 2:
            continue
        q1, q2, q3 = statistics.quantiles(values, n=4)
        xs.append(x)
        median.append(q2)
        down.append(q2 - q1)
        up.append(q3 - q2)
    return xs, median, down, up


def quartile_points(times_by_bits: dict[int, list[float]]) -> tuple[list[float], list[float], list[float], list[float]]:
    """(KiB, median, median - Q1, Q3 - median) for every budget with at least two timings."""
    return quartiles({bits / BITS_PER_KIB: times_by_bits.get(bits, []) for bits in BUDGETS})


def plot_time_per_value(rows: list[dict], stem: Path) -> list[Path]:
    """Median nanoseconds per value (with the interquartile range) against the budget."""
    timed = {s: {bits: nanoseconds_per_value(rows, s, bits) for bits in BUDGETS} for s in SKETCHES}
    timed = {s: by_bits for s, by_bits in timed.items() if any(len(t) >= 2 for t in by_bits.values())}
    if not timed:
        return []
    kib = [bits / BITS_PER_KIB for bits in BUDGETS]
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.set_xscale("log", base=2)
        ax.set_xticks(kib, [f"{k:g} KiB" for k in kib])
        ax.xaxis.set_minor_locator(NullLocator())
        for index, (sketch, by_bits) in enumerate(timed.items()):
            style = SKETCH_STYLE[sketch]
            xs, median, down, up = quartile_points(by_bits)
            dodge = 2 ** (TIME_DODGE * (index - (len(timed) - 1) / 2))
            ax.errorbar([x * dodge for x in xs], median, yerr=[down, up], color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
                        markersize=6, linewidth=1.5, elinewidth=1, capsize=0, label=style["label"])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("memory budget")
        ax.set_ylabel("nanoseconds per value (median, interquartile range)")
        ax.set_title(f"Update speed: one pass over streams of {MIN_TIMED_VALUES:,}+ values, hashing included",
                     loc="left", color=INK, fontsize=11, pad=28)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5)
        fig.tight_layout()
        return save(fig, stem)


def plot_time_vs_length(rows: list[dict], stem: Path) -> list[Path]:
    """Median nanoseconds per value (interquartile range) against the stream's number of values,
    one panel per budget: start-up cost on short streams, the steady rate on long ones."""
    timed = {s: {bits: times_by_length(rows, s, bits) for bits in BUDGETS} for s in SKETCHES}
    if not any(len(t) >= 2 for by_bits in timed.values() for by_length in by_bits.values() for t in by_length.values()):
        return []
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), sharex=True, sharey=True)
        top = 0.0
        for ax, bits in zip(axes.flat, BUDGETS):
            ax.set_title(f"{bits // BITS_PER_KIB} KiB ({bits:,} bits)", loc="left")
            ax.set_xscale("log")
            for index, sketch in enumerate(SKETCHES):
                style = SKETCH_STYLE[sketch]
                xs, median, down, up = quartiles(timed[sketch][bits])
                top = max([top, *(m + u for m, u in zip(median, up))])
                dodge = 2 ** (TIME_DODGE * 2 * (index - (len(SKETCHES) - 1) / 2))
                ax.errorbar([x * dodge for x in xs], median, yerr=[down, up], color=style["color"], linestyle=style["linestyle"],
                            marker=style["marker"], markersize=5, linewidth=1.5, elinewidth=0.8, capsize=0)
        axes.flat[0].set_ylim(0, top * 1.05)  # shared: every interquartile range fits
        fig.suptitle("Update speed against stream length (distribution generator streams, hashing included)",
                     x=0.06, ha="left", color=INK, fontsize=11)
        fig.supxlabel("values in the stream", color=INK_SECONDARY)
        fig.supylabel("nanoseconds per value (median, interquartile range)", color=INK_SECONDARY)
        return finish(fig, sketch_handles(), stem, top=0.9)


def plot_speed_vs_accuracy(rows: list[dict], stem: Path) -> list[Path]:
    """Each sketch at each budget as one point: median nanoseconds per value (streams of
    MIN_TIMED_VALUES+ values) against relative RMSE at n = SPEED_ACCURACY_N. Saturated points go on
    the top edge."""
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, len(BUDGETS), figsize=(12, 4.4), sharex=True, sharey=True)
        drawn, fastest_limit, pending = False, 0.0, []
        for ax, bits in zip(axes, BUDGETS):
            percent_axis(ax)
            ax.set_title(f"{bits // BITS_PER_KIB} KiB", loc="left")
            labels = []
            for index, sketch in enumerate(SKETCHES):
                times, found = nanoseconds_per_value(rows, sketch, bits), at_n(rows, sketch, bits, SPEED_ACCURACY_N)
                if not times or found is None:
                    continue
                drawn = True
                speed = statistics.median(times)
                fastest_limit = max(fastest_limit, speed)
                if edge(found) is None:
                    style = SKETCH_STYLE[sketch]
                    ax.plot([speed], [percent_of_n(found["rmse"], found)], linestyle="none", marker=style["marker"],
                            markersize=8, color=style["color"], markeredgecolor=SURFACE, markeredgewidth=1)
                else:
                    top = [speed] if edge(found) == "top" else []
                    draw_edges(ax, sketch, top, [speed] if edge(found) == "bottom" else [], index)
                    labels.append((index, top, top_edge_text([found])))
            pending.append((ax, labels))
        if not drawn:
            plt.close(fig)
            return []
        axes[0].set_xlim(0, fastest_limit * 1.12)  # shared: the slowest sketch fits in every panel
        for ax, labels in pending:
            label_top_edges(ax, labels)
        axes[0].set_ylabel(f"relative RMSE at n = {SPEED_ACCURACY_N:,}, % of n")
        fig.supxlabel(f"nanoseconds per value (median, streams of {MIN_TIMED_VALUES:,}+ values); lower left is better",
                      color=INK_SECONDARY)
        fig.suptitle("Speed against accuracy at equal memory", x=0.06, ha="left", color=INK, fontsize=11)
        return finish(fig, {**sketch_handles(), **edge_handles(exact=False)}, stem, ncol=3, top=0.84)


# --- Key numbers ----------------------------------------------------------------------------

def table(header: list[str], body: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(row) + " |" for row in body)]


def budget_header(first: str) -> list[str]:
    return [first, *(f"{bits // BITS_PER_KIB} KiB" for bits in BUDGETS)]


def bits_used(rows: list[dict], sketch: str, bits: int) -> str:
    """The bits the sketch used at this budget, or their range where it varies (HyperLogLog++'s
    sparse list grows with n)."""
    used = [row["bits_used"] for row in rows if row["sketch"] == sketch and row["bits"] == bits]
    if not used:
        return "—"
    low, high = min(used), max(used)
    return f"{low:,}" if low == high else f"{low:,} to {high:,}"


def rmse_cell(row: dict | None) -> str:
    if row is None:
        return "—"
    if state(row) == "saturated":
        return "saturated"
    if state(row) == "exact":
        return "0 (exact)"
    return f"{percent_of_n(row['rmse'], row):.2f}%"


def confirmed_cell(group: list[dict]) -> str:
    if not group:
        return "—"
    confirmed = sum(bool(row["rmse_confirmed"]) for row in group)
    return f"{confirmed} of {len(group)} ({confirmed_share(group):.0f}%)"


def merge_cell(rows: list[dict], sketch: str) -> str:
    merges = [row for row in rows if row["sketch"] == sketch + MERGE_SUFFIX and row["merged_equals_direct"] is not None]
    if not merges:
        return "not benchmarked"
    return f"{sum(row['merged_equals_direct'] for row in merges):,} of {sum(row['trials'] for row in merges):,}"


def saturation_cell(rows: list[dict], sketch: str, bits: int) -> str:
    first = first_saturated_n(rows, sketch, bits)
    return f"{first:,}" if first else "none"


def speed_cell(rows: list[dict], sketch: str, bits: int) -> str:
    times = nanoseconds_per_value(rows, sketch, bits)
    return f"{statistics.median(times):.1f}" if times else "—"


def key_numbers(rows: list[dict]) -> str:
    labels = {s: SKETCH_STYLE[s]["label"] for s in SKETCHES}
    lines = ["# Sketch comparison: key numbers", "",
             "Relative RMSE is on the observations streams (one per n). Theory confirmed counts every stream and "
             f"budget group with n >= {MIN_JUDGED_N} where the theory is judged.", "",
             "## Bits of sketch state actually used", ""]
    lines += table(budget_header("Sketch"), [[labels[s], *(bits_used(rows, s, b) for b in BUDGETS)] for s in SKETCHES])
    for n in N_TARGETS:
        lines += ["", f"## Relative RMSE at n = {n:,}", ""]
        lines += table(budget_header("Sketch"), [[labels[s], *(rmse_cell(at_n(rows, s, b, n)) for b in BUDGETS)] for s in SKETCHES])
    lines += ["", "## Predicted RMSE inside the measured 95% interval", ""]
    lines += table([*budget_header("Sketch"), "All budgets"],
                   [[labels[s], *(confirmed_cell(judged_groups(rows, s, b)) for b in BUDGETS), confirmed_cell(judged_groups(rows, s))]
                    for s in SKETCHES])
    once = one_per_observations_n(rows)
    lines += ["", "The same, counting each observations stream's n once (its copies for other distributions of "
              "repeats hold the same values and repeat one result):", ""]
    lines += table([*budget_header("Sketch"), "All budgets"],
                   [[labels[s], *(confirmed_cell(judged_groups(once, s, b)) for b in BUDGETS), confirmed_cell(judged_groups(once, s))]
                    for s in SKETCHES])
    lines += ["", "## Smallest n with a saturated seed", ""]
    lines += table(budget_header("Sketch"), [[labels[s], *(saturation_cell(rows, s, b) for b in BUDGETS)] for s in SKETCHES])
    lines += ["", "## Merge: seeds where the merged sketch gives the same estimate as the direct one", ""]
    lines += table(["Sketch", "Trials"], [[labels[s], merge_cell(rows, s)] for s in SKETCHES])
    lines += ["", f"## Median nanoseconds per value (streams of {MIN_TIMED_VALUES:,}+ values)", ""]
    lines += table(budget_header("Sketch"), [[labels[s], *(speed_cell(rows, s, b) for b in BUDGETS)] for s in SKETCHES])
    return "\n".join(lines) + "\n"


# --- Command line ---------------------------------------------------------------------------

def draw_all(rows: list[dict], output: Path) -> list[Path]:
    written = [
        *plot_relative_rmse_vs_n(rows, output / "relative_rmse_vs_n"),
        *plot_relative_bias_vs_n(rows, output / "relative_bias_vs_n"),
        *plot_equal_memory(rows, output / "equal_memory"),
        *plot_theory_confirmed(rows, output / "theory_confirmed"),
        *plot_theory_ratio(rows, output / "theory_ratio"),
        *plot_merge(rows, output / "merge"),
        *plot_time_per_value(rows, output / "time_per_value"),
        *plot_time_vs_length(rows, output / "time_vs_length"),
        *plot_speed_vs_accuracy(rows, output / "speed_vs_accuracy"),
        *plot_rmse_vs_n_at_budget(rows, output / f"rmse_vs_n_{LARGEST_BUDGET // BITS_PER_KIB}kib"),
    ]
    numbers = output / "comparison_key_numbers.md"
    numbers.parent.mkdir(parents=True, exist_ok=True)
    numbers.write_text(key_numbers(rows), encoding="utf-8")
    return written + [numbers]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="summary.csv from results_analyser.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "comparison", help="directory for the figures")
    args = parser.parse_args()

    if not args.summary.is_file():
        print(f"no summary at {args.summary} (run python/results_analyser.py first)", file=sys.stderr)
        return 1
    names = tuple(name for spec in SPECS.values() for name in (spec.name, spec.merge) if name)
    for path in draw_all(load_rows(args.summary, names, tuple(KIND_STYLE)), args.output):
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
