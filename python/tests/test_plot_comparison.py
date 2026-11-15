"""Tests for plot_comparison.py: picking one row per n, edges, timings, merges, the key numbers,
and that every comparison figure is written."""
from pathlib import Path

import pytest

from plot_comparison import (
    LARGEST_BUDGET,
    PERCENT_CEILING,
    PERCENT_FLOOR,
    SKETCHES,
    TWIN_DODGE,
    TWIN_LABEL,
    absolute_values,
    bits_used,
    by_n,
    dodged,
    draw_all,
    edge,
    edge_text_place,
    key_numbers,
    merge_shares,
    nanoseconds_per_value,
    one_per_observations_n,
    plot_rmse_vs_n_at_budget,
    rmse_cell,
    state,
    times_by_length,
    top_edge_text,
    twin_handles,
    values_per_stream,
)
from plot_sketches import stream_kind

BUDGETS = (8192, 32768, 65536, 131072)


def row(sketch: str, stream: str, bits: int = 8192, true_n: int = 1000, rmse: float | None = 10.0, saturated: int = 0,
        confirmed: bool | None = True, merged: int | None = None, time_ms: float = 5.0, bits_used: int | None = None) -> dict:
    """One row as load_rows returns it."""
    return {
        "stream": stream, "sketch": sketch, "bits": bits, "bits_used": bits if bits_used is None else bits_used,
        "true_n": true_n, "trials": 64, "saturated": saturated,
        "bias": None if rmse is None else 0.1 * rmse, "bias_low": None, "bias_high": None,
        "rmse": rmse, "rmse_low": None if rmse is None else 0.8 * rmse, "rmse_high": None if rmse is None else 1.2 * rmse,
        "predicted_bias": 0.0, "predicted_rmse": 10.0, "bias_confirmed": confirmed, "rmse_confirmed": confirmed,
        "merged_equals_direct": merged, "kind": stream_kind(stream), "mean_time_ms": time_ms,
    }


def made_up_rows() -> list[dict]:
    """Every sketch at every budget on observations streams (two identical copies per n), SQL and
    distribution generator streams; Linear Counting saturates at n = 1,000,000 and KMV is exact
    at n = 10; Linear Counting, HyperLogLog and HyperLogLog++ have merge rows."""
    rows = []
    for sketch in SKETCHES:
        for bits in BUDGETS:
            for n in (10, 1_000, 10_000, 100_000, 1_000_000):
                saturated = 3 if sketch == "LinearCounting" and n == 1_000_000 else 0
                rmse = None if saturated else (0.0 if sketch == "KMinVals" and n == 10 else 0.01 * n)
                for copy in ("poisson", "geometric"):
                    rows.append(row(sketch, f"{copy}_observations_seed1_m{n}", bits, n, rmse, saturated))
            rows.append(row(sketch, "job_query_1a", bits, 500, 5.0, confirmed=False))
            for stream in ("poisson_seed1_nsample1000000", "beta_seed1_nsample1000000", "beta_seed1_nsample10000"):
                rows.append(row(sketch, stream, bits, 900, 9.0, time_ms=10.0))
            if sketch in ("LinearCounting", "HyperLogLog", "HyperLogLogPlusPlus"):
                merged = 40 if sketch == "HyperLogLogPlusPlus" else 64
                rows.append(row(sketch + "Merge", "poisson_observations_seed1_m1000", bits, 1000, 10.0, merged=merged))
    return rows


def test_by_n_keeps_one_observations_row_per_n_in_order() -> None:
    points = by_n(made_up_rows(), "HyperLogLog", 8192)
    assert [p["true_n"] for p in points] == [10, 1_000, 10_000, 100_000, 1_000_000]
    assert all(p["kind"] == "observations" for p in points)


def test_state_and_edge_put_saturated_on_top_and_exact_or_tiny_on_the_bottom() -> None:
    saturated = row("LinearCounting", "x_observations_m1", rmse=None, saturated=1)
    exact = row("KMinVals", "x_observations_m1", rmse=0.0)
    tiny = row("HyperLogLogPlusPlus", "x_observations_m1", true_n=1000, rmse=PERCENT_FLOOR / 200 * 1000)
    normal = row("HyperLogLog", "x_observations_m1", true_n=1000, rmse=10.0)
    assert [state(r) for r in (saturated, exact, tiny, normal)] == ["saturated", "exact", "measured", "measured"]
    assert [edge(r) for r in (saturated, exact, tiny, normal, None)] == ["top", "bottom", "bottom", None, None]


def test_stream_length_and_time_per_value_use_long_distribution_streams_only() -> None:
    assert values_per_stream("beta_seed1_nsample1000000_bins64") == 1_000_000
    assert values_per_stream("job_query_1a") is None
    times = nanoseconds_per_value(made_up_rows(), "BloomFilter", 8192)
    assert times == [pytest.approx(10.0), pytest.approx(10.0)]  # 10 ms over 1,000,000 values, twice


def test_times_by_length_groups_by_stream_length() -> None:
    by_length = times_by_length(made_up_rows(), "HyperLogLog", 8192)
    assert list(by_length) == [10_000, 1_000_000]
    assert by_length[10_000] == [pytest.approx(1000.0)]  # 10 ms over 10,000 values
    assert by_length[1_000_000] == [pytest.approx(10.0), pytest.approx(10.0)]


def test_merge_shares_are_percent_of_seeds() -> None:
    rows = made_up_rows()
    assert merge_shares(rows, "HyperLogLogPlusPlus", 8192) == [(1000, pytest.approx(62.5))]
    assert merge_shares(rows, "KMinVals", 8192) == []


def test_bits_used_shows_a_range_where_it_varies() -> None:
    rows = [row("HyperLogLogPlusPlus", "a_observations_m1", bits_used=used) for used in (32, 6144, 640)]
    assert bits_used(rows, "HyperLogLogPlusPlus", 8192) == "32 to 6,144"
    assert bits_used([row("LinearCounting", "a_observations_m1")], "LinearCounting", 8192) == "8,192"


def test_rmse_cell_names_saturated_and_exact_groups() -> None:
    assert rmse_cell(None) == "—"
    assert rmse_cell(row("LinearCounting", "a", rmse=None, saturated=2)) == "saturated"
    assert rmse_cell(row("KMinVals", "a", rmse=0.0)) == "0 (exact)"
    assert rmse_cell(row("HyperLogLog", "a", true_n=1000, rmse=25.0)) == "2.50%"


def test_key_numbers_tables() -> None:
    text = key_numbers(made_up_rows())
    assert "| Linear Counting | 1.00% | 1.00% | 1.00% | 1.00% |" in text  # n = 1,000
    assert "| Linear Counting | saturated | saturated | saturated | saturated |" in text  # n = 1,000,000
    assert "| Bloom filter | not benchmarked |" in text
    assert "| HyperLogLog++ | 160 of 256 |" in text
    assert "| Linear Counting | 1,000,000 | 1,000,000 | 1,000,000 | 1,000,000 |" in text
    assert "| K-Minimum Values | 10.0 | 10.0 | 10.0 | 10.0 |" in text


def test_rmse_over_the_ceiling_goes_on_the_top_edge() -> None:
    assert edge(row("LinearCounting", "x_observations_m1", true_n=10, rmse=PERCENT_CEILING / 10)) == "top"


def test_confirmation_counts_each_observations_n_once_in_the_second_table() -> None:
    once = one_per_observations_n(made_up_rows())
    assert len([r for r in once if r["sketch"] == "HyperLogLog" and r["kind"] == "observations"]) == 5 * len(BUDGETS)
    text = key_numbers(made_up_rows())
    # Per budget, n >= 100: 4 observations n x 2 copies + 1 SQL (not confirmed) + 3 nsample = 12 groups,
    # or 8 counting each observations n once.
    assert "| HyperLogLog | 11 of 12 (92%) | 11 of 12 (92%) | 11 of 12 (92%) | 11 of 12 (92%) | 44 of 48 (92%) |" in text
    assert "| HyperLogLog | 7 of 8 (88%) | 7 of 8 (88%) | 7 of 8 (88%) | 7 of 8 (88%) | 28 of 32 (88%) |" in text


def test_absolute_values_leave_out_rows_drawn_on_the_edges() -> None:
    points = [row("LinearCounting", "a_observations_m1", rmse=None, saturated=2),
              row("KMinVals", "a_observations_m1", rmse=0.0),
              row("HyperLogLog", "a_observations_m1", rmse=10.0)]
    assert absolute_values(points) == ([None, None, 10.0], [None, None, 8.0], [None, None, 12.0])


def test_hllpp_moves_right_only_where_it_equals_hll() -> None:
    hll = [row("HyperLogLog", "a_observations_m1", rmse=r) for r in (10.0, 20.0)] + [None]
    plus = [row("HyperLogLogPlusPlus", "a_observations_m1", rmse=r) for r in (10.0, 15.0, 5.0)]
    assert dodged("HyperLogLogPlusPlus", [1, 2, 3], plus, hll) == ([pytest.approx(TWIN_DODGE), 2, 3], True)
    assert dodged("HyperLogLog", [1, 2, 3], hll, hll) == ([1, 2, 3], False)
    assert twin_handles(False) == {} and list(twin_handles(True)) == [TWIN_LABEL]


def test_top_edge_text_names_saturated_and_off_scale_rows() -> None:
    saturated = row("LinearCounting", "a_observations_m1", rmse=None, saturated=1)
    off_scale = row("LinearCounting", "a_observations_m1", true_n=10, rmse=PERCENT_CEILING)
    assert top_edge_text([saturated]) == "saturated"
    assert top_edge_text([off_scale]) == "off scale"
    assert top_edge_text([saturated, off_scale]) == "saturated / off scale"


def test_edge_text_goes_right_then_left_then_in_the_widest_gap() -> None:
    assert edge_text_place([10, 100], 1, 1e6) == (100, "left")
    assert edge_text_place([1e5, 1e6], 1, 1e6) == (1e5, "right")
    x, align = edge_text_place([1, 2, 1e6], 1, 1e6)
    assert align == "center" and 2 < x < 1e6
    assert edge_text_place([5.0], 0, 100, log=False) == (5.0, "left")


def test_rmse_at_one_budget_needs_rows_at_that_budget(tmp_path: Path) -> None:
    small_only = [r for r in made_up_rows() if r["bits"] == 8192]
    assert plot_rmse_vs_n_at_budget(small_only, tmp_path / "none") == []
    assert len(plot_rmse_vs_n_at_budget(made_up_rows(), tmp_path / "largest")) == 2
    assert LARGEST_BUDGET == 131072


def test_draw_all_writes_every_figure_and_the_key_numbers(tmp_path: Path) -> None:
    paths = draw_all(made_up_rows(), tmp_path)
    stems = ("relative_rmse_vs_n", "relative_bias_vs_n", "equal_memory", "theory_confirmed", "theory_ratio",
             "merge", "time_per_value", "time_vs_length", "speed_vs_accuracy", "rmse_vs_n_16kib")
    expected = {f"{stem}.{suffix}" for stem in stems for suffix in ("pdf", "png")} | {"comparison_key_numbers.md"}
    assert {path.name for path in paths} == expected
    assert all(path.stat().st_size > 0 for path in paths)


def test_draw_all_without_rows_writes_only_the_key_numbers(tmp_path: Path) -> None:
    assert [path.name for path in draw_all([], tmp_path)] == ["comparison_key_numbers.md"]
