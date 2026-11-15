"""Tests for plot_results.py: the data preparation, and that both figures are written."""
import csv
from pathlib import Path

import pytest

from plot_results import (
    comparison_series,
    load_summary,
    log_error_bars,
    percent_tick,
    pick_comparison_stream,
    plot_rmse_vs_theory,
    plot_sketch_comparison,
    spread_log_positions,
    theory_curve,
    theory_panels,
)
from results_analyser import CSV_FIELDS, linear_counting_theory

BUDGETS = (8192, 32768, 65536, 131072)


def row(stream: str = "s", sketch: str = "LinearCounting", bits: int = 8192, true_n: int = 1000, rmse: float | None = 10.0, saturated: int = 0, confirmed: bool | None = True) -> dict:
    """One summary row, already parsed, as plot_results.load_summary returns it."""
    return {
        "stream": stream,
        "sketch": sketch,
        "bits": bits,
        "bits_used": bits,
        "true_n": true_n,
        "trials": 64,
        "saturated": saturated,
        "bias": 0.5,
        "bias_low": -1.0,
        "bias_high": 2.0,
        "rmse": rmse,
        "rmse_low": None if rmse is None else rmse * 0.8,
        "rmse_high": None if rmse is None else rmse * 1.2,
        "predicted_bias": 0.4,
        "predicted_rmse": 9.0,
        "bias_confirmed": confirmed,
        "rmse_confirmed": confirmed,
        "command": "./build/benchmarks s.cebs",
    }


def test_load_summary_converts_types_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row() | {"bias_confirmed": True, "rmse_confirmed": False})
        writer.writerow(row(rmse=None, saturated=64, confirmed=None))

    first, saturated = load_summary(path)

    assert (first["bits"], first["true_n"], first["rmse"], first["bias_confirmed"], first["rmse_confirmed"]) == (8192, 1000, 10.0, True, False)
    assert (saturated["saturated"], saturated["rmse"], saturated["rmse_low"], saturated["rmse_confirmed"]) == (64, None, None, None)


def test_theory_panels_keep_only_judgeable_rows_sorted_by_n() -> None:
    rows = [
        row(true_n=500),
        row(true_n=100),
        row(true_n=200, saturated=3),
        row(true_n=300, sketch="HyperLogLog"),
        row(true_n=400, rmse=0.0),
        row(true_n=600, bits=32768),
    ]
    rows.append(row(true_n=700) | {"rmse_low": None, "rmse_high": None})

    panels = theory_panels(rows, "LinearCounting")

    assert {bits: [r["true_n"] for r in group] for bits, group in panels.items()} == {8192: [100, 500], 32768: [600]}


def test_theory_curve_spans_the_range_with_the_analyser_formula() -> None:
    ns, rmse = theory_curve("LinearCounting", 8192, 10, 10_000, points=50)
    assert len(ns) == len(rmse) == 50
    assert ns[0] == pytest.approx(10) and ns[-1] == pytest.approx(10_000)
    assert rmse[0] == pytest.approx(linear_counting_theory(10, 8192)[1])
    assert rmse[-1] == pytest.approx(linear_counting_theory(10_000, 8192)[1])


def test_pick_comparison_stream_wants_every_sketch_unsaturated_at_every_budget() -> None:
    sketches = ("LinearCounting", "BloomFilter", "HyperLogLog")
    complete_small = [row(stream="small", sketch=s, bits=b, true_n=5000) for s in sketches for b in BUDGETS]
    incomplete_big = [row(stream="big", sketch=s, bits=b, true_n=9000, saturated=5 if (s, b) == ("HyperLogLog", 8192) else 0) for s in sketches for b in BUDGETS]

    assert pick_comparison_stream(complete_small + incomplete_big) == "small"
    assert pick_comparison_stream(incomplete_big) is None


def test_comparison_series_uses_fixed_sketch_order_and_sorts_budgets() -> None:
    rows = [
        row(sketch="HyperLogLog", bits=32768),
        row(sketch="LinearCounting", bits=32768),
        row(sketch="LinearCounting", bits=8192),
        row(sketch="BloomFilter", bits=8192, rmse=None, saturated=64),
        row(sketch="DummySketch", bits=8192),
        row(stream="other", sketch="BloomFilter"),
    ]

    series = comparison_series(rows, "s")

    assert list(series) == ["LinearCounting", "HyperLogLog"]
    assert [r["bits"] for r in series["LinearCounting"]] == [8192, 32768]


def test_log_error_bars_draw_an_interval_reaching_zero_to_a_tenth() -> None:
    below, above = log_error_bars([10.0, 2.0], [8.0, 0.0], [12.0, 5.0])
    assert below == pytest.approx([2.0, 1.8])
    assert above == pytest.approx([2.0, 3.0])


def test_percent_tick_labels_only_the_1_2_3_5_steps() -> None:
    assert [percent_tick(v, 0) for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 3.0)] == ["0.1%", "0.2%", "0.3%", "", "0.5%", "", "1%", "3%"]
    assert percent_tick(0.0, 0) == ""


def test_spread_log_positions_separates_close_labels_and_keeps_order() -> None:
    assert spread_log_positions([1.0, 1.1, 5.0]) == pytest.approx([1.0, 1.25, 5.0])
    assert spread_log_positions([5.0, 1.0, 1.1]) == pytest.approx([5.0, 1.0, 1.25])
    assert spread_log_positions([1.0, 1.0, 1.0]) == pytest.approx([1.0, 1.25, 1.5625])


def test_figures_are_written(tmp_path: Path) -> None:
    rows = [row(bits=b, true_n=n, rmse=n / 50, confirmed=n != 2000) for b in BUDGETS for n in (500, 1000, 2000)]
    rows += [row(sketch=s, bits=b, true_n=1000) for s in ("BloomFilter", "HyperLogLog") for b in BUDGETS]

    written = plot_rmse_vs_theory(rows, "LinearCounting", tmp_path / "theory") + plot_sketch_comparison(rows, "s", tmp_path / "comparison")

    assert [path.name for path in written] == ["theory.pdf", "theory.png", "comparison.pdf", "comparison.png"]
    assert all(path.stat().st_size > 0 for path in written)


def test_no_figure_without_data(tmp_path: Path) -> None:
    assert plot_rmse_vs_theory([], "LinearCounting", tmp_path / "theory") == []
    assert plot_sketch_comparison([], "s", tmp_path / "comparison") == []
    assert not any(tmp_path.iterdir())
