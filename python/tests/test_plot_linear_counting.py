"""Tests for plot_linear_counting.py: stream kinds, the theory helpers, the key numbers, and that
every figure is written."""
import csv
import math
from pathlib import Path

import pytest

from plot_linear_counting import (
    MERGE_SKETCH,
    SKETCH,
    SQL_LABEL,
    distribution_of,
    key_numbers,
    load_rows,
    plot_bias,
    plot_by_distribution,
    plot_rmse_vs_theory,
    plot_saturation,
    predicted_rmse_curve,
    rmse_ratios,
    saturation_n,
    saturation_probability,
    stream_kind,
)
from results_analyser import CSV_FIELDS, linear_counting_theory

M = 8192


def row(stream: str = "poisson_observations_seed1_m1000", sketch: str = SKETCH, bits: int = M, true_n: int = 1000,
        rmse: float = 10.0, saturated: int = 0, confirmed: bool = True, merged: int | None = None) -> dict:
    """One row as load_rows returns it."""
    return {
        "stream": stream, "sketch": sketch, "bits": bits, "bits_used": bits, "true_n": true_n, "trials": 64,
        "saturated": saturated, "bias": 0.5, "bias_low": -1.0, "bias_high": 2.0,
        "rmse": rmse, "rmse_low": rmse * 0.8, "rmse_high": rmse * 1.2,
        "predicted_bias": 0.4, "predicted_rmse": 10.0, "bias_confirmed": confirmed, "rmse_confirmed": confirmed,
        "merged_equals_direct": merged, "kind": stream_kind(stream),
    }


def made_up_rows() -> list[dict]:
    rows = []
    for bits in (8192, 32768, 65536, 131072):
        for n in (100, 1000, 10_000):
            rows.append(row(stream=f"poisson_observations_seed1_m{n}", bits=bits, true_n=n))
            rows.append(row(stream=f"pareto_observations_seed1_m{n}", bits=bits, true_n=n, rmse=12.0, confirmed=False))
            rows.append(row(stream=f"job_query_{n}", bits=bits, true_n=n))
        rows.append(row(stream="beta_seed1_nsample10000", bits=bits, true_n=500))
        for share in (0.5, 1.0, 1.5):
            n = round(share * saturation_n(bits))
            rows.append(row(stream=f"poisson_observations_seed1_m{n}", bits=bits, true_n=n,
                            saturated=round(64 * saturation_probability(n, bits))))
    rows.append(row(sketch=MERGE_SKETCH, merged=64))
    rows.append(row(sketch=MERGE_SKETCH, merged=63))
    return rows


@pytest.mark.parametrize(("stream", "kind", "distribution"), [
    ("poisson_observations_seed7427369_m1000", "observations", "poisson"),
    ("negative_binomial_observations_seed7427369_m50", "observations", "negative_binomial"),
    ("chi_squared_seed7427369_nsample10000", "nsample", None),
    ("job_query_10c", "sql", SQL_LABEL),
    ("lichess_query_004", "sql", SQL_LABEL),
    ("range_python", None, None),
])
def test_stream_kind_and_distribution_from_the_name(stream: str, kind: str | None, distribution: str | None) -> None:
    assert stream_kind(stream) == kind
    assert distribution_of(stream) == distribution


def test_saturation_probability_rises_from_zero_to_one_around_m_ln_m() -> None:
    assert saturation_probability(0.5 * saturation_n(M), M) < 1e-6
    assert saturation_probability(saturation_n(M), M) == pytest.approx(math.exp(-1))
    assert saturation_probability(2 * saturation_n(M), M) > 0.999


def test_predicted_rmse_curve_stops_at_saturation_and_matches_theory() -> None:
    ns, rmse = predicted_rmse_curve(M, 10, 10 * saturation_n(M))
    assert ns[0] == pytest.approx(10)
    assert ns[-1] == pytest.approx(saturation_n(M))
    assert rmse[0] == pytest.approx(linear_counting_theory(10, M)[1])


def test_rmse_ratios_by_distribution_skip_small_n_and_put_sql_last() -> None:
    rows = made_up_rows() + [row(stream="geometric_observations_seed1_m50", true_n=50)]
    ratios = rmse_ratios(rows)
    assert list(ratios) == ["pareto", "poisson", SQL_LABEL]
    assert ratios["pareto"] == [pytest.approx(1.2)] * 12
    assert len(ratios[SQL_LABEL]) == 12


def test_key_numbers_count_merges_and_confirmations() -> None:
    text = key_numbers(made_up_rows())
    assert "stream in **127 of 128** trials" in text  # merges: 64 + 63
    assert "stream-lib" not in text
    assert "| real SQL query results, 16 KiB | 3 of 3 | 3 of 3 |" in text
    assert "| pareto | 0 of 12 | 0 of 12 | 1.200 |" in text


def test_every_figure_is_written(tmp_path: Path) -> None:
    rows = made_up_rows()
    figures = {
        "rmse": plot_rmse_vs_theory(rows, tmp_path / "rmse"),
        "bias": plot_bias(rows, tmp_path / "bias"),
        "saturation": plot_saturation(rows, tmp_path / "saturation"),
        "distribution": plot_by_distribution(rows, tmp_path / "distribution"),
    }
    for name, paths in figures.items():
        assert [path.suffix for path in paths] == [".pdf", ".png"], name
        assert all(path.stat().st_size > 0 for path in paths), name


def test_figures_are_skipped_without_data(tmp_path: Path) -> None:
    assert plot_rmse_vs_theory([], tmp_path / "rmse") == []
    assert plot_saturation([], tmp_path / "saturation") == []
    assert plot_by_distribution([], tmp_path / "distribution") == []


def test_load_rows_reads_summary_csv_and_filters_kinds(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    blank = dict.fromkeys(CSV_FIELDS, "")
    common = {"bits": 8192, "true_n": 5, "trials": 64, "saturated": 0, "bits_used": 8192}
    with open(summary, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow({**blank, **common, "stream": "job_query_1a", "sketch": SKETCH})
        writer.writerow({**blank, **common, "stream": "job_query_1a", "sketch": MERGE_SKETCH, "merged_equals_direct": 64})
        writer.writerow({**blank, **common, "stream": "job_query_1a", "sketch": "HyperLogLog"})
        writer.writerow({**blank, **common, "stream": "beta_seed1_nsample10000", "sketch": SKETCH})

    rows = load_rows(summary)
    assert [(r["sketch"], r["kind"], r["merged_equals_direct"]) for r in rows] == [
        (SKETCH, "sql", None), (MERGE_SKETCH, "sql", 64), (SKETCH, "nsample", None)]
    assert [r["kind"] for r in load_rows(summary, ("sql",))] == ["sql", "sql"]
