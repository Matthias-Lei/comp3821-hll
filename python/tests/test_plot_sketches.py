"""Tests for plot_sketches.py: each sketch's theory helpers, regions, key numbers, and that every
figure is written for all five sketches."""
import math
from pathlib import Path

import pytest

from plot_sketches import (
    BLOOM_FILTER,
    HYPERLOGLOG,
    HYPERLOGLOG_PLUS_PLUS,
    K_MINIMUM_VALUES,
    LINEAR_COUNTING,
    SPECS,
    draw_sketch,
    key_numbers,
    kmv_k,
    plot_rmse_vs_theory,
    plot_saturation,
    predicted_rmse_curve,
    rmse_ratios,
    saturation_n,
    saturation_probability,
    stream_kind,
    unjudged_ranges,
)
from results_analyser import THEORY, hyperloglog_registers

M = 8192
BUDGETS = (8192, 32768, 65536, 131072)


def row(sketch: str, stream: str = "poisson_observations_seed1_m1000", bits: int = M, true_n: int = 1000,
        rmse: float = 10.0, saturated: int = 0, confirmed: bool | None = True, merged: int | None = None) -> dict:
    """One row as load_rows returns it."""
    return {
        "stream": stream, "sketch": sketch, "bits": bits, "bits_used": bits, "true_n": true_n, "trials": 64,
        "saturated": saturated, "bias": 0.5, "bias_low": -1.0, "bias_high": 2.0,
        "rmse": rmse, "rmse_low": rmse * 0.8, "rmse_high": rmse * 1.2,
        "predicted_bias": 0.0, "predicted_rmse": 10.0, "bias_confirmed": confirmed, "rmse_confirmed": confirmed,
        "merged_equals_direct": merged, "kind": stream_kind(stream), "mean_time_ms": 1.0,
    }


def made_up_rows(sketch: str) -> list[dict]:
    """Per budget: judged groups at n = 100 to 100,000 (SQL ones not confirmed), one group the
    theory does not judge, an exact group at n = 10, and three groups around saturation."""
    rows = []
    for bits in BUDGETS:
        for n in (100, 1000, 10_000, 100_000):
            rows.append(row(sketch, stream=f"poisson_observations_seed1_m{n}", bits=bits, true_n=n))
            rows.append(row(sketch, stream=f"job_query_{n}", bits=bits, true_n=n, confirmed=False))
        rows.append(row(sketch, stream="geometric_observations_seed1_m50000", bits=bits, true_n=50_000, confirmed=None))
        rows.append(row(sketch, stream="poisson_observations_seed1_m10", bits=bits, true_n=10, rmse=0.0))
        for share in (0.5, 1.0, 1.5):
            n = round(share * saturation_n(bits, 3))
            rows.append(row(sketch, stream=f"poisson_observations_seed1_m{n}", bits=bits, true_n=n,
                            saturated=round(64 * saturation_probability(n, bits, 3))))
    return rows


def test_every_sketch_has_a_spec_with_theory_and_a_unique_short_name() -> None:
    assert set(SPECS) == set(THEORY)
    assert len({spec.short for spec in SPECS.values()}) == len(SPECS)


def test_saturation_with_k_hashes_comes_k_times_sooner() -> None:
    assert saturation_n(M, 3) == pytest.approx(saturation_n(M) / 3)
    assert saturation_probability(saturation_n(M, 3), M, 3) == pytest.approx(math.exp(-1))


def test_bloom_curve_stops_at_its_own_saturation() -> None:
    ns, rmse = predicted_rmse_curve(BLOOM_FILTER, M, 10, 10 * saturation_n(M))
    assert ns[-1] == pytest.approx(saturation_n(M, 3))
    assert rmse[0] == pytest.approx(THEORY["BloomFilter"](10, M)[1])


def test_kmv_curve_leaves_out_the_exact_range_below_k() -> None:
    ns, rmse = predicted_rmse_curve(K_MINIMUM_VALUES, M, 10, 1e6)
    assert kmv_k(M) == 128
    assert min(ns) >= 128
    assert all(value > 0 for value in rmse)


def test_unjudged_ranges_follow_results_analyser() -> None:
    r = hyperloglog_registers(M)
    [(low, high)] = unjudged_ranges(HYPERLOGLOG, M, 100, 1e6)
    assert low == pytest.approx(2 * r, rel=0.03)
    assert high == pytest.approx(5 * r, rel=0.03)
    [(low, high)] = unjudged_ranges(HYPERLOGLOG_PLUS_PLUS, M, 100, 1e6)
    assert low == pytest.approx(100)
    assert high == pytest.approx(5 * r, rel=0.03)
    assert unjudged_ranges(LINEAR_COUNTING, M, 100, 1e6) == []


def test_rmse_ratios_leave_out_groups_where_the_theory_is_not_judged() -> None:
    ratios = rmse_ratios(made_up_rows("HyperLogLog"), HYPERLOGLOG)
    assert list(ratios) == ["poisson", "real SQL"]


def test_key_numbers_of_kmv_count_exactness_and_say_merge_is_not_benchmarked() -> None:
    text = key_numbers(made_up_rows("KMinVals"), K_MINIMUM_VALUES)
    assert "# K-Minimum Values: key numbers" in text
    assert "- **Merge:** not benchmarked for this sketch." in text
    # n < k: n = 10 (exact) at every budget, n = 100 everywhere and n = 1,000 at 8 and 16 KiB.
    assert "every seed gave exactly n in **4 of 16**" in text
    assert "## Saturation" not in text


def test_key_numbers_count_merges_and_list_bloom_saturation() -> None:
    rows = made_up_rows("HyperLogLog") + [row("HyperLogLogMerge", merged=60), row("HyperLogLogMerge", merged=64)]
    assert "in **124 of 128** trials" in key_numbers(rows, HYPERLOGLOG)
    bloom = key_numbers(made_up_rows("BloomFilter"), BLOOM_FILTER)
    assert "## Saturation" in bloom
    assert f"| 1 KiB | {saturation_n(M, 3):,.0f} |" in bloom


def test_saturation_figure_only_for_sketches_whose_bitmap_fills(tmp_path: Path) -> None:
    assert plot_saturation(made_up_rows("KMinVals"), K_MINIMUM_VALUES, tmp_path / "kmv") == []
    assert len(plot_saturation(made_up_rows("BloomFilter"), BLOOM_FILTER, tmp_path / "bloom")) == 2


def test_rmse_figure_draws_groups_where_the_theory_is_not_judged(tmp_path: Path) -> None:
    rows = [row("HyperLogLog", confirmed=None, true_n=n) for n in (300, 3000)]
    assert len(plot_rmse_vs_theory(rows, HYPERLOGLOG, tmp_path / "hll")) == 2


@pytest.mark.parametrize("name", list(SPECS))
def test_draw_sketch_writes_every_figure_and_the_key_numbers(tmp_path: Path, name: str) -> None:
    spec = SPECS[name]
    paths = draw_sketch(made_up_rows(name), spec, tmp_path)
    figures = ["rmse_vs_theory", "bias", "by_distribution"] + (["saturation"] if spec.hashes else [])
    expected = {f"{spec.short}_{figure}.{suffix}" for figure in figures for suffix in ("pdf", "png")}
    assert {path.name for path in paths} == expected | {f"{spec.short}_key_numbers.md"}
    assert all(path.stat().st_size > 0 for path in paths)


def test_draw_sketch_without_rows_writes_only_the_key_numbers(tmp_path: Path) -> None:
    assert [path.name for path in draw_sketch([], HYPERLOGLOG, tmp_path)] == ["hll_key_numbers.md"]
