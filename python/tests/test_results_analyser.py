"""Tests for results_analyser.py.

Each test writes tiny made-up benchmark results whose answers can be worked out by hand,
then checks that the analyser gets the same numbers.
"""
import csv
import json
import math
from pathlib import Path

import pytest

from results_analyser import (
    Interval,
    group_trials,
    linear_counting_theory,
    load_all_trials,
    load_trials,
    mean_with_interval,
    sketch_name,
    summarise,
    summarise_all,
    write_csv,
    write_markdown,
)

N = 1000  # true number of distinct values in the made-up streams
BITS = 8192  # memory budget


def run(seed: int, estimate: float | None, sketch: str = "LinearCounting", bits: int = BITS, true_n: int = N, bits_used: int | None = None) -> dict:
    """One benchmark entry, shaped like Google Benchmark's JSON. estimate=None means saturated."""
    entry = {
        "name": f"BM_{sketch}/bits:{bits}/seed:{seed}/iterations:1",
        "run_type": "iteration",
        "bits": float(bits),
        "bits_used": float(bits if bits_used is None else bits_used),
        "seed": float(seed),
        "true_n": float(true_n),
        "saturated": 0.0 if estimate is not None else 1.0,
    }
    if estimate is not None:
        entry["estimate"] = estimate
    return entry


def write_results(directory: Path, stream: str, entries: list[dict], command: str = "./build/benchmarks s.cebs") -> Path:
    """Writes one results file, shaped like run_benchmarks.py's output."""
    path = directory / f"{stream}.json"
    path.write_text(json.dumps({"context": {}, "benchmarks": entries, "command": command}))
    return path


# --- Loading ---------------------------------------------------------------------------------

def test_sketch_name() -> None:
    assert sketch_name("BM_LinearCounting/bits:8192/seed:7/iterations:1") == "LinearCounting"
    assert sketch_name("BM_DummySketch/bits:8192/seed:1") == "DummySketch"


def test_load_trials_reads_estimates_and_saturation(tmp_path: Path) -> None:
    aggregate = run(1, 5.0) | {"run_type": "aggregate"}
    path = write_results(tmp_path, "poisson", [run(1, 1001.0), run(2, None), aggregate])

    trials = load_trials(path)

    assert [(t.stream, t.sketch, t.bits, t.seed, t.true_n, t.estimate) for t in trials] == [
        ("poisson", "LinearCounting", BITS, 1, N, 1001.0),
        ("poisson", "LinearCounting", BITS, 2, N, None),
    ]
    assert trials[0].command == "./build/benchmarks s.cebs"


def test_load_trials_rejects_old_output_without_seed(tmp_path: Path) -> None:
    old = run(1, 1001.0)
    del old["seed"]
    with pytest.raises(ValueError, match="seed"):
        load_trials(write_results(tmp_path, "old", [old]))


def test_load_all_trials_reads_every_json_file(tmp_path: Path) -> None:
    write_results(tmp_path, "a", [run(1, 1000.0)])
    write_results(tmp_path, "b", [run(1, 1000.0), run(2, 1000.0)])
    (tmp_path / "notes.txt").write_text("not json")

    assert sorted(t.stream for t in load_all_trials(tmp_path)) == ["a", "b", "b"]


# --- Grouping and statistics -----------------------------------------------------------------

def test_group_trials_by_stream_sketch_and_budget(tmp_path: Path) -> None:
    path = write_results(tmp_path, "s", [run(1, 1.0), run(2, 1.0), run(1, 1.0, bits=32768), run(1, 1.0, sketch="DummySketch")])
    groups = group_trials(load_trials(path))
    assert {key: len(value) for key, value in groups.items()} == {
        ("s", "LinearCounting", BITS): 2,
        ("s", "LinearCounting", 32768): 1,
        ("s", "DummySketch", BITS): 1,
    }


def test_mean_with_interval() -> None:
    # {1, -1, 1, -1}: mean 0, sample sd sqrt(4/3), t(0.975, 3) = 3.182446, so half width t * sd / sqrt(4).
    mean, ci = mean_with_interval([1.0, -1.0, 1.0, -1.0])
    half_width = 3.182446 * math.sqrt(4 / 3) / 2
    assert mean == 0
    assert ci.low == pytest.approx(-half_width, abs=1e-4)
    assert ci.high == pytest.approx(half_width, abs=1e-4)
    assert mean_with_interval([3.0]) == (3.0, None)


def test_bias_interval_uses_the_sample_standard_deviation(tmp_path: Path) -> None:
    # Errors {0, 2, 4, 6}: mean 3, sample sd sqrt(20 / 3). Dividing by 4 instead of 3 would fail this.
    summary = summarise(load_trials(write_results(tmp_path, "s", [run(seed, N + 2.0 * (seed - 1)) for seed in range(1, 5)])))
    half_width = 3.182446 * math.sqrt(20 / 3) / 2
    assert summary.bias == pytest.approx(3.0)
    assert summary.bias_ci.low == pytest.approx(3 - half_width, abs=1e-4)
    assert summary.bias_ci.high == pytest.approx(3 + half_width, abs=1e-4)


def test_summarise_counts_saturation_and_rmse(tmp_path: Path) -> None:
    entries = [run(1, N + 1.0), run(2, N - 1.0), run(3, N + 1.0), run(4, N - 1.0), run(5, None)]
    summary = summarise(load_trials(write_results(tmp_path, "s", entries)))

    assert (summary.trials, summary.saturated) == (5, 1)
    assert summary.bias == pytest.approx(0.0)
    assert summary.rmse == pytest.approx(1.0)
    # Every squared error is 1, so the RMSE interval collapses to [1, 1].
    assert summary.rmse_ci.low == pytest.approx(1.0)
    assert summary.rmse_ci.high == pytest.approx(1.0)


def test_rmse_interval_low_end_is_clamped_at_zero(tmp_path: Path) -> None:
    # Squared errors {0, 4}: mean 2, half width t(0.975, 1) * 2 = 25.41, so the low end would be negative.
    summary = summarise(load_trials(write_results(tmp_path, "s", [run(1, N + 0.0), run(2, N + 2.0)])))
    assert summary.rmse_ci.low == 0.0
    assert summary.rmse_ci.high == pytest.approx(math.sqrt(2 + 12.7062 * 2), rel=1e-4)


def test_all_saturated_gives_no_statistics(tmp_path: Path) -> None:
    summary = summarise(load_trials(write_results(tmp_path, "s", [run(seed, None) for seed in range(1, 65)])))
    assert (summary.saturated, summary.bias, summary.rmse, summary.bias_confirmed) == (64, None, None, None)


def test_summarise_rejects_mixed_true_n(tmp_path: Path) -> None:
    trials = load_trials(write_results(tmp_path, "s", [run(1, 1.0), run(2, 1.0, true_n=N + 1)]))
    with pytest.raises(ValueError, match="true_n"):
        summarise(trials)


# --- Theory ----------------------------------------------------------------------------------

def test_linear_counting_theory() -> None:
    # t = 1: e^t - t - 1 = e - 2, so bias (e - 2) / 2 and RMSE sqrt(m (e - 2)), the formulas from issue #6.
    bias, rmse = linear_counting_theory(1000, 1000)
    assert bias == pytest.approx((math.e - 2) / 2)
    assert rmse == pytest.approx(math.sqrt(1000 * (math.e - 2)))
    # Small t: e^t - t - 1 is about t^2 / 2, which exp(t) - t - 1 would round to 0.
    assert linear_counting_theory(1, 10 ** 9)[0] == pytest.approx(0.25e-18, rel=1e-6)
    assert linear_counting_theory(10 ** 6, 1) == (math.inf, math.inf)


def test_theory_uses_the_budget_not_bits_used(tmp_path: Path) -> None:
    entries = [run(1, N + 1.0, bits_used=4096), run(2, N - 1.0, bits_used=4096)]
    summary = summarise(load_trials(write_results(tmp_path, "s", entries)))
    assert (summary.predicted_bias, summary.predicted_rmse) == linear_counting_theory(N, BITS)


def test_theory_confirmed_when_prediction_is_inside_both_intervals(tmp_path: Path) -> None:
    # Predicted for n = 1000, m = 8192: bias 0.004 and RMSE 7.98. Errors {3, -3, 11, -11} give a
    # bias interval of about [-14.8, 14.8] and an RMSE interval of about [0, 12.96].
    errors = [3.0, -3.0, 11.0, -11.0]
    summary = summarise(load_trials(write_results(tmp_path, "s", [run(seed, N + e) for seed, e in enumerate(errors, 1)])))
    assert (summary.bias_confirmed, summary.rmse_confirmed) == (True, True)


def test_theory_rejected_when_prediction_is_outside_both_intervals(tmp_path: Path) -> None:
    # Errors near 100 give narrow intervals around 100, far from bias 0.004 and RMSE 7.98.
    errors = [100.0, 102.0, 98.0, 100.0]
    summary = summarise(load_trials(write_results(tmp_path, "s", [run(seed, N + e) for seed, e in enumerate(errors, 1)])))
    assert (summary.bias_confirmed, summary.rmse_confirmed) == (False, False)


def test_theory_not_checked_when_a_trial_saturated(tmp_path: Path) -> None:
    errors = [3.0, -3.0, 11.0, -11.0]
    entries = [run(seed, N + e) for seed, e in enumerate(errors, 1)] + [run(5, None)]
    summary = summarise(load_trials(write_results(tmp_path, "s", entries)))
    assert summary.bias_ci is not None
    assert (summary.bias_confirmed, summary.rmse_confirmed) == (None, None)


def test_no_theory_for_sketches_without_a_formula(tmp_path: Path) -> None:
    entries = [run(1, N + 0.0, sketch="DummySketch"), run(2, N + 0.0, sketch="DummySketch")]
    summary = summarise(load_trials(write_results(tmp_path, "d", entries)))
    assert (summary.predicted_bias, summary.predicted_rmse, summary.bias_confirmed, summary.rmse_confirmed) == (None, None, None, None)


def test_interval_contains_its_ends() -> None:
    assert Interval(0.0, 2.0).contains(2.0)
    assert not Interval(0.0, 2.0).contains(2.1)


# --- Output ----------------------------------------------------------------------------------

def test_writers(tmp_path: Path) -> None:
    entries = [run(1, N + 1.0), run(2, N - 1.0), run(1, None, bits=64), run(2, None, bits=64)]
    write_results(tmp_path, "s", entries, command="./b s.cebs --benchmark_filter=A|B")
    summaries = summarise_all(load_all_trials(tmp_path))

    write_csv(summaries, tmp_path / "out" / "summary.csv")
    with open(tmp_path / "out" / "summary.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["bits"], row["saturated"], row["bias"]) for row in rows] == [("64", "2", ""), ("8192", "0", "0.0")]

    write_markdown(summaries, tmp_path / "out" / "summary.md")
    markdown = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert markdown.startswith("### s\n")
    assert "| LinearCounting | 64 | 64 | 1000 | — | — | — | — | 2 of 2 |" in markdown
    assert "A\\|B" in markdown
