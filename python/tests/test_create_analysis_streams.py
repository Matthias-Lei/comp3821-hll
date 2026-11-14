"""Tests for create_analysis_streams.py."""
from pathlib import Path

import numpy as np
import pytest

from create_analysis_streams import (
    DISTRIBUTIONS,
    PARAMETER_SETS,
    PILOT_M,
    as_uint64,
    bin_counts_for,
    estimated_observations_length,
    is_decimal,
    main,
    parameter_sets,
    set_seed,
    size_grid,
    sizes_to_make,
    stream_label,
    write_stream,
)
from generator import DEFAULT_BINS, DistributionSampler, Generator
from serialiser import HEADER_SIZE, CEBSHeader, CEBSObject, CEBSStream, StreamType

SEED = 7427369


def read_back(path: Path) -> tuple[CEBSHeader, np.ndarray]:
    raw = path.read_bytes()
    header = CEBSHeader.from_bytes(raw[:HEADER_SIZE])
    return header, np.frombuffer(raw, dtype="<u8", offset=header.data_start, count=header.number_of_units)


def as_create_test_streams_writes(data: np.ndarray) -> bytes:
    """The bytes create_test_streams.py writes for this data."""
    return CEBSObject(stream=CEBSStream(data, StreamType.UINT64)).to_bytes()


def distinct(path: Path) -> int:
    return np.unique(read_back(path)[1]).size


def test_size_grid_is_one_two_five_within_the_bounds() -> None:
    assert size_grid(10, 1000) == [10, 20, 50, 100, 200, 500, 1000]
    assert size_grid(1, 7) == [1, 2, 5]
    assert size_grid(15, 60) == [20, 50]


def test_chosen_sizes_replace_the_grid() -> None:
    assert sizes_to_make(10, 100, None) == [10, 20, 50, 100]
    assert sizes_to_make(10, 100, [76_800, 500, 76_800]) == [500, 76_800]


def test_every_team_distribution_has_parameter_sets_starting_with_the_defaults() -> None:
    assert DISTRIBUTIONS == tuple(DistributionSampler.supported_distributions())
    assert set(PARAMETER_SETS) == set(DISTRIBUTIONS)
    assert all(sets[0] == {} and len(sets) == 3 for sets in PARAMETER_SETS.values())


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_every_parameter_set_is_accepted_by_the_sampler(distribution: str) -> None:
    for params in PARAMETER_SETS[distribution]:
        assert DistributionSampler(SEED).sample(distribution, 100, **params).size == 100


def test_default_parameter_sets_only() -> None:
    assert parameter_sets("poisson", "default") == ({},)
    assert len(parameter_sets("poisson", "all")) == 3


def test_defaults_keep_the_team_seed_and_other_sets_get_their_own() -> None:
    assert [set_seed(SEED, index) for index in range(3)] == [SEED, SEED + 1, SEED + 2]


@pytest.mark.parametrize(("distribution", "params", "label"), [
    ("poisson", {}, "poisson"),
    ("poisson", {"lambda": 1000}, "poisson-lambda=1000"),
    ("bernoulli", {"p": 0.1}, "bernoulli-p=0.1"),
    ("uniform", {"a": -1_000_000, "b": 1_000_000}, "uniform-a=-1000000-b=1000000"),
])
def test_stream_labels_name_non_default_parameters(distribution: str, params: dict, label: str) -> None:
    assert stream_label(distribution, params) == label


@pytest.mark.parametrize(("distribution", "params", "known_m", "m"), [
    ("uniform", {}, PILOT_M, 4000), ("normal", {"sigma": 100}, PILOT_M, 3000),
    ("poisson", {}, PILOT_M, 20_000), ("poisson", {"lambda": 1000}, 10_000, 20_000), ("poisson", {"lambda": 1000}, 20_000, 50_000),
])
def test_estimate_from_a_nearby_size_is_close_to_the_real_length(distribution: str, params: dict, known_m: int, m: int) -> None:
    generator = Generator.get_generator("observations", seed=SEED)
    known = generator.generate(distribution, known_m, **params).size
    actual = generator.generate(distribution, m, **params).size
    assert estimated_observations_length(known, known_m, distribution, m) == pytest.approx(actual, rel=0.15)


def test_bins_apply_to_decimal_distributions_only() -> None:
    assert is_decimal("normal") and not is_decimal("poisson")
    assert bin_counts_for("normal", [64, 65_536]) == [64, 65_536]
    assert bin_counts_for("poisson", [64, 65_536]) == [DEFAULT_BINS]


def test_decimal_draws_keep_their_8_bytes_and_whole_numbers_are_unchanged() -> None:
    draws = np.array([0.25, 0.75, 0.25, 170.5])
    assert as_uint64(draws).dtype == np.uint64
    assert np.array_equal(as_uint64(draws).view(np.float64), draws)
    assert np.unique(as_uint64(draws)).size == 3  # converting to whole numbers would give 2: {0, 170}
    counts = np.array([1, 5, 2], dtype=np.int64)
    assert as_uint64(counts) is counts


def test_whole_number_streams_match_create_test_streams_byte_for_byte(tmp_path: Path) -> None:
    data = Generator.get_generator("distribution", seed=SEED).generate("poisson", 10_000)
    path = tmp_path / "s.cebs"

    write_stream(data, path)

    assert path.read_bytes() == as_create_test_streams_writes(data)


@pytest.mark.parametrize("distribution", ["uniform", "normal", "beta"])
def test_decimal_streams_hold_every_draw(distribution: str, tmp_path: Path) -> None:
    draws = Generator.get_generator("distribution", seed=SEED).generate(distribution, 10_000)
    path = tmp_path / "s.cebs"

    write_stream(draws, path)

    header, values = read_back(path)
    assert header.stream_type == StreamType.UINT64
    assert np.array_equal(values.view(np.float64), draws)
    assert np.unique(values).size == np.unique(draws).size > 500  # create_test_streams.py's beta stream has 1


def test_default_streams_match_create_test_streams_names(tmp_path: Path) -> None:
    observations = Generator.get_generator("observations", seed=SEED).generate("gamma", 1000)
    draws = Generator.get_generator("distribution", seed=SEED).generate("gamma", 10_000)

    main(["--output", str(tmp_path), "--distributions", "gamma", "--parameter-sets", "default", "--sizes", "1000",
          "--nsamples", "10000", "--bins", str(DEFAULT_BINS)])

    assert sorted(path.name for path in tmp_path.iterdir()) == [f"gamma_observations_seed{SEED}_m1000.cebs", f"gamma_seed{SEED}_nsample10000.cebs"]
    assert (tmp_path / f"gamma_observations_seed{SEED}_m1000.cebs").read_bytes() == as_create_test_streams_writes(observations)
    assert np.array_equal(read_back(tmp_path / f"gamma_seed{SEED}_nsample10000.cebs")[1].view(np.float64), draws)


def test_observations_have_exactly_m_distinct_values_for_every_parameter_set(tmp_path: Path) -> None:
    assert main(["--output", str(tmp_path), "--kinds", "observations", "--distributions", "poisson", "--sizes", "100", "1000"]) == 0

    for label, seed in (("poisson", SEED), ("poisson-lambda=4", SEED + 1), ("poisson-lambda=50", SEED + 2)):
        for m in (100, 1000):
            header, values = read_back(tmp_path / f"{label}_observations_seed{seed}_m{m}.cebs")
            assert header.stream_type == StreamType.UINT64
            assert np.unique(values).size == m


def test_every_parameter_set_gets_its_own_observations_stream(tmp_path: Path) -> None:
    # The observations generator rescales decimal draws, so heights in cm repeat like the default
    # normal; their own seed still gives them their own draws and order.
    assert main(["--output", str(tmp_path), "--kinds", "observations", "--distributions", "normal", "--sizes", "100"]) == 0

    streams = [(tmp_path / f"{label}_observations_seed{seed}_m100.cebs").read_bytes()
               for label, seed in (("normal", SEED), ("normal-mu=170-sigma=10", SEED + 1), ("normal-mu=8000-sigma=3000", SEED + 2))]
    assert len(set(streams)) == 3


def test_long_observation_streams_are_not_written(tmp_path: Path) -> None:
    # uniform at m = 5000 is about 12.5 million values; poisson with lambda = 50 has about
    # 20 repeats per value, so 5000 values make about 100,000.
    assert main(["--output", str(tmp_path), "--kinds", "observations", "--distributions", "uniform", "poisson",
                 "--sizes", "100", "5000", "--max-values", "50000"]) == 0

    names = {path.name for path in tmp_path.iterdir()}
    assert f"uniform_observations_seed{SEED}_m100.cebs" in names
    assert f"uniform_observations_seed{SEED}_m5000.cebs" not in names
    assert f"poisson_observations_seed{SEED}_m5000.cebs" in names
    assert f"poisson-lambda=50_observations_seed{SEED + 2}_m100.cebs" in names
    assert f"poisson-lambda=50_observations_seed{SEED + 2}_m5000.cebs" not in names


def test_bins_set_the_number_of_distinct_draws(tmp_path: Path) -> None:
    assert main(["--output", str(tmp_path), "--kinds", "distribution", "--distributions", "normal", "beta", "poisson",
                 "--nsamples", "100000", "--bins", "64", "1024", "65536"]) == 0

    for label, seed in (("normal", SEED), ("normal-mu=8000-sigma=3000", SEED + 2), ("beta", SEED), ("beta-alpha=2-beta=98", SEED + 1)):
        assert distinct(tmp_path / f"{label}_seed{seed}_nsample100000_bins64.cebs") <= 64
        assert 500 < distinct(tmp_path / f"{label}_seed{seed}_nsample100000.cebs") <= 1024
        assert distinct(tmp_path / f"{label}_seed{seed}_nsample100000_bins65536.cebs") > 5000
    # Whole-number distributions are made once, with no bins suffix.
    assert f"poisson_seed{SEED}_nsample100000_bins64.cebs" not in {path.name for path in tmp_path.iterdir()}
    assert distinct(tmp_path / f"poisson-lambda=50_seed{SEED + 2}_nsample100000.cebs") > 30


def test_streams_are_reproducible(tmp_path: Path) -> None:
    for run in ("a", "b"):
        main(["--output", str(tmp_path / run), "--kinds", "observations", "--distributions", "geometric", "--sizes", "500"])
    name = f"geometric-p=0.3_observations_seed{SEED + 1}_m500.cebs"
    assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


@pytest.mark.parametrize("argv", [["--min-m", "0"], ["--min-m", "50", "--max-m", "10"], ["--sizes", "0"],
                                  ["--nsamples", "0"], ["--bins", "1"], ["--max-values", "0"]])
def test_invalid_arguments_are_refused(argv: list[str], tmp_path: Path) -> None:
    assert main(["--output", str(tmp_path), *argv]) == 1
