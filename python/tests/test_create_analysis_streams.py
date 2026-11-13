"""Tests for create_analysis_streams.py."""
from pathlib import Path

import numpy as np
import pytest

from create_analysis_streams import (
    distinct_ids,
    sequential_stream,
    size_grid,
    skewed_stream,
    stream_rng,
    uniform_stream,
    write_stream,
    zipf_weights,
)
from serialiser import HEADER_SIZE, CEBSHeader, StreamType


def rng(seed: int = 1) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_size_grid_is_one_two_five_within_the_bounds() -> None:
    assert size_grid(10, 1000) == [10, 20, 50, 100, 200, 500, 1000]
    assert size_grid(1, 7) == [1, 2, 5]
    assert size_grid(15, 60) == [20, 50]


def test_distinct_ids_are_distinct_64_bit_values() -> None:
    ids = distinct_ids(rng(), 10_000)
    assert ids.dtype == np.uint64
    assert np.unique(ids).size == 10_000
    assert ids.max() > np.uint64(2 ** 63)


@pytest.mark.parametrize("make", [uniform_stream, skewed_stream, sequential_stream])
@pytest.mark.parametrize(("n", "repeats"), [(1, 1), (10, 1), (1000, 2), (1000, 5)])
def test_streams_have_exactly_n_distinct_values_and_repeats_times_n_values(make, n: int, repeats: int) -> None:
    stream = make(rng(), n, repeats)
    assert stream.dtype == np.uint64
    assert stream.size == repeats * n
    assert np.unique(stream).size == n


def test_sequential_stream_holds_zero_to_n_minus_one() -> None:
    assert set(sequential_stream(rng(), 100, 3).tolist()) == set(range(100))


def test_uniform_repeats_are_spread_evenly() -> None:
    # 100,000 extra copies over 1,000 identifiers: about 101 copies each, never extreme.
    _, counts = np.unique(uniform_stream(rng(), 1000, 101), return_counts=True)
    assert counts.max() < 2 * np.median(counts)


def test_skewed_repeats_favour_a_few_values() -> None:
    _, counts = np.unique(skewed_stream(rng(), 1000, 10), return_counts=True)
    assert counts.max() > 50 * np.median(counts)


def test_zipf_weights_sum_to_one_and_decrease() -> None:
    weights = zipf_weights(100)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights) < 0)


def test_streams_are_reproducible_and_independent() -> None:
    first = uniform_stream(stream_rng(7427369, "uniform", 1000), 1000, 2)
    again = uniform_stream(stream_rng(7427369, "uniform", 1000), 1000, 2)
    skewed = skewed_stream(stream_rng(7427369, "skewed", 1000), 1000, 2)
    bigger = uniform_stream(stream_rng(7427369, "uniform", 2000), 2000, 2)

    assert np.array_equal(first, again)
    assert not np.intersect1d(first, skewed).size
    assert not np.intersect1d(first, bigger).size


def test_written_file_is_a_uint64_cebs_stream(tmp_path: Path) -> None:
    data = uniform_stream(rng(), 500, 2)
    path = tmp_path / "s.cebs"

    write_stream(data, path)

    raw = path.read_bytes()
    header = CEBSHeader.from_bytes(raw[:HEADER_SIZE])
    assert (header.stream_type, header.number_of_units, header.bytes_per_unit) == (StreamType.UINT64, 1000, 8)
    assert np.array_equal(np.frombuffer(raw, dtype="<u8", offset=header.data_start, count=header.number_of_units), data)
