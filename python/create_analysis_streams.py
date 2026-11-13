"""Create the streams the report's experiments need, with an exact, chosen number of distinct values.

Run from the repository root:

    python3 python/create_analysis_streams.py
    python3 python/create_analysis_streams.py --max-n 3000000 --repeats 4

The proposal tests every sketch "varying the number of unique items from very small counts to very
large streams", on "random 64-bit identifiers with duplicates chosen uniformly at random", and
repeats the runs "with a skewed distribution". The streams from create_test_streams.py stop at about
6,400 distinct values (so no bitmap ever fills), leave gaps between sizes, and draw their values from
one shared pool (so their errors are correlated). These streams fix that:

- uniform_ids_n<N>:  N distinct random 64-bit identifiers, each repeated uniformly at random.
- skewed_ids_n<N>:   the same kind of identifiers with Zipf-distributed repeats: a few values appear
                     very often, most only once. Sketches ignore repeats, so the results should match
                     the uniform stream with the same N.
- sequential_n<N>:   the integers 0 to N - 1, as real identifiers often are, to check that the hash
                     scrambles structured input as well as random input.

N follows a 1-2-5 grid (10, 20, 50, 100, ... up to --max-n), so the load n/m covers small streams,
the theory's middle range and saturation at every memory budget. Every stream has --repeats x N
values, and every stream uses its own random identifiers, so results on different streams are
independent. Streams are written to data/streams/ in the CEBS format.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from serialiser import CEBSObject, CEBSStream, Serialiser, StreamLocation, StreamType

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "streams"
DEFAULT_SEED = 7427369  # the data seed create_test_streams.py uses
DEFAULT_MAX_N = 1_000_000
DEFAULT_REPEATS = 2
ZIPF_EXPONENT = 1.1  # how strongly skewed_ids repeats its most common values
KINDS = ("uniform", "skewed", "sequential")


def size_grid(min_n: int, max_n: int) -> list[int]:
    """The 1-2-5 sequence from min_n to max_n: 10, 20, 50, 100, 200, 500, ..."""
    sizes = []
    decade = 1
    while decade <= max_n:
        sizes += [step * decade for step in (1, 2, 5) if min_n <= step * decade <= max_n]
        decade *= 10
    return sizes


def distinct_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """n distinct random 64-bit identifiers, in random order."""
    ids = np.empty(0, dtype=np.uint64)
    while ids.size < n:
        draw = rng.integers(0, np.iinfo(np.uint64).max, size=n - ids.size, dtype=np.uint64, endpoint=True)
        ids = np.unique(np.concatenate([ids, draw]))
    rng.shuffle(ids)
    return ids


def with_repeats(rng: np.random.Generator, ids: np.ndarray, total: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Every identifier once, plus total - len(ids) extra copies drawn with the given weights
    (uniform when None), shuffled together. The stream keeps exactly len(ids) distinct values."""
    extra = rng.choice(ids, size=total - ids.size, p=weights) if total > ids.size else np.empty(0, dtype=ids.dtype)
    stream = np.concatenate([ids, extra])
    rng.shuffle(stream)
    return stream


def zipf_weights(n: int, exponent: float = ZIPF_EXPONENT) -> np.ndarray:
    """Probability of drawing the identifier of rank r is proportional to 1 / r^exponent."""
    weights = 1.0 / np.arange(1, n + 1, dtype=np.float64) ** exponent
    return weights / weights.sum()


def uniform_stream(rng: np.random.Generator, n: int, repeats: int) -> np.ndarray:
    return with_repeats(rng, distinct_ids(rng, n), repeats * n)


def skewed_stream(rng: np.random.Generator, n: int, repeats: int) -> np.ndarray:
    return with_repeats(rng, distinct_ids(rng, n), repeats * n, zipf_weights(n))


def sequential_stream(rng: np.random.Generator, n: int, repeats: int) -> np.ndarray:
    return with_repeats(rng, np.arange(n, dtype=np.uint64), repeats * n)


STREAMS = {
    "uniform": ("uniform_ids_n{n}_seed{seed}", uniform_stream),
    "skewed": ("skewed_ids_n{n}_seed{seed}", skewed_stream),
    "sequential": ("sequential_n{n}_seed{seed}", sequential_stream),
}


def stream_rng(seed: int, kind: str, n: int) -> np.random.Generator:
    """A generator of its own for every stream, so each stream is reproducible on its own and no
    two streams share identifiers."""
    return np.random.default_rng([seed, KINDS.index(kind), n])


def write_stream(data: np.ndarray, path: Path) -> None:
    stream = CEBSObject(stream=CEBSStream(data, StreamType.UINT64)).to_bytes()
    Serialiser().write_stream(stream, StreamLocation.TEMP_FILE, path=path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for the .cebs files")
    parser.add_argument("--min-n", type=int, default=10, help="smallest number of distinct values")
    parser.add_argument("--max-n", type=int, default=DEFAULT_MAX_N, help="largest number of distinct values")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="stream length as a multiple of n")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="data seed")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS), help="which kinds of stream to make")
    args = parser.parse_args()

    if not 1 <= args.min_n <= args.max_n:
        print("need 1 <= --min-n <= --max-n", file=sys.stderr)
        return 1
    if args.repeats < 1:
        print("--repeats must be at least 1", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    sizes = size_grid(args.min_n, args.max_n)
    for kind in args.kinds:
        name, make = STREAMS[kind]
        for n in sizes:
            data = make(stream_rng(args.seed, kind, n), n, args.repeats)
            path = args.output / f"{name.format(n=n, seed=args.seed)}.cebs"
            write_stream(data, path)
            print(f"wrote {path.name}: {data.size:,} values, {n:,} distinct", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
