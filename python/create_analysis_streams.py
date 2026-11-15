"""Create the synthetic streams for the report with the team's generators (generator.py).

Run from the repository root:

    python3 python/create_analysis_streams.py
    python3 python/create_analysis_streams.py --distributions poisson --parameter-sets default --sizes 76800

Follows create_test_streams.py, the source of truth for how streams are made: the same two
generators, called the same way, saved as uint64 and, for the default parameters and bins, with the
same file names. What it adds is coverage, plus one fix to how decimal draws are saved (below):

- observations: m distinct values (0 to m - 1), each repeated a number of times drawn from a
  distribution. Exactly m distinct values, so these are the accuracy streams. m follows a 1-2-5
  grid (10, 20, 50, ... up to --max-m), or exactly --sizes.
  File: <label>_observations_seed<seed>_m<m>.cebs
- distribution: nsample draws from the distribution itself, decimal draws binned to --bins levels.
  File: <label>_seed<seed>_nsample<nsample>.cebs, plus _bins<b> when b is not the default 1,024.

Every distribution runs with several parameter sets (PARAMETER_SETS): the sampler's defaults, whose
label is the bare distribution name as in create_test_streams.py, and two that model something real
(pages viewed per visit, emails per day, heights in cm, response times in ms, ...), whose label adds
the parameters, e.g. poisson-lambda=50.

Each parameter set has its own generator seed: the defaults use --seed (as create_test_streams.py
does) and the next sets --seed + 1 and --seed + 2, named in the file. The observations generator
rescales decimal draws to 1 to m repeats, so with one shared seed a set that only changes the scale
(heights in cm against the default normal) would give byte-for-byte the same stream; its own seed
gives it its own draws and its own order. Note that observations streams always hold the values
0 to m - 1, so a sketch that ignores repeats and order builds the same state from all of them.

Saving decimal draws: create_test_streams.py passes float64 draws to CEBSStream(..., UINT64), which
converts each value to a whole number, so every draw between 0 and 1 becomes 0 (each of its beta
streams has one distinct value). Here a float64 draw is saved as the uint64 with the same 8 bytes
(numpy's view, not a conversion): still a uint64 stream, but no two different draws become the same
value, and the benchmark hashes exactly those 8 bytes. Whole-number data is saved exactly as
create_test_streams.py saves it.

Observations streams grow quickly: decimal draws give each value 1 to m repeats (about m^2 / 2
values), and wide whole-number draws give hundreds of repeats each. Each stream's length is
estimated from the previous size made with the same generator (or a small pilot run), and once a
stream is estimated past --max-values, it and the larger sizes are skipped, with a message.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from generator import DEFAULT_BINS, DistributionSampler, Generator
from serialiser import CEBSObject, CEBSStream, Serialiser, StreamLocation, StreamType

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "streams"
DEFAULT_SEED = 7427369  # the data seed create_test_streams.py uses
DEFAULT_MAX_M = 1_000_000
DEFAULT_NSAMPLES = (10_000, 100_000, 1_000_000)
DEFAULT_BIN_COUNTS = (64, DEFAULT_BINS, 65_536)
DEFAULT_MAX_VALUES = 5_000_000  # 40 MB per stream
PILOT_M = 1000  # size of the pilot run used to estimate an observations stream's length
KINDS = ("observations", "distribution")
DISTRIBUTIONS = tuple(DistributionSampler.supported_distributions())

# For each distribution: the sampler's defaults first (as create_test_streams.py uses), then two
# sets that model something real, in the units noted. In an observations stream the draw is how
# often each key appears; in a distribution stream it is the value itself. Parameter names are the
# sampler's (DistributionSampler.SUPPORTED_DISTRIBUTIONS); the sampler's negative binomial counts
# trials, so its mean is k / p, and its exponential and gamma take a rate lambda (mean 1 / lambda).
PARAMETER_SETS: dict[str, tuple[dict[str, float], ...]] = {
    "bernoulli": (
        {},
        {"p": 0.02},  # whether an ad shown is clicked (2% click-through)
        {"p": 0.3},  # whether a visitor is returning (30%)
    ),
    "binomial": (
        {},
        {"n": 30, "p": 0.2},  # days a casual user logs in, out of a 30-day month (mean 6)
        {"n": 365, "p": 0.05},  # days a customer shops, out of a year (mean about 18)
    ),
    "geometric": (
        {},
        {"p": 0.3},  # pages viewed in a visit, leaving after each with chance 0.3 (mean 3.3)
        {"p": 0.05},  # posts scrolled past before one is clicked (mean 20)
    ),
    "negative_binomial": (
        {},
        {"k": 2, "p": 0.2},  # store visits per customer per year, overdispersed (mean 10)
        {"k": 5, "p": 0.05},  # API calls per client per hour (mean 100)
    ),
    "poisson": (
        {},
        {"lambda": 4},  # orders per customer per year
        {"lambda": 50},  # emails received per day
    ),
    "uniform": (
        {},
        {"a": 0, "b": 86_400},  # second of the day an event happens
        {"a": 0, "b": 1_000_000_000},  # randomly assigned 9-digit IDs
    ),
    "normal": (
        {},
        {"mu": 170, "sigma": 10},  # adult heights in cm
        {"mu": 8000, "sigma": 3000},  # steps walked per day
    ),
    "lognormal": (
        {},
        {"mu": 5.3, "sigma": 0.5},  # web response times in ms (median about 200)
        {"mu": 11, "sigma": 0.7},  # yearly incomes in dollars (median about 60,000)
    ),
    "t": (
        {},
        {"df": 3},  # daily stock returns: heavier tails than a normal
        {"df": 10},  # measurement errors estimated from a small sample
    ),
    "exponential": (
        {},
        {"lambda": 0.1},  # minutes between buses (mean 10)
        {"lambda": 0.005},  # phone call lengths in seconds (mean 200)
    ),
    "gamma": (
        {},
        {"alpha": 2, "lambda": 0.001},  # insurance claim sizes in dollars (mean 2,000)
        {"alpha": 0.5, "lambda": 0.1},  # daily rainfall in mm: mostly light, sometimes heavy (mean 5)
    ),
    "chi_squared": (
        {},
        {"df": 1},  # test statistic of a 2 x 2 contingency table
        {"df": 10},  # goodness-of-fit statistic over 11 categories
    ),
    "beta": (
        {},
        {"alpha": 2, "beta": 98},  # click-through rates of ads (mean 2%)
        {"alpha": 8, "beta": 2},  # share of an online course a student completes (mean 80%)
    ),
    "pareto": (
        {},
        {"alpha": 1.16, "lambda": 10_000},  # wealth in dollars: the 80/20 rule
        {"alpha": 1.5, "lambda": 1000},  # file sizes in bytes
    ),
}


def size_grid(min_m: int, max_m: int) -> list[int]:
    """The 1-2-5 sequence from min_m to max_m: 10, 20, 50, 100, 200, 500, ..."""
    sizes = []
    decade = 1
    while decade <= max_m:
        sizes += [step * decade for step in (1, 2, 5) if min_m <= step * decade <= max_m]
        decade *= 10
    return sizes


def sizes_to_make(min_m: int, max_m: int, sizes: list[int] | None) -> list[int]:
    """The chosen sizes if any were given, otherwise the 1-2-5 grid from min_m to max_m."""
    return sorted(set(sizes)) if sizes else size_grid(min_m, max_m)


def is_decimal(distribution: str) -> bool:
    """Whether DistributionSampler draws decimals (float64) for this distribution."""
    return DistributionSampler.SUPPORTED_DISTRIBUTIONS[distribution]["format"] == np.float64


def parameter_sets(distribution: str, which: str) -> tuple[dict[str, float], ...]:
    """All of a distribution's parameter sets, or only the defaults."""
    sets = PARAMETER_SETS[distribution]
    return sets if which == "all" else sets[:1]


def number_text(value: float) -> str:
    """1000 for 1000.0, 0.1 for 0.1: short and exact enough for a file name."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def stream_label(distribution: str, params: dict[str, float]) -> str:
    """The distribution's name, plus its parameters when they are not the defaults."""
    return "-".join([distribution, *(f"{name}={number_text(value)}" for name, value in params.items())])


def estimated_observations_length(known_length: int, known_m: int, distribution: str, m: int) -> float:
    """Scales the length of a stream already made at known_m to m: repeats per value grow with m
    for decimal draws (so the length grows as m^2) and stay about the same for whole-number draws
    (as m; they creep up a little as the smallest draw falls, so scale from a nearby size)."""
    power = 2 if is_decimal(distribution) else 1
    return known_length * (m / known_m) ** power


def as_uint64(data: np.ndarray) -> np.ndarray:
    """Float64 draws as the uint64s with the same 8 bytes, so that no two different draws become
    the same value; whole-number data unchanged."""
    return data.view(np.uint64) if data.dtype == np.float64 else data


def write_stream(data: np.ndarray, path: Path) -> None:
    """Saves the data as a uint64 CEBS stream, as create_test_streams.py does, but with decimal
    draws kept whole (as_uint64) instead of cut to whole numbers."""
    stream = CEBSObject(stream=CEBSStream(as_uint64(data), StreamType.UINT64)).to_bytes()
    Serialiser().write_stream(stream, StreamLocation.TEMP_FILE, path=path)


def set_seed(seed: int, index: int) -> int:
    """The generator seed of a distribution's index-th parameter set: the defaults (index 0) keep
    the base seed, as create_test_streams.py uses it, and every other set gets its own."""
    return seed + index


def make_observations(output: Path, seed: int, distributions: list[str], which_sets: str, sizes: list[int],
                      max_values: int) -> list[Path]:
    written = []
    for distribution in distributions:
        for index, params in enumerate(parameter_sets(distribution, which_sets)):
            label = stream_label(distribution, params)
            stream_seed = set_seed(seed, index)
            observation_generator = Generator.get_generator("observations", seed=stream_seed)
            known_m = min(PILOT_M, sizes[0])
            known_length = observation_generator.generate(distribution, known_m, **params).size
            for m in sizes:
                estimate = estimated_observations_length(known_length, known_m, distribution, m)
                if estimate > max_values:
                    print(f"skipped {label} observations from m = {m:,}: about {estimate:,.0f} values", file=sys.stderr)
                    break  # sizes only grow from here
                data = observation_generator.generate(distribution, m, **params)
                path = output / f"{label}_observations_seed{stream_seed}_m{m}.cebs"
                write_stream(data, path)
                written.append(path)
                known_m, known_length = m, data.size
                print(f"wrote {path.name}: {data.size:,} values, {m:,} distinct", file=sys.stderr)
    return written


def bin_counts_for(distribution: str, bin_counts: list[int]) -> list[int]:
    """Bins only change decimal draws, so whole-number distributions are made once, with the default."""
    return list(bin_counts) if is_decimal(distribution) else [DEFAULT_BINS]


def make_distributions(output: Path, seed: int, distributions: list[str], which_sets: str, nsamples: list[int],
                       bin_counts: list[int]) -> list[Path]:
    written = []
    for distribution in distributions:
        for bins in bin_counts_for(distribution, bin_counts):
            suffix = "" if bins == DEFAULT_BINS else f"_bins{bins}"
            for index, params in enumerate(parameter_sets(distribution, which_sets)):
                label = stream_label(distribution, params)
                stream_seed = set_seed(seed, index)
                distribution_generator = Generator.get_generator("distribution", seed=stream_seed, bins=bins)
                for nsample in nsamples:
                    data = distribution_generator.generate(distribution, nsample, **params)
                    path = output / f"{label}_seed{stream_seed}_nsample{nsample}{suffix}.cebs"
                    write_stream(data, path)
                    written.append(path)
                    print(f"wrote {path.name}: {data.size:,} values, {np.unique(data).size:,} distinct", file=sys.stderr)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for the .cebs files")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS), help="which generators to run")
    parser.add_argument("--distributions", nargs="+", choices=DISTRIBUTIONS, default=list(DISTRIBUTIONS), help="which distributions")
    parser.add_argument("--parameter-sets", choices=("all", "default"), default="all", help="every parameter set, or only the defaults")
    parser.add_argument("--min-m", type=int, default=10, help="smallest number of distinct values (observations)")
    parser.add_argument("--max-m", type=int, default=DEFAULT_MAX_M, help="largest number of distinct values (observations)")
    parser.add_argument("--sizes", nargs="+", type=int, help="exact numbers of distinct values, instead of the 1-2-5 grid")
    parser.add_argument("--nsamples", nargs="+", type=int, default=list(DEFAULT_NSAMPLES), help="stream lengths (distribution)")
    parser.add_argument("--bins", nargs="+", type=int, default=list(DEFAULT_BIN_COUNTS), help="bin counts for decimal draws (distribution)")
    parser.add_argument("--max-values", type=int, default=DEFAULT_MAX_VALUES, help="skip observations streams longer than this")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="data seed")
    return parser.parse_args(argv)


def check_args(args: argparse.Namespace) -> str | None:
    """A message for the first invalid argument, or None when all are valid."""
    if not 1 <= args.min_m <= args.max_m:
        return "need 1 <= --min-m <= --max-m"
    if args.sizes and min(args.sizes) < 1:
        return "--sizes must all be at least 1"
    if min(args.nsamples) < 1:
        return "--nsamples must all be at least 1"
    if min(args.bins) < 2:
        return "--bins must all be at least 2"
    if args.max_values < 1:
        return "--max-values must be at least 1"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problem = check_args(args)
    if problem:
        print(problem, file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    if "observations" in args.kinds:
        sizes = sizes_to_make(args.min_m, args.max_m, args.sizes)
        make_observations(args.output, args.seed, args.distributions, args.parameter_sets, sizes, args.max_values)
    if "distribution" in args.kinds:
        make_distributions(args.output, args.seed, args.distributions, args.parameter_sets, args.nsamples, args.bins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
