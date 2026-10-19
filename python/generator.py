import ctypes
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np
import numpy.typing as npt
from typing import Optional
import uuid

import re
import subprocess
import sys

PARENT_DIR = Path(__file__).resolve().parent
LIBGENERATOR_PATH = PARENT_DIR / "bin" / "libgenerator.so"


def _hash(data: bytes) -> int:
    import xxhash
    return xxhash.xxh64_intdigest(data)



class Generator(ABC):
    _registry: dict[str, type["Generator"]] = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_generator(cls, name: str, **kwargs):
        return cls._registry[name](**kwargs)
    
    @abstractmethod
    def generate(self, *arguments: object) -> list[int] | Iterable[int] | np.ndarray[np.int64]:
        raise NotImplementedError()


@Generator.register("python_test_range")
class PythonTestRangeGenerator(Generator):
    def generate(self, start: int, stop: int, step: Optional[int] = 1) -> Iterable[int]:
        return range(start, stop, step)


@Generator.register("c_test_range")
class CTestRangeGenerator(Generator):
    def __init__(self) -> None:
        self.cGenerator = ctypes.CDLL(str(LIBGENERATOR_PATH))

        class CArray(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("length", ctypes.c_uint64),
            ]

        self.CArray = CArray

        self.cGenerator.generate_range.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64] # uint64_t, uint64_t, uint64_t
        self.cGenerator.generate_range.restype = self.CArray # CArray

        self.cGenerator.free_range.argtypes = [self.CArray] # CArray
        self.cGenerator.free_range.restype = None # void

    def generate(self, start: int, stop: int, step: int = 1) -> npt.NDArray[np.uint64]:
        cRange = self.cGenerator.generate_range(start, stop, step)

        data = ctypes.cast(cRange.data, ctypes.POINTER(ctypes.c_uint64))
        result = np.ctypeslib.as_array(data, shape=(cRange.length,)).copy()
        self.cGenerator.free_range(cRange)

        return result

@Generator.register("empty")
class EmptyGenerator(Generator):
    def generate(self) -> list[int]:
        return []




class DistributionSampler:
    SUPPORTED_DISTRIBUTIONS = {
        "bernoulli": {"method": "_generate_bernoulli", "params": {"p": 0.5}, "format": np.int64},
        "binomial": {"method": "_generate_binomial", "params": {"n": 10, "p": 0.5}, "format": np.int64},
        "geometric": {"method": "_generate_geometric", "params": {"p": 0.5}, "format": np.int64},
        "negative_binomial": {"method": "_generate_negative_binomial", "params": {"k": 10, "p": 0.5}, "format": np.int64},
        "poisson": {"method": "_generate_poisson", "params": {"lambda": 1.0}, "format": np.int64},
        "uniform": {"method": "_generate_uniform", "params": {"a": 0.0, "b": 1.0}, "format": np.float64},
        "normal": {"method": "_generate_normal", "params": {"mu": 0.0, "sigma": 1.0}, "format": np.float64},
        "lognormal": {"method": "_generate_lognormal", "params": {"mu": 0.0, "sigma": 1.0}, "format": np.float64},
        "t": {"method": "_generate_t", "params": {"df": 20}, "format": np.float64},
        "exponential": {"method": "_generate_exponential", "params": {"lambda": 1.0}, "format": np.float64},
        "gamma": {"method": "_generate_gamma", "params": {"alpha": 2.0, "lambda": 1.0}, "format": np.float64},
        "chi_squared": {"method": "_generate_chi_squared", "params": {"df": 20}, "format": np.float64},
        "beta": {"method": "_generate_beta", "params": {"alpha": 2.0, "beta": 2.0}, "format": np.float64},
        "pareto": {"method": "_generate_pareto", "params": {"alpha": 3.0, "lambda": 1.0}, "format": np.float64},
    }

    PARAMETER_REBINDS = {
        "lambda": "lambda_",
    }

    def __init__(self, seed: int) -> None:
        self.seed = seed

    @classmethod
    def supported_distributions(cls) -> list[str]:
        return list(cls.SUPPORTED_DISTRIBUTIONS.keys())

    def _parse_parameters(self, distribution: str, **params: float) -> dict:
        defaults = self.SUPPORTED_DISTRIBUTIONS[distribution]["params"]
        rebinds = {value: key for key, value in self.PARAMETER_REBINDS.items()}
        params = {rebinds.get(key, key): value for key, value in params.items()}
        for key in params:
            if key not in defaults:
                raise ValueError(f"invalid parameter {key}")
        return defaults | params

    def sample(self, distribution: str, nsamples: int, **params: float) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
        self.rng = np.random.default_rng(self.seed)

        distribution_config = self.SUPPORTED_DISTRIBUTIONS[distribution]
        parameters = {self.PARAMETER_REBINDS.get(key, key): value for key, value in self._parse_parameters(distribution, **params).items()}

        generator = getattr(self, distribution_config["method"])
        result = generator(nsamples=nsamples, **parameters)

        return result.astype(distribution_config["format"])

    def _generate_bernoulli(self, nsamples: int, p: float) -> npt.NDArray[np.int64]:
        return self.rng.binomial(1, p, size=nsamples).astype(np.int64)

    def _generate_binomial(self, nsamples: int, n: int, p: float) -> npt.NDArray[np.int64]:
        return self.rng.binomial(n, p, size=nsamples).astype(np.int64)

    def _generate_geometric(self, nsamples: int, p: float) -> npt.NDArray[np.int64]:
        return self.rng.geometric(p, size=nsamples).astype(np.int64)

    def _generate_negative_binomial(self, nsamples: int, k: int, p: float) -> npt.NDArray[np.int64]:
        return (self.rng.negative_binomial(k, p, size=nsamples) + k).astype(np.int64)

    def _generate_poisson(self, nsamples: int, lambda_: float) -> npt.NDArray[np.int64]:
        return self.rng.poisson(lambda_, size=nsamples).astype(np.int64)

    def _generate_uniform(self, nsamples: int, a: float, b: float) -> npt.NDArray[np.float64]:
        return self.rng.uniform(a, b, size=nsamples)

    def _generate_normal(self, nsamples: int, mu: float, sigma: float) -> npt.NDArray[np.float64]:
        return self.rng.normal(mu, sigma, size=nsamples)

    def _generate_lognormal(self, nsamples: int, mu: float, sigma: float) -> npt.NDArray[np.float64]:
        return self.rng.lognormal(mu, sigma, size=nsamples)

    def _generate_t(self, nsamples: int, df: int) -> npt.NDArray[np.float64]:
        return self.rng.standard_t(df, size=nsamples)

    def _generate_exponential(self, nsamples: int, lambda_: float) -> npt.NDArray[np.float64]:
        return self.rng.exponential(1 / lambda_, size=nsamples)

    def _generate_gamma(self, nsamples: int, alpha: float, lambda_: float) -> npt.NDArray[np.float64]:
        return self.rng.gamma(alpha, 1 / lambda_, size=nsamples)

    def _generate_chi_squared(self, nsamples: int, df: int) -> npt.NDArray[np.float64]:
        return self.rng.chisquare(df, size=nsamples)

    def _generate_beta(self, nsamples: int, alpha: float, beta: float) -> npt.NDArray[np.float64]:
        return self.rng.beta(alpha, beta, size=nsamples)

    def _generate_pareto(self, nsamples: int, alpha: float, lambda_: float) -> npt.NDArray[np.float64]:
        return lambda_ * self.rng.pareto(alpha, size=nsamples)



DEFAULT_SEED = 7427369
DEFAULT_BINS = 1024

@Generator.register("distribution")
class DistributionGenerator(Generator):
    def __init__(self, seed: int = DEFAULT_SEED, bins: int = DEFAULT_BINS) -> None:
        self.sampler = DistributionSampler(seed)
        self.bins = bins

    def generate(self, distribution: str, nsamples: int, **params: float) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
        result = self.sampler.sample(distribution, nsamples, **params)

        if result.dtype == np.float64:
            minimum, span = result.min(), result.max() - result.min()
            result = np.floor((result - minimum) / span * (self.bins - 1)) / (self.bins - 1) * span + minimum

        return result


@Generator.register("observations")
class ObservationsGenerator(Generator):
    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.sampler = DistributionSampler(seed)
        self.seed = seed

    def generate(self, distribution: str, m: int, **params: float) -> npt.NDArray[np.int64]:
        repeats = self.sampler.sample(distribution, m, **params)
        if repeats.size == 0:
             return np.empty(0, dtype=np.int64)
        if repeats.dtype == np.float64:
            minimum, span = repeats.min(), repeats.max() - repeats.min()
            repeats = np.ones(m, dtype=np.int64) if span == 0 else np.floor((repeats - minimum) / span * (m - 1)).astype(np.int64) + 1
        else:
            repeats = (repeats - repeats.min()).astype(np.int64) + 1

        result = np.repeat(np.arange(m, dtype=np.int64), repeats)
        np.random.default_rng([self.seed, 1]).shuffle(result)
        return result

@Generator.register("uuid")
class UUIDGenerator(Generator):
    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(self, n: int) -> npt.NDArray[np.void]:
        result = self.rng.bytes(n * 16)
        return np.frombuffer(result, dtype="V16").copy()





DEFAULT_SQL_DIRECTORY = PARENT_DIR.parent / "data" / "sql"
DEFAULT_FETCH_SIZE = 2 ** 24




def _setup_directory(directory: Path, make_directories: bool = False) -> None:
    if not directory.exists():
        if not make_directories:
            if input(f"Create directories to {directory}? (y/N) ").lower() != "y":
                sys.exit()
        directory.mkdir(parents=True, exist_ok=True)


class SQLGenerator(Generator):
    directory: Path = DEFAULT_SQL_DIRECTORY

    def __init__(self, make_directories: bool = False) -> None:
        self.make_directories = make_directories
        self.connection = None
        self.repository = None

    def setup(self) -> None:
        import duckdb
        _setup_directory(self.directory, self.make_directories)
        self.connection = duckdb.connect(str(self.directory / self.DATABASE_NAME))
        self._setup()

    def generate(self, query: str | Path) -> Iterable[npt.NDArray[np.uint64]]:
        query = query.read_text() if isinstance(query, Path) else query
        query_result = self.connection.execute(query)

        while batch := query_result.fetchmany(DEFAULT_FETCH_SIZE):
            yield np.fromiter((_hash(str(row).encode()) for row in batch), dtype=np.uint64, count=len(batch))


@Generator.register("sql_job")
class JOBQueryGenerator(SQLGenerator):
    DATABASE_NAME = "job.duckdb"

    def _setup(self) -> None:
        if not self.connection.execute("SHOW TABLES").fetchall():
            self._setup_database()

        self.repository = self._setup_repository()

    def _setup_database(self) -> None:
        subprocess.run(["bash", str(PARENT_DIR / "scripts" / "job_setup_database.sh"), "--directory", str(self.directory)], check=True)

        self.connection.begin()
        self.connection.execute((self.directory / "schema.sql").read_text())
        self._import_csv()
        self.connection.commit()

    def _setup_repository(self) -> Path:
        subprocess.run(["bash", str(PARENT_DIR / "scripts" / "job_setup_repository.sh"), "--directory", str(self.directory)], check=True)
        return self.directory / "join-order-benchmark"

    def _import_csv(self) -> None:
        for path in sorted((self.directory / "imdb").rglob("*.csv")):
            self.connection.execute(f"COPY {path.stem} FROM '{path}' (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\', NULLSTR '', HEADER FALSE)")


    def queries(self) -> dict[str, str]:
        queries = {}

        import re

        for path in sorted(p for p in self.repository.glob("*.sql") if re.fullmatch(r"\d{1,2}[a-z]\.sql", p.name)):
            query = path.read_text()

            select_list, remainder = query.split("FROM", maxsplit=1)
            expression = select_list.removeprefix("SELECT").strip() \
                                    .split(" AS ", maxsplit=1)[0].strip() \
                                    .removeprefix("MIN(").removesuffix(")")
            remainder = re.sub(r"\bat\b", '"at"', remainder)
            expression = re.sub(r"\bat\b", '"at"', expression)

            queries[f"query_{path.stem}"] = f"SELECT {expression} FROM {remainder}"

        return queries

