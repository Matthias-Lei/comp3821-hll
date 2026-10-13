import ctypes
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np
import numpy.typing as npt
from typing import Optional
import uuid

PARENT_DIR = Path(__file__).resolve().parent
LIBGENERATOR_PATH = PARENT_DIR / "bin" / "libgenerator.so"


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

    def generate(self, distribution: str, m: int, **params: float) -> npt.NDArray[np.int64]:
        result = self.sampler.sample(distribution, m, **params)

        if result.dtype == np.float64:

            if result.size == 0:
                return result.astype(np.int64)

            minimum, maximum = result.min(), result.max()
            if maximum - minimum == 0:
                result = np.zeros_like(result)
            else:
                result = np.floor((result - minimum) / (maximum - minimum) * (m - 1))

        return result.astype(np.int64)

@Generator.register("uuid")
class UUIDGenerator(Generator):
    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(self, n: int) -> npt.NDArray[np.void]:
        result = self.rng.bytes(n * 16)
        return np.frombuffer(result, dtype="V16").copy()

