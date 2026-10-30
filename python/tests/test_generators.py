import numpy as np
from scipy.stats import poisson
import hashlib

import pytest

from generator import Generator
from serialiser import CEBSObject, CEBSStream, Serialiser, StreamLocation, StreamType
from cardinality import compute_cardinality

SEED = 7427369

POISSON_TAIL_LIMIT = 1e-16

def calculate_poisson_moments(n: int, lambda_: float) -> tuple[float, float]:
    p: np.ndarray = poisson.pmf(np.arange(int(poisson.ppf(1 - POISSON_TAIL_LIMIT, lambda_)) + 1), lambda_)
    q: np.ndarray = (1 - p) ** n

    expectation = sum(1 - qk for qk in q)
    variance = sum(qk * (1 - qk) for qk in q)

    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            covariance = (1 - p[i] - p[j]) ** n - q[i] * q[j]
            variance += 2 * covariance

    return expectation, variance

def poisson_generator_test(generator: str, nsamples: int, seed: int, lambda_: float) -> None:
    generator = Generator.get_generator(generator, seed=seed)
    data = generator.generate("poisson", nsamples, lambda_=lambda_)

    stream = CEBSObject(stream=CEBSStream(data, StreamType.INT64)).to_bytes()
    serialiser = Serialiser()
    sharedmemory, cleanup = serialiser.write_stream(stream, StreamLocation.SHARED_MEMORY)
    elements, cardinality = compute_cardinality(StreamLocation.SHARED_MEMORY, sharedmemory)
    cleanup()

    expectation, variance = calculate_poisson_moments(elements, lambda_)

    print(f"{'Observed cardinality':<23}{cardinality}")
    print(f"{'Expected cardinality':<23}{expectation:.4f}")
    print(f"{'Standard deviation':<23}{np.sqrt(variance):.2f}")
    print(f"{'p-value':<23}{'too difficult to calculate...'}")



def reproducibility_test(generator: str, distribution: str, nsamples: int, seed: int, **kwargs) -> None:
    data = Generator.get_generator(generator, seed=seed).generate(distribution, nsamples, **kwargs)
    hex1 = hashlib.sha256(CEBSObject(stream=CEBSStream(data, StreamType.INT64)).to_bytes()).hexdigest().upper()

    data = Generator.get_generator(generator, seed=seed).generate(distribution, nsamples, **kwargs)
    hex2 = hashlib.sha256(CEBSObject(stream=CEBSStream(data, StreamType.INT64)).to_bytes()).hexdigest().upper()

    print(f"{'Run 1':<23}{hex1}")
    print(f"{'Run 2':<23}{hex2}")

    print(f"{'Result':<23}{'MATCH' if hex1 == hex2 else 'MISMATCH'}")



@pytest.mark.parametrize(("generator", "nsamples"), [
    ("observations", 10 ** 6),
    ("distribution", 10 ** 6),
    ("distribution", 10 ** 7),
    ("distribution", 4 * 10 ** 7),
])
def test_poisson(generator: str, nsamples: int) -> None:
    poisson_generator_test(generator, nsamples, seed=SEED, lambda_=10)


@pytest.mark.parametrize("seed", np.random.default_rng(SEED).integers(0, 2 ** 32, size=10))
def reproducibility_test_pytest(seed: int) -> None:
    reproducibility_test("observations", "poisson", 10 ** 6, seed=int(seed), lambda_=10)



# if __name__ == "__main__":
#     poisson_generator_test("observations", 10 ** 6, seed=SEED, lambda_=10)
#     print()
#     poisson_generator_test("distribution", 10 ** 6, seed=SEED, lambda_=10)
#     print()
#     poisson_generator_test("distribution", 10 ** 7, seed=SEED, lambda_=10)
#     print()
#     poisson_generator_test("distribution", 4 * 10 ** 7, seed=SEED, lambda_=10)
#     print()

#     seeds = np.random.default_rng(SEED).integers(0, 2 ** 32, size=10)
#     for seed in seeds:
#         reproducibility_test("observations", "poisson", 10 ** 6, seed=int(seed), lambda_=10)
#         print()
