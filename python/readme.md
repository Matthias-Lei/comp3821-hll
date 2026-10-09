## Usage



```zsh
cd python
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make
python3 main.py
python3 create_test_streams.py
```


# Files

| File                     | Purpose                                                              |
|--------------------------|----------------------------------------------------------------------|
| `generator.py`           | Generates streams with various characteristics.                      |
| `serialiser.py`          | Writes streams to files/memory.                                      |
| `demo.py`                | Miscellaneous demonstrations/tests of available classes and methods. |
| `create_test_streams.py` | Creates test streams using the available methods.                    |



# generator.py

uhh inputs arent checked so the caller should make sure theyre valid


## DistributionSampler
`sample(distribution, nsamples, **params)`
Generate miscellaneous distributions.

| Distribution        | Default Params             |
|---------------------|----------------------------|
| `bernoulli`         | `p=0.5`                    |
| `binomial`          | `n=10`, `p=0.5`            |
| `geometric`         | `p=0.5`                    |
| `negative_binomial` | `k=10`, `p=0.5`            |
| `poisson`           | `lambda=1.0`               |
| `uniform`           | `a=0.0`, `b=1.0`           |
| `normal`            | `mu=0.0`, `sigma=1.0`      |
| `lognormal`         | `mu=0.0`, `sigma=1.0`      |
| `t`                 | `df=20`                    |
| `exponential`       | `lambda=1.0`               |
| `gamma`             | `alpha=2.0`, `lambda=1.0`  |
| `chi_squared`       | `df=20`                    |
| `beta`              | `alpha=2.0`, `beta=2.0`    |
| `pareto`            | `alpha=3.0`, `lambda=1.0`  |

Returns an `np.ndarray[np.int64]` for discrete distributions and `np.ndarray[np.float64]` for continuous distributions.


## python_test_range
`generate(start, stop, step=1)`
Integers spaced by `step` from `start` to `stop`.
Returns an `Iterable`

## c_test_range
`generate(start, stop, step=1)`
Integers spaced by `step` from `start` to `stop`.
Returns an `np.ndarray[np.uint64]`

## empty
`generate()`
Empty stream.
Returns a `list[int]`

## distribution
`generate(distribution, nsamples, **params)`
Generate samples from a distribution.
Returns an `np.ndarray[np.int64]` for discrete distributions and `np.ndarray[np.float64]` for continuous distributions.
Continuous distributions are quantised to a default of 1024 possible values.

## observations
`generate(distribution, nsamples, **params)`
Generate n samples from a distribution, with each sample occuring a number of times proportional to the result of sampling the distribution.
Returns an `np.ndarray[np.int64]`

## uuid
`generate(n)`
`n` random 16-byte values. They are random bytes, so the UUID version and variant bits are not set. Accepts a `seed` argument.
Returns an `np.ndarray` of 16-byte values.







# serialiser.py

### `CEBSHeader`
Dataclass representing a CEBS file header.
See [./cebs_spec.md](./cebs_spec.md) for more.

### `CEBSStream`
Dataclass that represents a CEBS stream.

### `StreamDestination`
Enum of destinations for serialisation.

### `Serialiser`
Take a `StreamDestination` and write it to *something* through `write_stream(data, ...)`


# demo.py
Demonstration of generators and serialisers to the terminal.


# create_test_streams.py
Creates test streams in `./data/streams`.

