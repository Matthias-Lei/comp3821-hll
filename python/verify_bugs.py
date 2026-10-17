"""Tests demonstrating bugs in the current Python branch.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from generator import Generator
from serialiser import (
    CEBSHeader,
    CEBSObject,
    CEBSStream,
    HEADER_SIZE,
    StreamType,
)

SEED = 7427369


# ---------------------------------------------------------------------------
# 1. RNG is recreated on every sample() call, so consecutive calls return
#    identical data. np.random.default_rng(seed) is deterministic.
# ---------------------------------------------------------------------------

def test_rng_advances_between_calls():
    gen = Generator.get_generator("distribution", seed=SEED)
    first = gen.generate("poisson", 100, lambda_=10)
    second = gen.generate("poisson", 100, lambda_=10)
    assert not np.array_equal(first, second), (
        "Two consecutive calls to DistributionGenerator.generate() returned "
        "identical data. DistributionSampler.sample() recreates the RNG from "
        "self.seed on every call, so it always starts from the same state. "
        "Fix: move `self.rng = np.random.default_rng(seed)` into "
        "DistributionSampler.__init__ and delete the line from sample()."
    )


# ---------------------------------------------------------------------------
# 2. len(ndarray) is the first dimension; .tobytes() serialises every element.
#    Header undercounts for multi-dimensional arrays.
# ---------------------------------------------------------------------------

def test_2d_array_header_matches_payload():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
    obj = CEBSObject(stream=CEBSStream(data, StreamType.UINT64))
    raw = obj.to_bytes()
    header = CEBSHeader.from_bytes(raw[:HEADER_SIZE])
    payload_values = (len(raw) - HEADER_SIZE) // 8
    assert header.number_of_units == payload_values, (
        f"Header says number_of_units={header.number_of_units}, but payload "
        f"contains {payload_values} values. _stream_to_bytes uses len(array), "
        f"which for a 2D array returns only the first dimension. Use "
        f"array.size instead, and reject arrays with ndim != 1 to make the "
        f"contract explicit."
    )


# ---------------------------------------------------------------------------
# 3. If header.data_start != 128, payload is still appended at offset 128.
#    File is self-inconsistent, and the C++ reader (which trusts data_start)
#    reads the wrong bytes.
# ---------------------------------------------------------------------------

def test_custom_data_start_respected():
    header = CEBSHeader(data_start=256, stream_type=StreamType.UINT64)
    obj = CEBSObject(
        header=header,
        stream=CEBSStream([1, 2, 3, 4], StreamType.UINT64),
    )
    raw = obj.to_bytes()
    assert len(raw) >= 256 + 32, (
        f"header.data_start=256 but file is only {len(raw)} bytes. "
        f"CEBSObject.to_bytes() appends payload immediately after the "
        f"{HEADER_SIZE}-byte header, ignoring data_start. Either force "
        f"data_start={HEADER_SIZE}, or pad the file to data_start before "
        f"appending the payload."
    )
    first = int.from_bytes(raw[256:264], "little")
    assert first == 1, (
        f"Reader following data_start=256 finds {first}, not 1. Payload was "
        f"written at offset {HEADER_SIZE} instead."
    )


# ---------------------------------------------------------------------------
# 4. Spec documents a `unique_elements` field at offset 40; code doesn't
#    write it. Bytes 40-47 are zeros (part of the reserved region).
# ---------------------------------------------------------------------------

def test_unique_elements_field_written():
    data = [5, 5, 5, 7, 7, 9]  # 3 distinct values
    obj = CEBSObject(stream=CEBSStream(data, StreamType.UINT64))
    raw = obj.to_bytes()
    value = int.from_bytes(raw[40:48], "little")
    assert value == 3, (
        f"cebs_spec.md says unique_elements is at offset 40. Found {value} "
        f"instead of 3. CEBSHeader has no unique_elements field; bytes 40-47 "
        f"are part of the reserved region. Either add the field to the header "
        f"and populate it during serialisation, or remove it from the spec."
    )


# ---------------------------------------------------------------------------
# 5. Spec table: unique_elements (8 @ 40) + reserved (88 @ 48) => 136 bytes.
#    But data_start = 128. Reserved overlaps data.
# ---------------------------------------------------------------------------

def test_spec_header_fits_within_data_start():
    spec_header_end = 48 + 88  # reserved ends here per spec table
    assert spec_header_end <= HEADER_SIZE, (
        f"Spec places end of header at {spec_header_end} (reserved ends at "
        f"48 + 88), but data_start={HEADER_SIZE}. Bytes "
        f"{HEADER_SIZE}-{spec_header_end - 1} of the spec's reserved region "
        f"overlap the data section. Reserved must be 80 bytes, not 88, or "
        f"data_start must be 136."
    )


# ---------------------------------------------------------------------------
# 6. CEBSHeader.to_bytes mutates self when called with override arguments.
# ---------------------------------------------------------------------------

def test_to_bytes_does_not_mutate_header():
    header = CEBSHeader()
    original = header.data_start
    header.to_bytes(data_start=256)
    assert header.data_start == original, (
        f"header.data_start changed from {original} to {header.data_start} "
        f"after calling to_bytes(data_start=256). Mutating the receiver is "
        f"surprising; use a local variable or return a new header."
    )


# ---------------------------------------------------------------------------
# 7. generator.c calls exit(1) on invalid input, killing the Python process
#    with no catchable exception.
# ---------------------------------------------------------------------------

def test_c_generator_step_zero_raises_exception():
    import subprocess

    lib = HERE / "bin" / "libgenerator.so"
    if not lib.exists():
        print(f"      skipping: {lib} not built; run `make -C python`")
        return

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from generator import Generator\n"
        "try:\n"
        "    Generator.get_generator('c_test_range').generate(0, 10, step=0)\n"
        "    print('NO_EXCEPTION')\n"
        "except Exception as e:\n"
        "    print('CAUGHT', type(e).__name__)\n"
    ) % str(HERE)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert "CAUGHT" in result.stdout, (
        f"generate(0, 10, step=0) exited with code {result.returncode} "
        f"(stdout={result.stdout!r}, stderr={result.stderr!r}) instead of "
        f"raising a Python exception. Replace `exit(1)` in generator.c with "
        f"`errno = EINVAL; return (Array){{NULL, 0}};` and raise ValueError "
        f"from the ctypes wrapper when data == NULL."
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}")
            for line in str(e).splitlines():
                print(f"      {line}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
