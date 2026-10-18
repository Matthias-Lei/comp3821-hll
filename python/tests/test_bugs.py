import sys
from pathlib import Path

import numpy as np
import pytest

from serialiser import CEBSHeader, CEBSObject, CEBSStream, HEADER_SIZE, StreamType


@pytest.mark.parametrize("data", [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64), [[1, 2, 3], [4, 5, 6]]])
def test_multidimensional_array(data) -> None:
    with pytest.raises(ValueError):
        CEBSStream(data, StreamType.UINT64)


def test_custom_data_start_respected() -> None:
    header = CEBSHeader(data_start=256, stream_type=StreamType.UINT64)
    obj = CEBSObject(header=header, stream=CEBSStream([1, 2, 3, 4], StreamType.UINT64))

    raw = obj.to_bytes()

    assert len(raw) >= 256 + 32, f"header.data_start=256 but file is only {len(raw)} bytes"
    assert int.from_bytes(raw[256:264], "little") == 1, f"expected first value at offset 256 to be 1, found {first}"


def test_spec_header_conformance() -> None:
    header = CEBSHeader().to_bytes()
    assert len(header) == HEADER_SIZE
    assert len(header[40:]) == 88


def test_to_bytes_mutate_header() -> None:
    header = CEBSHeader()
    original = header.data_start

    header.to_bytes(data_start=256)

    assert header.data_start == original, f"header.data_start changed from {original} to {header.data_start} after calling to_bytes(data_start=256)"
