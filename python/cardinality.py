import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import shared_memory

from serialiser import Serialiser, StreamLocation

DEFAULT_CHUNK_SIZE = 2 ** 24


def compute_cardinality(location: StreamLocation, source: Path | shared_memory.SharedMemory, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[int, int]:
    match location:
        case StreamLocation.TEMP_FILE:
            cebs = Serialiser().read_stream(location, path=source, chunk_elements=chunk_size)
        case StreamLocation.SHARED_MEMORY:
            cebs = Serialiser().read_stream(location, sharedmemory=source, size=source.size)
        case _:
            raise NotImplementedError()

    reader = cebs.stream
    number_of_units = cebs.header.number_of_units
    chunks = reader.chunks(chunk_size)
    dtype = reader.dtype

    accumulated: np.ndarray = np.empty(0, dtype=dtype)

    with tqdm(total=number_of_units, unit=" elements", unit_scale=True, disable=number_of_units <= chunk_size) as progressbar:
        for chunk in chunks:
            new = np.union1d(accumulated, np.unique(chunk))
            progressbar.update(len(chunk))
            accumulated = new

    return number_of_units, len(accumulated)