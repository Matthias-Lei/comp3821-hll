import numpy as np
from multiprocessing import shared_memory
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Iterable, Callable, Iterator
import uuid

import hashlib

def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

class StreamType(Enum):
    UINT64 = 0b00000000
    INT64 = 0b00000001
    FLOAT64 = 0b00000010
    UUID = 0b00001000
    INVALID = 0b11111111

HEADER_SIZE = 128

@dataclass
class CEBSHeader:
    magic: bytes = b"CEBS"
    version: bytes = "a\0\0\1".encode()
    data_start: int = HEADER_SIZE
    stream_type: StreamType = StreamType.INVALID
    bytes_per_unit: int = 0x00000008
    number_of_units: int = 0x00000000
    reserved: bytes = bytes(88)

    def to_bytes(self, data_start: int | None = None, stream_type: StreamType | None = None, bytes_per_unit: int | None = None, number_of_units: int | None = None) -> bytes:
        if data_start is not None: self.data_start = data_start
        if stream_type is not None: self.stream_type = stream_type
        if bytes_per_unit is not None: self.bytes_per_unit = bytes_per_unit
        if number_of_units is not None: self.number_of_units = number_of_units
        return (
            self.magic
            + self.version
            + self.data_start.to_bytes(8, "little")
            + self.stream_type.value.to_bytes(8, "little")
            + self.bytes_per_unit.to_bytes(8, "little")
            + self.number_of_units.to_bytes(8, "little")
            + self.reserved
        )

    @classmethod
    def from_bytes(cls, stream: bytes) -> "CEBSHeader":
        return cls(
            magic=stream[:4],
            version=stream[4:8],
            data_start=int.from_bytes(stream[8:16], "little"),
            stream_type=StreamType(int.from_bytes(stream[16:24], "little")),
            bytes_per_unit=int.from_bytes(stream[24:32], "little"),
            number_of_units=int.from_bytes(stream[32:40], "little"),
            reserved=stream[40:HEADER_SIZE],
        )



STREAM_TYPE_MAP = {
    StreamType.UINT64: {"np_dtype": np.dtype("uint64"), "dtype": np.dtype("<u8"), "bytes_per_unit": 8},
    StreamType.INT64: {"np_dtype": np.dtype("int64"), "dtype": np.dtype("<i8"), "bytes_per_unit": 8},
    StreamType.FLOAT64: {"np_dtype": np.dtype("float64"), "dtype": np.dtype("<f8"), "bytes_per_unit": 8},
    StreamType.UUID: {"np_dtype": np.dtype("V16"), "dtype": np.dtype("V16"), "bytes_per_unit": 16},
}


class StreamLocation(Enum):
    TEMP_FILE = "file"
    SHARED_MEMORY = "shared_memory"
    # PIPE = "pipe"
    # ARROW_FILE = "arrow"
    # RING_BUFFER = "ring"


@dataclass
class CEBSStream:
    data: list[int] | Iterable[int] | np.ndarray
    stream_type: StreamType | None = None

    @staticmethod
    def _stream_to_bytes(data: list[int] | Iterable[int] | np.ndarray, stream_type: StreamType | None = None) -> tuple[np.ndarray, StreamType, int, int]:

        match data:
            case np.ndarray():
                config = {config["np_dtype"]: (stream_type, config) for stream_type, config in STREAM_TYPE_MAP.items()}[data.dtype]
                stream_type, config = config
                array = data.astype(config["dtype"], copy=False)

            case list():
                if stream_type is None:
                    raise ValueError("unspecified stream_type")
                config = STREAM_TYPE_MAP[stream_type]
                array = np.asarray(data, dtype=config["dtype"])

            case Iterable():
                if stream_type is None:
                    raise ValueError("unspecified stream_type")
                config = STREAM_TYPE_MAP[stream_type]
                array = np.fromiter(data, dtype=config["dtype"])

            case _:
                raise TypeError(f"unsupported data type: {type(data)}")

        return array, stream_type, config["bytes_per_unit"], len(array)

    @property
    def dtype(self) -> np.dtype:
        if isinstance(self.data, np.ndarray):
            return self.data.dtype
        return STREAM_TYPE_MAP[self.stream_type]["dtype"]

    def chunks(self, chunk_elements: int = 2 ** 24) -> Iterator[np.ndarray]:
        if isinstance(self.data, np.ndarray):
            for i in range(0, len(self.data), chunk_elements):
                yield self.data[i:i + chunk_elements]
            return

        chunk = []
        for value in self.data:
            chunk.append(value)
            if len(chunk) >= chunk_elements:
                yield np.asarray(chunk, dtype=self.dtype)
                chunk.clear()

        if chunk:
            yield np.asarray(chunk, dtype=self.dtype)




class CEBSChunkedReader:

    def __init__(self, path: Path, chunk_elements: int = 2 ** 24) -> None:
        self.path = path
        self.chunk_elements = chunk_elements
        with open(path, "rb") as file:
            self.header = CEBSHeader.from_bytes(file.read(HEADER_SIZE))
        self.dtype: np.dtype = STREAM_TYPE_MAP[self.header.stream_type]["dtype"]

    def chunks(self, chunk_elements: int | None = None) -> Iterator[np.ndarray]:
        if chunk_elements is None:
            chunk_elements = self.chunk_elements
        bytes_per_element = self.header.bytes_per_unit
        with open(self.path, "rb") as file:
            file.seek(self.header.data_start)
            while raw := file.read(chunk_elements * bytes_per_element):
                yield np.frombuffer(raw, dtype=self.dtype)



@dataclass
class CEBSObject:
    stream: CEBSStream | CEBSChunkedReader
    header: CEBSHeader = field(default_factory=CEBSHeader)
    location: StreamLocation | None = None

    def __str__(self) -> str:
        data = self.to_bytes()
        stream_type = self.header.stream_type

        lines = []
        magic_str = "".join(chr(b) if 32 <= b < 127 else "." for b in data[0:4])
        version_str = ".".join([chr(data[4]), *(str(b) for b in data[5:8])])
        data_start = int.from_bytes(data[8:16], "little")
        stream_type_name = stream_type.name.lower()
        bytes_per_element = int.from_bytes(data[24:32], "little")
        number_of_elements = int.from_bytes(data[32:40], "little")
        reserved_bytes = data_start - 40

        hex_width = max(32, bytes_per_element * 3)

        lines.append(f"{'Offset':<6}{' ' * 4}{'Hex':<{hex_width}}{' ' * 4}Field")

        lines.append(f"{0:6d}{' ' * 4}{' '.join(f'{byte:02X}' for byte in data[0:4]):<{hex_width}}{' ' * 4}magic = \"{magic_str}\"")
        lines.append(f"{4:6d}{' ' * 4}{' '.join(f'{byte:02X}' for byte in data[4:8]):<{hex_width}}{' ' * 4}version = {version_str}")
        lines.append(f"{8:6d}{' ' * 4}{' '.join(f'{byte:02X}' for byte in data[8:16]):<{hex_width}}{' ' * 4}data_start = {data_start}")
        lines.append(f"{16:6d}{' ' * 4}{' '.join(f'{byte:02X}' for byte in data[16:24]):<{hex_width}}{' ' * 4}stream_type = {stream_type_name}")
        lines.append(f"{24:6d}{' ' * 4}{' '.join(f'{byte:02X}' for byte in data[24:32]):<{hex_width}}{' ' * 4}bytes_per_element = {bytes_per_element}")
        lines.append(f"{32:6d}{' ' * 4}{' '.join(f'{byte:02X}' for byte in data[32:40]):<{hex_width}}{' ' * 4}number_of_elements = {number_of_elements}")
        lines.append(f"{40:6d}{' ' * 4}{f'?? ?? ... ?? ({reserved_bytes} bytes)':<{hex_width}}{' ' * 4}reserved")
        lines.append("")

        for i in range(number_of_elements):
            offset = data_start + (i * bytes_per_element)
            chunk = data[offset:offset + bytes_per_element]

            decoded = ""
            match stream_type:
                case StreamType.UINT64:
                    decoded = str(int.from_bytes(chunk, "little", signed=False))
                case StreamType.INT64:
                    decoded = str(int.from_bytes(chunk, "little", signed=True))
                case StreamType.FLOAT64:
                    decoded = str(np.frombuffer(chunk, dtype="<f8")[0])
                case StreamType.UUID:
                    decoded = str(uuid.UUID(bytes=chunk))
                case _:
                    raise NotImplementedError()

            hex_bytes = " ".join(f"{byte:02X}" for byte in chunk)
            lines.append(f"{offset:6d}{' ' * 4}{hex_bytes:<{hex_width}}{' ' * 4}{f'value[{i}]':<10} = {decoded}")

        return "\n".join(lines) + "\n"





    @property
    def dtype(self) -> np.dtype:
        return STREAM_TYPE_MAP[self.header.stream_type]["dtype"]

    @classmethod
    def from_string(cls, string: str) -> "CEBSStream":
        raise NotImplementedError("from_string not implemented")

    
    def to_bytes(self) -> bytes:
        if not isinstance(self.stream, CEBSStream):
            raise TypeError()

        payload, stream_type, bytes_per_unit, number_of_units = CEBSStream._stream_to_bytes(self.stream.data, self.stream.stream_type)

        header_bytes = self.header.to_bytes(
            stream_type=stream_type,
            bytes_per_unit=bytes_per_unit,
            number_of_units=number_of_units,
        )

        return header_bytes + payload.tobytes()



class Serialiser:
    def write_stream(self, data: bytes, destination: StreamLocation, **kwargs) -> None | tuple[shared_memory.SharedMemory, Callable[[], None]]:
        match destination:
            case StreamLocation.TEMP_FILE:
                self._write_stream_to_file(data, **kwargs)
                return None
            case StreamLocation.SHARED_MEMORY:
                return self._write_stream_to_shared_memory(data)
            case _:
                raise NotImplementedError(f"destination {destination.name} is not implemented")

    @staticmethod
    def _write_stream_to_file(data: bytes, path: Path) -> None:
        path.write_bytes(data)

    @staticmethod
    def _write_stream_to_shared_memory(data: bytes) -> tuple[shared_memory.SharedMemory, Callable[[], None]]:
        sharedmemory = shared_memory.SharedMemory(create=True, size=len(data))
        sharedmemory.buf[:len(data)] = data
        return sharedmemory, lambda: (sharedmemory.close(), sharedmemory.unlink())

    def read_stream(self, source: StreamLocation, **kwargs) -> CEBSObject:
        match source:
            case StreamLocation.TEMP_FILE:
                return self._read_stream_from_file(**kwargs)
            case StreamLocation.SHARED_MEMORY:
                return self._read_stream_from_shared_memory(**kwargs)
            case _:
                raise NotImplementedError(f"source {source.name} is not implemented")

    @staticmethod
    def _read_stream_from_file(path: Path, chunk_elements: int = 2 ** 24) -> CEBSObject:
        reader = CEBSChunkedReader(path, chunk_elements=chunk_elements)
        return CEBSObject(header=reader.header, stream=reader, location=StreamLocation.TEMP_FILE)

    @staticmethod
    def _read_stream_from_shared_memory(sharedmemory: shared_memory.SharedMemory, size: int) -> CEBSObject:
        header = CEBSHeader.from_bytes(bytes(sharedmemory.buf[:HEADER_SIZE]))
        data = np.frombuffer(sharedmemory.buf[:size], dtype=STREAM_TYPE_MAP[header.stream_type]["dtype"], offset=header.data_start, count=header.number_of_units)
        return CEBSObject(header=header, stream=CEBSStream(data, header.stream_type), location=StreamLocation.SHARED_MEMORY)


