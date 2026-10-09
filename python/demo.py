from generator import Generator, DistributionSampler
from serialiser import CEBSHeader, CEBSStream, StreamType, cebs_to_string
import numpy as np


if __name__ == "__main__":
    python_generator = Generator.get_generator("python_test_range")
    python_result = python_generator.generate(0, 20, 2)
    print(f"python_test_range:\n{python_result}\n")

    c_generator = Generator.get_generator("c_test_range")
    c_result = c_generator.generate(0, 20, 2)
    print(f"c_test_range:\n{c_result}\n")

    header = CEBSHeader()
    header_bytes = header.to_bytes()
    print(f"Header bytes:\n{cebs_to_string(header_bytes)}")

    test = CEBSHeader.from_bytes(header_bytes)
    print(f"\nDeserialised header:\n{test}\n")

    cebs = CEBSStream()
    stream = cebs.to_bytes(python_result, stream_type=StreamType.UINT64)
    print(f"CEBS stream:\n{cebs_to_string(stream, StreamType.UINT64)}")

    SEED = 7427369
    distribution_generator = Generator.get_generator("distribution", seed=SEED)
    cebs = CEBSStream()
    for distribution in DistributionSampler.supported_distributions():
        result = distribution_generator.generate(distribution, 20)
        stream = cebs.to_bytes(result)
        print(f"{distribution}:\n{cebs_to_string(stream)}")

    observation_generator = Generator.get_generator("observations", seed=SEED)
    cebs = CEBSStream()
    for distribution in DistributionSampler.supported_distributions():
        result = observation_generator.generate(distribution, 20)
        stream = cebs.to_bytes(result)
        print(f"{distribution}:\n{cebs_to_string(stream)}")

    uuid_generator = Generator.get_generator("uuid", seed=SEED)
    cebs = CEBSStream()

    result = uuid_generator.generate(10)
    stream = cebs.to_bytes(result, stream_type=StreamType.UUID)

    print(f"UUIDs:\n{cebs_to_string(stream, StreamType.UUID)}")



