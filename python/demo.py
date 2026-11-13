from generator import Generator, DistributionSampler
from serialiser import CEBSHeader, CEBSObject, CEBSStream, Serialiser, StreamLocation, StreamType
from pathlib import Path
from cardinality import compute_cardinality

if __name__ == "__main__":
    python_generator = Generator.get_generator("python_test_range")
    python_result = python_generator.generate(0, 20, 2)
    print(f"python_test_range:\n{python_result}\n")

    c_generator = Generator.get_generator("c_test_range")
    c_result = c_generator.generate(0, 20, 2)
    print(f"c_test_range:\n{c_result}\n")

    header = CEBSHeader()
    header_bytes = header.to_bytes()
    stream = CEBSObject(header=header, stream=CEBSStream([], StreamType.UINT64))
    print(f"Header bytes:\n{stream}")

    test = CEBSHeader.from_bytes(header_bytes)
    print(f"\nDeserialised header:\n{test}\n")

    stream = CEBSObject(stream=CEBSStream(python_result, StreamType.UINT64))
    print(f"CEBS stream:\n{stream}")

    SEED = 7427369
    distribution_generator = Generator.get_generator("distribution", seed=SEED)
    for distribution in DistributionSampler.supported_distributions():
        result = distribution_generator.generate(distribution, 20)
        stream = CEBSObject(stream=CEBSStream(result, StreamType.FLOAT64 if DistributionSampler.SUPPORTED_DISTRIBUTIONS[distribution]["format"].__name__ == "float64" else StreamType.INT64))
        print(f"{distribution}:\n{stream}")

    observation_generator = Generator.get_generator("observations", seed=SEED)
    for distribution in DistributionSampler.supported_distributions():
        result = observation_generator.generate(distribution, 20)
        stream = CEBSObject(stream=CEBSStream(result, StreamType.INT64))
        print(f"{distribution}:\n{stream}")

    uuid_generator = Generator.get_generator("uuid", seed=SEED)
    result = uuid_generator.generate(10)
    stream = CEBSObject(stream=CEBSStream(result, StreamType.UUID))
    print(f"UUIDs:\n{stream}")


    BASE_DIR = Path(__file__).resolve().parent.parent
    CEBS_INPUT_DIR = BASE_DIR / "data" / "streams"

    print(f"{'stream':<61} {'cardinality':>12} {'elements':>12}")
    for file in sorted(file for file in CEBS_INPUT_DIR.glob("*.cebs") if file.is_file()):
        elements, cardinality = compute_cardinality(StreamLocation.TEMP_FILE, file)
        print(f"{file.name:<61} {cardinality:>12,} {elements:>12,}")
    print("")


    data = observation_generator.generate("poisson", 10 ** 6, lambda_=10)
    stream = CEBSObject(stream=CEBSStream(data, StreamType.INT64)).to_bytes()
    serialiser = Serialiser()
    sharedmemory, cleanup = serialiser.write_stream(stream, StreamLocation.SHARED_MEMORY)

    elements, cardinality = compute_cardinality(StreamLocation.SHARED_MEMORY, sharedmemory)
    print(f"cardinality: {cardinality}, elements: {elements}")
    cleanup()
