import sys
from pathlib import Path
from generator import Generator, DistributionSampler
from serialiser import CEBSStream, Serialiser, StreamDestination, StreamType


CEBS_OUTPUT_DIR = Path("./data/streams")

if __name__ == "__main__":

    if not CEBS_OUTPUT_DIR.exists():
        if "--make-directories" not in sys.argv:
            if input(f"Create directories to {CEBS_OUTPUT_DIR}? (yN) ").lower() != "y":
                sys.exit()
        CEBS_OUTPUT_DIR.mkdir(parents=True)

    serialiser = Serialiser()

    generator = Generator.get_generator("python_test_range")
    data = generator.generate(0, 20, 2)
    stream = CEBSStream().to_bytes(data, stream_type=StreamType.UINT64)

    path = CEBS_OUTPUT_DIR / "range_python.cebs"

    serialiser.write_stream(stream, StreamDestination.TEMP_FILE, path=path)

    print(f"Wrote {len(data)} values to {path}")

    SEED = 7427369
    distribution_generator = Generator.get_generator("distribution", seed=SEED)
    for distribution in DistributionSampler.supported_distributions():
        data = distribution_generator.generate(distribution, int(1E4))
        stream = CEBSStream().to_bytes(data)

        path = CEBS_OUTPUT_DIR / f"{distribution}.cebs"
        serialiser.write_stream(stream, StreamDestination.TEMP_FILE, path=path)

        print(f"Wrote {len(data)} values to {path}")


    observation_generator = Generator.get_generator("observations", seed=SEED)
    for distribution in DistributionSampler.supported_distributions():
        data = observation_generator.generate(distribution, int(1E4))
        stream = CEBSStream().to_bytes(data)

        path = CEBS_OUTPUT_DIR / f"{distribution}_observations.cebs"
        serialiser.write_stream(stream, StreamDestination.TEMP_FILE, path=path)

        print(f"Wrote {len(data)} values to {path}")


    uuid_generator = Generator.get_generator("uuid", seed=SEED)
    data = uuid_generator.generate(20)
    stream = CEBSStream().to_bytes(data, stream_type=StreamType.UUID)

    path = CEBS_OUTPUT_DIR / "uuid.cebs"
    serialiser.write_stream(stream, StreamDestination.TEMP_FILE, path=path)

    print(f"Wrote {len(data)} values to {path}")


    from serialiser import cebs_to_string

    sharedmemory, cleanup = serialiser.write_stream(stream, StreamDestination.SHARED_MEMORY)
    shared_stream = bytes(sharedmemory.buf[:len(stream)])
    print(f"Shared memory stream:\n{cebs_to_string(shared_stream)}")
    cleanup()
