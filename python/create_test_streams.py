import sys
from pathlib import Path
from generator import Generator, DistributionSampler
from serialiser import CEBSObject, CEBSStream, Serialiser, StreamLocation, StreamType


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    CEBS_OUTPUT_DIR = BASE_DIR / "data" / "streams"

    if not CEBS_OUTPUT_DIR.exists():
        if "--make-directories" not in sys.argv:
            if input(f"Create directories to {CEBS_OUTPUT_DIR}? (yN) ").lower() != "y":
                sys.exit()
        CEBS_OUTPUT_DIR.mkdir(parents=True)

    serialiser = Serialiser()

    generator = Generator.get_generator("python_test_range")
    data = generator.generate(0, 20, 2)
    stream = CEBSObject(stream=CEBSStream(data, StreamType.UINT64)).to_bytes()

    path = CEBS_OUTPUT_DIR / "range_python.cebs"

    serialiser.write_stream(stream, StreamLocation.TEMP_FILE, path=path)

    print(f"Wrote {len(data)} values to {path}")

    SEED = 7427369
    distribution_generator = Generator.get_generator(f"distribution", seed=SEED)
    for distribution in DistributionSampler.supported_distributions():
        data = distribution_generator.generate(distribution, int(1E4))
        stream = CEBSObject(stream=CEBSStream(data, StreamType.UINT64)).to_bytes()

        path = CEBS_OUTPUT_DIR / f"{distribution}_seed{SEED}.cebs"
        serialiser.write_stream(stream, StreamLocation.TEMP_FILE, path=path)

        print(f"Wrote {len(data)} values to {path}")


    observation_generator = Generator.get_generator(f"observations", seed=SEED)
    for distribution in DistributionSampler.supported_distributions():
        data = observation_generator.generate(distribution, int(1E4))
        stream = CEBSObject(stream=CEBSStream(data, StreamType.UINT64)).to_bytes()

        path = CEBS_OUTPUT_DIR / f"{distribution}_observations_seed{SEED}.cebs"
        serialiser.write_stream(stream, StreamLocation.TEMP_FILE, path=path)

        print(f"Wrote {len(data)} values to {path}")


    uuid_generator = Generator.get_generator(f"uuid", seed=SEED)
    data = uuid_generator.generate(20)
    stream = CEBSObject(stream=CEBSStream(data, StreamType.UUID)).to_bytes()

    path = CEBS_OUTPUT_DIR / f"uuid_seed{SEED}.cebs"
    serialiser.write_stream(stream, StreamLocation.TEMP_FILE, path=path)

    print(f"Wrote {len(data)} values to {path}")


    sharedmemory, cleanup = serialiser.write_stream(stream, StreamLocation.SHARED_MEMORY)
    shared_stream = bytes(sharedmemory.buf[:len(stream)])
    print(f"Shared memory stream:\n{shared_stream}")
    cleanup()

