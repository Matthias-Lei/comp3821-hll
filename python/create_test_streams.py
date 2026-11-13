import sys
import os
from pathlib import Path
from generator import Generator, DistributionSampler
from serialiser import CEBSObject, CEBSStream, Serialiser, StreamLocation, StreamType


if __name__ == "__main__":


    is_remote = "--remote" in sys.argv or any(os.getenv(name) for name in ("COLAB_RELEASE_TAG", "CODESPACES")) and not any(os.getenv(name) for name in ("CI",))

    if is_remote and "--remote" not in sys.argv:
        print("environment detected as remote workspace. is this correct? (Yn) ", end="")
        if input().lower() == "n":
            is_remote = False

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

    nsamples = [int(1E4), int(1E5), int(1E6), int(1E7)] if is_remote else [int(1E4)]
    ms = [int(1E3), int(1E4)] if is_remote else [int(1E3)]

    distribution_generator = Generator.get_generator(f"distribution", seed=SEED)
    for distribution in DistributionSampler.supported_distributions():
        for nsample in nsamples:
            data = distribution_generator.generate(distribution, nsample)
            stream = CEBSObject(stream=CEBSStream(data, StreamType.UINT64)).to_bytes()

            path = CEBS_OUTPUT_DIR / f"{distribution}_seed{SEED}_nsample{nsample}.cebs"
            serialiser.write_stream(stream, StreamLocation.TEMP_FILE, path=path)

            print(f"Wrote {len(data)} values to {path}")


    observation_generator = Generator.get_generator(f"observations", seed=SEED)
    for distribution in DistributionSampler.supported_distributions():
        for m in ms:
            data = observation_generator.generate(distribution, m)
            stream = CEBSObject(stream=CEBSStream(data, StreamType.UINT64)).to_bytes()

            path = CEBS_OUTPUT_DIR / f"{distribution}_observations_seed{SEED}_m{m}.cebs"
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


    if is_remote:
        import numpy as np

        for sql_generator_name in ("sql_job", "sql_policing", "sql_lichess", "sql_weather"):
            sql_generator = Generator.get_generator(sql_generator_name, make_directories="--make-directories" in sys.argv)
            sql_generator.setup()

            for name, sql in sql_generator.queries().items():
                data = np.concatenate([np.empty(0, dtype=np.uint64), *sql_generator.generate(sql)])
                stream = CEBSObject(stream=CEBSStream(data, StreamType.XXHASH64)).to_bytes()

                path = CEBS_OUTPUT_DIR / f"{sql_generator.NAME}_{name}.cebs"
                serialiser.write_stream(stream, StreamLocation.TEMP_FILE, path=path)

                print(f"Wrote {len(data)} values to {path}")

            sql_generator.connection.close()
            (sql_generator.directory / sql_generator.DATABASE_NAME).unlink()
