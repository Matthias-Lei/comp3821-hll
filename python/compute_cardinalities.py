import csv
from pathlib import Path

from cardinality import compute_cardinality
from serialiser import StreamLocation


BASE_DIR = Path(__file__).resolve().parent.parent
CEBS_INPUT_DIR = BASE_DIR / "data" / "streams"
CARDINALITY_PATH = BASE_DIR / "data" / "cardinalities.tsv"



def calculate_cardinalities() -> None:
    with open(CARDINALITY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["stream", "number_of_units", "cardinality"], delimiter="\t", lineterminator="\n")
        writer.writeheader()

        for stream in sorted(CEBS_INPUT_DIR.glob("*.cebs")):
            if not stream.is_file():
                continue

            number_of_units, cardinality = compute_cardinality(StreamLocation.TEMP_FILE, stream)

            writer.writerow({
                "stream": stream.name,
                "number_of_units": number_of_units,
                "cardinality": cardinality,
            })

            print(f"{f'{stream.name}:':<48} {cardinality:>8,}/{number_of_units:>8,}")


if __name__ == "__main__":
    calculate_cardinalities()