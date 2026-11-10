"""Generate the HLL++ bias header, optionally summarizing raw trials first."""

import csv
import math
import sys


PRECISIONS = (10, 12, 13, 14)
TRIALS = 5000


def summarize(raw_path, summary_path):
    expected = {
        (p, (5 * (1 << p) * point + 99) // 199)
        for p in PRECISIONS
        for point in range(200)
    }
    totals = {}
    seen = set()
    current_trial = 0
    with open(raw_path, newline="") as source:
        for row in csv.DictReader(source):
            trial = int(row["trial"])
            if trial != current_trial:
                if trial != current_trial + 1 or seen != expected:
                    raise ValueError("missing, duplicated, or unordered trial records")
                current_trial = trial
                seen.clear()
            p, n, estimate = int(row["p"]), int(row["n"]), float(row["estimate"])
            key = p, n
            if p not in PRECISIONS or key in seen or not math.isfinite(estimate):
                raise ValueError("invalid calibration record")
            seen.add(key)
            total, count = totals.get(key, (0.0, 0))
            totals[key] = total + estimate, count + 1
    if current_trial != TRIALS - 1 or seen != expected or set(totals) != expected:
        raise ValueError("incomplete calibration")
    if any(count != TRIALS for _, count in totals.values()):
        raise ValueError("incomplete checkpoint")

    with open(summary_path, "w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("p", "n", "trials", "mean_raw", "bias"))
        for (p, n), (total, count) in sorted(totals.items()):
            mean = total / count
            writer.writerow((p, n, count, mean, mean - n))


def generate_header(summary_path, header_path):
    tables = {p: [] for p in PRECISIONS}
    with open(summary_path, newline="") as source:
        for row in csv.DictReader(source):
            p, n = int(row["p"]), int(row["n"])
            raw, bias = float(row["mean_raw"]), float(row["bias"])
            if (
                p not in tables
                or int(row["trials"]) != TRIALS
                or not math.isfinite(raw)
                or not math.isfinite(bias)
                or not math.isclose(raw - n, bias, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("invalid calibration summary")
            tables[p].append((n, raw, bias))

    for p, rows in tables.items():
        expected = {(5 * (1 << p) * point + 99) // 199 for point in range(200)}
        if len(rows) != len(expected) or {n for n, _, _ in rows} != expected:
            raise ValueError(f"incomplete calibration for precision {p}")

    with open(header_path, "w") as target:
        target.write(
            "#pragma once\n\n"
            "// locally calibrated from 5000 synthetic datasets per cardinality "
            "(heule et al., section 5.2).\n"
            "// bias = mean raw hll estimate - true count; checkpoints reuse dataset prefixes.\n"
            "// generated from experiments/hllpp/bias_calibration.csv.\n"
            "struct BiasPoint { double raw; double bias; };\n"
        )
        for p, rows in tables.items():
            target.write(f"\nstatic constexpr BiasPoint kBiasP{p}[] = {{\n")
            for _, raw, bias in sorted(rows, key=lambda row: (row[1], row[0])):
                target.write(f"    {{ {raw!r}, {bias!r} }},\n")
            target.write("};\n")


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: generate_bias.py <summary.csv> <header.hpp> [raw.csv]")
    summary, header = sys.argv[1:3]
    if len(sys.argv) == 4:
        summarize(sys.argv[3], summary)
    generate_header(summary, header)


if __name__ == "__main__":
    main()
