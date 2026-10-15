import sys

from scipy.stats import chi2

DF = 1023
# From cpp/tests/test_hash.cpp
LOWER_CRITICAL = 910.2463
UPPER_CRITICAL = 1143.2653
TOLERANCE = 1e-3

checks = {
    "lower (ppf 0.005)": (chi2.ppf(0.005, DF), LOWER_CRITICAL),
    "upper (ppf 0.995)": (chi2.ppf(0.995, DF), UPPER_CRITICAL),
}

failed = False
for name, (expected, hard_coded) in checks.items():
    ok = abs(expected - hard_coded) <= TOLERANCE
    failed |= not ok
    print(f"{name}: scipy={expected:.6f} test_hash.cpp={hard_coded} {'OK' if ok else 'MISMATCH'}")

sys.exit(1 if failed else 0)
