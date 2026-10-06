#!/usr/bin/env python3
"""Compare two MNIST.gsl error traces (e.g. CPU build vs GPU build).

Each gslparser run prints lines like
    6002 : current error = 0.310935 : wins ratio = 0.336721
This script pairs those lines by iteration and reports how far apart the
two runs are. Runs must use the same .gsl file, seed (-s) and thread count.

Usage: compare_mnist_traces.py run_a.out run_b.out [--tol-error 1e-3] [--tol-wins 1e-2]
Exit status is 0 when every checkpoint is within tolerance, 1 otherwise.
"""
import argparse
import re
import sys

LINE = re.compile(r"^\s*(\d+)\s*:\s*current error\s*=\s*([-+0-9.eE]+)\s*:\s*wins ratio\s*=\s*([-+0-9.eE]+)")


def read_trace(path):
    trace = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = LINE.match(line)
            if m:
                trace[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return trace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--tol-error", type=float, default=1e-3, help="max |difference| in current error")
    ap.add_argument("--tol-wins", type=float, default=1e-2, help="max |difference| in wins ratio")
    args = ap.parse_args()

    a, b = read_trace(args.run_a), read_trace(args.run_b)
    if not a or not b:
        sys.exit(f"no 'current error' lines found in {args.run_a if not a else args.run_b}")
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("the two runs share no checkpoints (different .gsl settings?)")

    worst_err = worst_wins = 0.0
    failures = 0
    print(f"{'iteration':>10} {'error A':>10} {'error B':>10} {'|d err|':>9} {'wins A':>8} {'wins B':>8} {'|d wins|':>9}")
    for it in common:
        (ea, wa), (eb, wb) = a[it], b[it]
        de, dw = abs(ea - eb), abs(wa - wb)
        worst_err, worst_wins = max(worst_err, de), max(worst_wins, dw)
        flag = ""
        if de > args.tol_error or dw > args.tol_wins:
            failures += 1
            flag = "  <-- outside tolerance"
        print(f"{it:>10} {ea:>10.6f} {eb:>10.6f} {de:>9.2e} {wa:>8.4f} {wb:>8.4f} {dw:>9.2e}{flag}")

    only = (set(a) ^ set(b))
    if only:
        print(f"\nnote: {len(only)} checkpoint(s) appear in only one run")
    print(f"\ncheckpoints compared: {len(common)}")
    print(f"max |d error| = {worst_err:.3e} (tolerance {args.tol_error:g})")
    print(f"max |d wins|  = {worst_wins:.3e} (tolerance {args.tol_wins:g})")
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} checkpoint(s) outside tolerance)")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
