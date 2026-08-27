"""What the start samplers actually produce, measured -- the input distribution of every sweep.

    uv run python -m benchmarks.sampler_stats [--samples 2000]

The geometry of a start is not cosmetic: of the 79 stored multi-start rows, 7 of the 13 begun at
a REENTRANT shape were excluded on `worst_sigma` against 2 of the other 66, and the excluded rows
repeat by start index across k. So the sampler decides how much of a sweep survives its own
guards, and until now nobody had looked at what it emits.

Reported per `N`, over `--samples` draws: the interior-angle extremes, the edge ratio, and the
fraction of draws that are reentrant. The acceptance table underneath prices the two rejection
bands that are actually on offer, so a band is chosen against its cost rather than guessed.
"""
import argparse

import numpy as np

from douse.sampling import random_convex, random_star_shaped, shape_stats

PERIM = 4.0


def min_edge(v):
    v = np.asarray(v, dtype=complex)
    return float(np.abs(v - np.roll(v, 1)).min())


def summarize(draws):
    a = np.array(draws)                       # (n, 3): min angle, max angle, edge ratio
    q = lambda col, p: np.percentile(a[:, col], p)                    # noqa: E731
    return (f"{q(0, 5):7.1f} {q(0, 50):7.1f}   {q(1, 50):7.1f} {q(1, 95):7.1f}   "
            f"{q(2, 5):6.3f} {q(2, 50):6.3f}   {np.mean(a[:, 1] > 180):6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f'{args.samples} draws per cell\n')
    print('                   min interior     max interior      edge ratio    reentrant  '
          'edge floor')
    print('   N  sampler         5%    50%       50%     95%        5%    50%      frac    '
          'violated')
    print('  ' + '-'*89)
    for N in (4, 5, 6, 8, 12):
        for name, gen in (('star-shaped', random_star_shaped), ('convex (Valtr)', random_convex)):
            vs = [gen(N, PERIM, rng) for _ in range(args.samples)]
            # A START THAT VIOLATES `MinEdgeLength` IS ALREADY INFEASIBLE when the run begins,
            # and the constraint's default floor is `perim/(10 n)` -- which a uniform convex
            # sampler runs into constantly at larger `N`, where its polygons grow slivers.
            floor = PERIM/(10.0*N)
            infeas = np.mean([min_edge(v) < floor for v in vs])
            print(f'  {N:2d}  {name:15s}{summarize([shape_stats(v) for v in vs])} {infeas:8.1%}')
        print()

    # WHAT A REJECTION BAND COSTS. A band is only worth stating next to its acceptance rate --
    # a filter that keeps 2% of draws is a different proposal from one that keeps 80%.
    print('  acceptance rate of a minimum-interior-angle band (spike rejection):\n')
    print('   N  sampler          >=10deg  >=20deg  >=30deg  >=45deg')
    print('  ' + '-'*58)
    for N in (4, 5, 6, 8, 12):
        for name, gen in (('star-shaped', random_star_shaped), ('convex (Valtr)', random_convex)):
            lo = np.array([shape_stats(gen(N, 4.0, rng))[0] for _ in range(args.samples)])
            rates = '  '.join(f'{np.mean(lo >= t):7.1%}' for t in (10, 20, 30, 45))
            print(f'  {N:2d}  {name:15s} {rates}')
        print()
if __name__ == "__main__":
    main()
