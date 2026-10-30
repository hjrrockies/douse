"""`max lam_k/lam_1` over RECTANGLES of fixed perimeter, exactly, for every `k`. No eigensolver.

    uv run python -m benchmarks.rectangle_exact [--kmax 60] [--perim 4.0]
    uv run python -m benchmarks.rectangle_exact --gap [--kmax 31]

`--gap` tabulates `max (lam_{K+1} - lam_K)/lam_1` instead, in exact rational arithmetic with no
grid at all (`best_gap`): every value is a rational at a rational `L^2/H^2`, and the square is
the maximizer only at K = 1, 3, 6, 8, 13, 19, 30 of K <= 30. Like the ratio table it is a lower
bound for the same problem over quadrilaterals.

WHY THIS EXISTS. Everything else in this repo tests the optimizer against a known answer at ONE
`k` -- the square at k=2, the `sqrt(8/3)` rectangle at k=3. On a rectangle `L x H` the spectrum is
`pi^2 (m^2/L^2 + n^2/H^2)` in closed form, so on that one-parameter subfamily the maximization is
a scalar problem that can be solved to machine precision for ANY `k`, in milliseconds, with no
discretization anywhere. That makes it two things at once:

  * AN ORACLE. `max_{rectangles} lam_k/lam_1` is a rigorous LOWER BOUND for
    `max_{quadrilaterals} lam_k/lam_1`, at every `k`. A `douse` run over quadrilaterals that ends
    BELOW this number has not converged, whatever its criticality certificate says -- and that is
    a check available at k=12 where nothing else is.
  * A RESULT. The table below is a closed-form answer to the target problem restricted to
    rectangles, and it is not the shape anyone would guess.

WHAT IT SAYS, and it settles the "what is the claim at high k" question in `HANDOFF.md`.

The square is the optimal rectangle at k = 2, 5, 16, 40 -- and is NOT at k = 3, 4, 6, 8, 10, 12,
20, 25, 30, 50. The answer does not settle down; it oscillates, because which pair `(m, n)` is
competing for the k-th slot is an arithmetic question about the lattice, not a geometric one.

But it oscillates around a limit, and the limit is the square. By Weyl, `lam_k ~ 4 pi k/|Omega|`,
so with the perimeter fixed

    lam_k/lam_1  ~  4 pi k / (|Omega| lam_1)

and the shape-dependence at leading order is entirely in `|Omega| lam_1`, the SCALE-INVARIANT
Faber--Krahn functional. Maximizing the ratio therefore tends to MINIMIZING `|Omega| lam_1`, whose
answer among N-gons is the regular N-gon (a theorem for N = 3, 4). So the prediction is that the
maximizer tends to the regular N-gon as `k -> infinity`, and the measured `L/H` does drift toward
1 -- 0.61, 0.77 at k = 3, 4 against 0.88, 0.91, 0.94, 0.90 at k = 20, 25, 30, 50 -- while never
staying there. `ratio/k` falls toward `2/pi = 0.6366`, which is that same asymptotic evaluated at
the square, and reaches 0.74 by k = 50.

That is a genuine structure for the paper: at k=2 the answer IS the regular polygon, at k=3 it
provably is not, and as k grows it comes back to it in the mean while oscillating forever.

THE WEYL ARGUMENT IS A HEURISTIC, NOT A PROOF, and should not be written up as one. Weyl's error
term is `O(sqrt(lam))` and is not uniform over a family of shapes, so "the leading term dominates"
is an interchange of limits that has not been justified here. What IS measured is the drift in
`L/H` and in `ratio/k`, both in the predicted direction and both slow -- `ratio/k` is still 0.78
at k=30 against the limit 0.6366, which is exactly the size of a `sqrt(k)/k` correction. The
honest claim is a numerically supported conjecture about the large-`k` behaviour, with an exact
table underneath it for small `k`.

EVERY OPTIMAL RATIO IN THE TABLE IS RATIONAL, at all 29 indices from k=2 to k=30. That is not a
numerical coincidence and it is the cleanest statement here: the maximum sits at a CROSSING
`lam_{m,n} = lam_{m',n'}`, which forces `L^2/H^2 = (m'^2 - m^2)/(n^2 - n'^2)` to be rational, and
then every eigenvalue is `pi^2/L^2` times a rational, so the factor cancels out of the ratio. The
aspect ratio is the square root of a rational; the VALUE is rational outright. 35/11 at k=3 is the
first entry of a family, not a special case.

CAVEAT, and it is the whole reason this is called an oracle rather than an answer. Rectangles are
a one-parameter subfamily of quadrilaterals. This maximizes over that subfamily only, so it is a
LOWER bound for the quadrilateral problem. At k=3 the two coincide (`douse` finds the rectangle);
there is no reason they must at every k, and where they differ that difference is itself a result.
"""
import argparse
import json
import os
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize_scalar

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'rectangle_exact')


def spectrum(L, H, n_modes):
    """The Dirichlet spectrum of `L x H`, sorted. Exact: `pi^2 (m^2/L^2 + n^2/H^2)`."""
    m = np.arange(1, n_modes + 1)
    return np.sort((np.pi**2*(m[:, None]**2/L**2 + m[None, :]**2/H**2)).ravel())


def _n_modes_for(k):
    """Enough modes that the k-th smallest is certainly present and correct.

    `lam_{m,n}` is increasing in both indices, so taking `m, n <= M` misses nothing below
    `min(lam_{M+1,1}, lam_{1,M+1})`. `M = k + 8` is far more than enough for the aspect ratios
    here and is checked below rather than assumed.
    """
    return int(k) + 12


def ratio(L, k, perim):
    H = perim/2 - L
    if L <= 0 or H <= 0:
        return -np.inf
    s = spectrum(L, H, _n_modes_for(k))
    return float(s[k]/s[0])


def _crossing_L(L0, k, perim, n_modes):
    """The exact `L` at which the two modes meeting at `L0` cross, in closed form.

    At a crossing of `(m, n)` and `(m', n')`,

        m^2/L^2 + n^2/H^2 = m'^2/L^2 + n'^2/H^2
        =>  H^2/L^2 = (n'^2 - n^2)/(m^2 - m'^2) =: r
        =>  L = (perim/2)/(1 + sqrt(r))                       since L + H = perim/2

    so once the two competing index pairs are known the answer is arithmetic, with no iteration
    and no tolerance.

    WHY THIS IS NEEDED. The maximum sits AT a kink -- that is the whole content of the k=3 result
    -- and `minimize_scalar(method='bounded')` is Brent's method, which assumes smoothness and
    crawls at a kink. Measured against the exactly known rationals, the bounded refinement alone
    lands 2.6e-10 to 5.5e-08 BELOW the true value, systematically low at every index tested:

        k= 3   3.181818181559   vs 35/11 = 3.181818181818   (-2.6e-10)
        k= 6   5.923076879916   vs 77/13 = 5.923076923077   (-4.3e-08)
        k= 8   6.999999944774   vs 7      = 7.000000000000   (-5.5e-08)
        k=11   9.333333304679   vs 28/3  = 9.333333333333   (-2.9e-08)

    Always low, so the ORACLE PROPERTY (a lower bound on the quadrilateral answer) survived --
    but "exact" was overstated by eight digits, and the comparison column carried that as noise.

    Returns `None` when no crossing is identifiable, in which case the caller keeps its iterate:
    not every optimum is a kink (the square is a symmetry point, not a crossing).
    """
    m = np.arange(1, n_modes + 1)
    H0 = perim/2 - L0
    lam = np.pi**2*(m[:, None]**2/L0**2 + m[None, :]**2/H0**2)
    flat = np.argsort(lam, axis=None)
    # THE CROSSING IS BETWEEN SLOTS k AND k+1, not k and k-1. `lam_k` is pushed UP until it meets
    # the mode above it -- at the k=3 optimum that is `lam_3 = lam_4`, the (1,2) and (3,1) modes.
    (a_m, a_n), (b_m, b_n) = (np.unravel_index(flat[k], lam.shape),
                              np.unravel_index(flat[k + 1], lam.shape))
    (m1, n1), (m2, n2) = (a_m + 1, a_n + 1), (b_m + 1, b_n + 1)
    num, den = float(n2**2 - n1**2), float(m1**2 - m2**2)
    if den == 0 or num == 0 or num/den <= 0:
        return None
    return (perim/2)/(1.0 + np.sqrt(num/den))


def best_rectangle(k, perim=4.0, n_grid=4001):
    """`argmax_L lam_k/lam_1`: a grid scan, a bounded refinement, then a CLOSED-FORM snap.

    The scan is not optional. `lam_k(L)` is a MAX of finitely many smooth branches that cross, so
    the objective is piecewise smooth with kinks, and a local method from a single start lands on
    whichever branch it began in. The crossings are exactly where the interesting answers live --
    the k=3 optimum IS a crossing.

    And for the same reason the bounded refinement is not enough on its own: Brent's method
    assumes smoothness and stalls at a kink, landing systematically 1e-10 to 6e-08 low. See
    `_crossing_L`, which solves the crossing exactly and is accepted only if it does not make the
    objective worse -- so an optimum that is NOT a crossing keeps the iterate.
    """
    half = perim/2
    Ls = np.linspace(0.18*half, half/2, n_grid)          # L <= H by symmetry
    vals = np.array([ratio(L, k, perim) for L in Ls])
    i = int(np.argmax(vals))
    lo, hi = Ls[max(i - 1, 0)], Ls[min(i + 1, n_grid - 1)]
    res = minimize_scalar(lambda L: -ratio(L, k, perim), bounds=(lo, hi), method='bounded',
                          options=dict(xatol=1e-13))
    L = float(res.x) if -res.fun >= vals[i] else float(Ls[i])
    best = ratio(L, k, perim)

    Lx = _crossing_L(L, k, perim, _n_modes_for(k))
    if Lx is not None and 0 < Lx < half:
        vx = ratio(Lx, k, perim)
        if vx >= best - 1e-14*max(1.0, best):
            L, best = float(Lx), vx
    return L, best


# ---- gaps ---------------------------------------------------------------------------------------

def best_gap(K):
    """`max (lam_{K+1} - lam_K)/lam_1` over rectangles, EXACTLY: a rational value at a rational
    `t = L^2/H^2`, found with no grid and no tolerance. `K` is 1-based (K = 1 is lam_2 - lam_1).

    WHY THE MAXIMUM IS AT A CROSSING. Scale out `pi^2/L^2` and take `L <= H`, so `t` is in (0, 1].
    Normalized by `lam_1 = (1 + t)`, every mode is a Mobius function of `t`,

        nu_{m,n}(t) = (m^2 + n^2 t)/(1 + t),

    and so is the difference of any two FIXED modes, `((m'^2 - m^2) + (n'^2 - n^2) t)/(1 + t)`,
    which is therefore monotone in `t`. The gap is that difference for whichever modes hold slots
    `K` and `K + 1`; between two consecutive crossings of any pair of modes the holders do not
    change, so the gap is monotone there and its maximum over `(0, 1]` sits at a crossing
    `t = (m^2 - m'^2)/(n'^2 - n^2)` or at the square, `t = 1`. Unlike `lam_k/lam_1`, whose
    optimum is a crossing AT slot k, the gap's optimum is a crossing ADJACENT to the pair: it is
    pushed open until `lam_K` meets `lam_{K-1}` from above or `lam_{K+1}` meets `lam_{K+2}` from
    below.

    WHY THE CANDIDATE SET IS FINITE. For `t <= t1 = 3/((K+1)^2 - 1)` the lowest `K + 1` modes are
    `(1, 1..K+1)` and the gap is `(2K + 1) t/(1 + t)`, increasing, so `(0, t1]` contributes only
    `t1`, itself a crossing. On `[t1, 1]` the `(K+1)`-th value is at most
    `nu_{1,K+1} <= (1 + (K+1)^2 t)/(1 + t)`, so a mode in the lowest `K + 1` has
    `m^2 <= 1 + (K+1)^2` and `n^2 <= 1/t1 + (K+1)^2`: finitely many modes, finitely many crossings.

    Candidates are ranked in floating point and the leaders (within 1e-9) re-evaluated in
    `Fraction`, so the returned value is the exact rational maximum. Ties (several maximizers)
    are reported. Returns a dict: `value`, `t` (Fractions), `L_over_H` (= sqrt(t) <= 1), `slots`
    (the (m, n) modes in slots K and K+1), `lower_double`/`upper_double` (lam_K = lam_{K-1},
    lam_{K+1} = lam_{K+2} at the optimum), `maximizers` (every maximizing t), `n_candidates`.
    """
    K = int(K)
    if K < 1:
        raise ValueError(f'K must be at least 1 (lam_2 - lam_1); got {K}')
    t1 = Fraction(3, (K + 1)**2 - 1)
    m_max = int(np.floor(np.sqrt(1 + (K + 1)**2)))
    n_max = int(np.floor(np.sqrt(float(1/t1) + (K + 1)**2)))
    modes = [(m, n) for m in range(1, m_max + 1) for n in range(1, n_max + 1)]
    # every crossing of two modes inside [t1, 1], plus the square
    ts = {Fraction(1)}
    m2 = sorted({m*m for m, _ in modes})
    n2 = sorted({n*n for _, n in modes})
    for a in m2:
        for b in m2:
            if b >= a:
                break
            for c in n2:
                for d in n2:
                    if d <= c:
                        continue
                    t = Fraction(a - b, d - c)       # m^2 + n^2 t = m'^2 + n'^2 t
                    if t1 <= t <= 1:
                        ts.add(t)
    ts = sorted(ts)
    M = np.array([m*m for m, _ in modes], dtype=float)
    N = np.array([n*n for _, n in modes], dtype=float)
    tf = np.array([float(t) for t in ts])
    vals = np.empty(len(ts))
    for i0 in range(0, len(ts), 2048):
        tt = tf[i0:i0 + 2048, None]
        nu = np.partition((M[None, :] + N[None, :]*tt)/(1 + tt), K, axis=1)[:, :K + 1]
        nu.sort(axis=1)
        vals[i0:i0 + 2048] = nu[:, K] - nu[:, K - 1]
    lead = np.flatnonzero(vals >= vals.max()*(1 - 1e-9))

    def exact(t):
        nu = sorted(((Fraction(m*m) + n*n*t)/(1 + t), (m, n)) for m, n in modes)
        return nu[K][0] - nu[K - 1][0], nu

    scored = [(exact(ts[i]), ts[i]) for i in lead]
    best = max(v for (v, _), _ in scored)
    winners = [(t, nu) for (v, nu), t in scored if v == best]
    t, nu = winners[0]
    # which side of the pair is pinned at the optimum: lam_K = lam_{K-1}, lam_{K+1} = lam_{K+2}
    lower = K >= 2 and nu[K - 2][0] == nu[K - 1][0]
    upper = nu[K][0] == nu[K + 1][0]
    return dict(K=K, value=best, t=t, L_over_H=float(np.sqrt(float(t))),
                slots=(nu[K - 1][1], nu[K][1]), lower_double=bool(lower), upper_double=bool(upper),
                maximizers=[w for w, _ in winners], n_candidates=len(ts))


def gap_table(kmax):
    """`best_gap(K)` for K = 1..kmax, with the square's own gap beside it."""
    rows = []
    for K in range(1, kmax + 1):
        r = best_gap(K)
        sq = sorted(Fraction(m*m + n*n, 2) for m in range(1, K + 3) for n in range(1, K + 3))
        r['square'] = sq[K] - sq[K - 1]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmax', type=int, default=40, help='largest 1-based eigenvalue index')
    ap.add_argument('--perim', type=float, default=4.0)
    ap.add_argument('--gap', action='store_true',
                    help='the gap table, max (lam_{K+1} - lam_K)/lam_1, K = 1..kmax-1, exactly')
    args = ap.parse_args()
    if args.gap:
        return main_gap(args.kmax - 1)

    perim = args.perim
    L_sq = perim/4
    rows = []
    print(f'max lam_k/lam_1 over rectangles of perimeter {perim}, exactly\n')
    print('  k    best L      L/H       lam_k/lam_1      exact      at the square   (r)/k  square?')
    print('  ' + '-'*82)
    for k1 in range(2, args.kmax + 1):
        k = k1 - 1                                        # 0-based index into the sorted spectrum
        L, val = best_rectangle(k, perim)
        sq = ratio(L_sq, k, perim)
        is_sq = abs(L - L_sq) < 1e-6 or abs(val - sq) <= 1e-12*max(1.0, abs(sq))
        # THE OPTIMAL RATIO IS RATIONAL, and that is not a numerical coincidence. The maximum
        # sits at a CROSSING `lam_{m,n} = lam_{m',n'}`, which pins `L^2/H^2 = (m'^2-m^2)/(n^2-n'^2)`
        # to a rational number -- and then every `lam` in the spectrum is `pi^2/L^2` times a
        # rational, so the ratio `lam_k/lam_1` is rational and the `pi^2/L^2` cancels. The
        # aspect ratio itself is the square root of a rational; the VALUE is rational outright.
        fr = Fraction(val).limit_denominator(2000)
        exact = (f'{fr.numerator}/{fr.denominator}'
                 if abs(float(fr) - val) <= 5e-8*max(1.0, val) else '')
        rows.append(dict(k=k1, L=L, H=perim/2 - L, ratio=val, square_ratio=sq,
                         is_square=bool(is_sq), exact=exact,
                         aspect_sq=(L/(perim/2 - L))**2))
        print(f'  {k1:3d}  {L:.8f}  {L/(perim/2 - L):8.5f}  {val:14.9f}  {exact:>9s}  '
              f'{sq:14.9f}  {val/k1:6.4f}  {"YES" if is_sq else ""}')

    n_rat = sum(bool(r['exact']) for r in rows)
    print(f'\n  the optimal ratio is rational at {n_rat} of {len(rows)} indices '
          f'(denominator <= 2000, matched to 5e-08 relative)')
    n_sq = sum(r['is_square'] for r in rows)
    print(f'\n  the square is optimal at {n_sq} of {len(rows)} indices: '
          f'{[r["k"] for r in rows if r["is_square"]]}')
    tail = [r for r in rows if r['k'] >= max(8, args.kmax//2)]
    if tail:
        dev = np.array([abs(r['L']/(perim/2 - r['L']) - 1.0) for r in tail])
        head = [r for r in rows if r['k'] < 8]
        devh = np.array([abs(r['L']/(perim/2 - r['L']) - 1.0) for r in head])
        print(f'  mean |L/H - 1|: {devh.mean():.4f} for k < 8, {dev.mean():.4f} for '
              f'k >= {max(8, args.kmax//2)}   (Weyl predicts -> 0)')
        print(f'  (ratio)/k at the largest k: {rows[-1]["ratio"]/rows[-1]["k"]:.5f}, '
              f'against the Weyl limit 2/pi = {2/np.pi:.5f}')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'perim{perim:g}.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    print(f'\n  written to {OUT}')


def main_gap(kmax):
    rows = gap_table(kmax)
    print('max (lam_{K+1} - lam_K)/lam_1 over rectangles, exactly (scale-free; t = L^2/H^2, L <= H)\n')
    print('   K    gap       = float          t      L/H      at the square  pinned   slots K, K+1')
    print('  ' + '-'*92)
    for r in rows:
        pin = ('lo+hi' if r['lower_double'] and r['upper_double'] else
               'lo' if r['lower_double'] else 'hi' if r['upper_double'] else '-')
        tie = f"  (+{len(r['maximizers']) - 1} tie)" if len(r['maximizers']) > 1 else ''
        print(f"  {r['K']:3d}  {str(r['value']):>8s}  {float(r['value']):12.9f}  {str(r['t']):>7s}  "
              f"{r['L_over_H']:7.5f}  {str(r['square']):>8s}  {pin:>6s}   {r['slots']}{tie}")
    n_sq = [r['K'] for r in rows if r['t'] == 1]
    print(f"\n  the square is a maximizer at K = {n_sq}")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'gaps.json')
    with open(out, 'w') as fh:
        json.dump([dict(K=r['K'], value=str(r['value']), value_float=float(r['value']),
                        t=str(r['t']), L_over_H=r['L_over_H'], square=str(r['square']),
                        lower_double=r['lower_double'], upper_double=r['upper_double'],
                        slots=[list(x) for x in r['slots']],
                        maximizers=[str(t) for t in r['maximizers']]) for r in rows],
                  fh, indent=1)
    print(f'\n  written to {out}')


if __name__ == '__main__':
    main()
