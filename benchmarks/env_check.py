"""Does this machine run douse? One printed block, nonzero exit on a failed check.  (2026-10-16)

    uv run python -m benchmarks.env_check

Run it first on a new machine (ARC's Linux nodes above all), before any grid. It reports Python
and the platform, chladni (importable, and which version), gsvd4py and the LAPACK it found for
`?ggsvd3`, the SHAs of douse and chladni as every ledger row records them (`douse.spec.provenance`),
and solves one Dirichlet problem: the unit square, whose first eigenvalue is 2 pi^2.

THE SOLVE IS THE CHECK THAT MATTERS. gsvd4py probes for LAPACK at CALL time, not at import, and
off macOS a pip-installed scipy does not satisfy the probe (chladni's `pyproject.toml`, the
`scipy-openblas32` dependency) -- so "chladni imports" says nothing about whether it solves.
"""
import importlib.metadata
import platform
import sys
import traceback
import warnings

import numpy as np

SQUARE_LAM1 = 2*np.pi**2
SQUARE_RTOL = 1e-7      # asked for 1e-10; chladni warns it may deliver ~1e-9 on the square


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def checks():
    """`[(name, ok, detail)]`, in print order. `ok` is None for a line that is information only."""
    out = [('python', None, f'{sys.version.split()[0]} ({sys.executable})'),
           ('platform', None, f'{platform.platform()} {platform.machine()}')]

    try:
        import chladni                                                # noqa: F401
        out.append(('chladni', True, f"{_version('chladni')} at {chladni.__file__}"))
    except Exception as exc:                                          # noqa: BLE001
        out.append(('chladni', False, f'not importable: {type(exc).__name__}: {exc}'))
        chladni = None

    try:
        import gsvd4py
        info = gsvd4py.lapack_info()
        route = (f"{info.get('lib_type')} ({info.get('source')}, {info.get('int_width')}-bit "
                 f"ints) {info.get('path')}")
        out.append(('gsvd4py', True, f"{_version('gsvd4py')}; LAPACK {route}"))
    except Exception as exc:                                          # noqa: BLE001
        out.append(('gsvd4py', False, f'no LAPACK route: {type(exc).__name__}: {exc}'))

    if chladni is not None:
        try:
            from chladni.eigenproblem import Eigenproblem
            from chladni.geometry import Polygon
            dom = Polygon(np.array([0, 1, 1 + 1j, 1j]), bc='dir')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _, rec = Eigenproblem(dom).plan_and_solve(1, prec=1e-10)
            lam = float(np.ravel(rec.lam)[0])
            err = abs(lam - SQUARE_LAM1)/SQUARE_LAM1
            out.append(('square solve', bool(err <= SQUARE_RTOL),
                        f'lam_1 = {lam:.12f} vs 2 pi^2 = {SQUARE_LAM1:.12f}, '
                        f'rel err {err:.1e} (tol {SQUARE_RTOL:.0e})'))
        except Exception as exc:                                      # noqa: BLE001
            tail = traceback.format_exc().strip().splitlines()[-1]
            out.append(('square solve', False, f'{type(exc).__name__}: {exc} [{tail}]'))
    else:
        out.append(('square solve', False, 'skipped: chladni not importable'))

    from douse import spec
    prov = spec.provenance()
    out.append(('douse sha', prov['douse_sha'] is not None, str(prov['douse_sha'])))
    out.append(('chladni sha', prov['chladni_sha'] is not None, str(prov['chladni_sha'])))
    return out


def report(results):
    """The printed block, as text."""
    mark = {True: 'ok  ', False: 'FAIL', None: '    '}
    lines = ['douse environment check']
    lines += [f'  [{mark[ok]}] {name:13s} {detail}' for name, ok, detail in results]
    failed = [name for name, ok, _ in results if ok is False]
    lines.append(f"  {'FAILED: ' + ', '.join(failed) if failed else 'all checks passed'}")
    return '\n'.join(lines)


def main():
    results = checks()
    print(report(results))
    return 1 if any(ok is False for _, ok, _ in results) else 0


if __name__ == '__main__':
    sys.exit(main())
