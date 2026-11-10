"""`douse/repulsive.py`: the tangent-point energy's reference values and invariances, and Kendall's
density in apex coordinates."""
import numpy as np

from douse import repulsive as rp


def test_reference_energies():
    square = np.exp(2j*np.pi*np.arange(4)/4)
    equilateral = np.exp(2j*np.pi*np.arange(3)/3)
    assert abs(rp.energy(square) - 9.9567) < 1e-3
    assert abs(rp.energy(equilateral) - 15.042) < 1e-3


def test_energy_is_a_shape_function():
    z = np.array([0, 1, 0.3 + 0.5j])
    E = rp.energy(z)
    assert abs(rp.energy(2.5*np.exp(0.7j)*z + (1 - 2j)) - E) < 1e-9*E     # similarity
    assert abs(rp.energy(z[::-1]) - E) < 1e-9*E                          # orientation
    assert abs(rp.energy(np.roll(z, 1)) - E) < 1e-9*E                    # relabelling
    assert abs(rp.energy(np.array([0, 0.5, 1, 0.3 + 0.5j])) - E) < 1e-6*E   # a straight vertex


def test_energy_grows_toward_a_fold_and_is_infinite_at_a_contact():
    flat = [rp.energy(np.array([0, 1, 0.3 + h*1j])) for h in (0.3, 0.1, 0.03)]
    assert flat[0] < flat[1] < flat[2]
    touching = np.array([0, 2, 2 + 1j, 1 + 0j, 0 + 1j])                  # a vertex on an edge
    assert not np.isfinite(rp.energy(touching))


def test_apex_is_the_canonical_representative():
    w = rp.apex(np.array([0, 1, 0.3 + 0.5j]))
    for z in (np.array([0.3 + 0.5j, 0, 1]), np.conj(np.array([0, 1, 0.3 + 0.5j])),
              3j*np.array([1, 0, 0.3 + 0.5j]) + 2):
        assert abs(rp.apex(z) - w) < 1e-12
    assert 0 <= w.real <= 0.5 and w.imag > 0


def test_kendall_density_is_centroid_size_to_the_minus_four():
    for w in (0.5 + 0.866j, 0.2 + 0.3j, 0.05 + 0.6j):
        z = np.array([0, 1, w])
        cs2 = np.sum(np.abs(z - z.mean())**2)
        ref = np.array([0, 1, 0.5 + 0.5j*np.sqrt(3)])
        cs2_ref = np.sum(np.abs(ref - ref.mean())**2)
        ratio = rp.kendall_apex_density(w)/rp.kendall_apex_density(ref[2])
        assert abs(ratio - (cs2/cs2_ref)**-2) < 1e-12


def _slit(gap):
    """The unit square with a slit of width `gap` cut down from the top: two long edges face each
    other across the gap. Dyadic vertices for dyadic gaps."""
    return np.array([0, 1, 1 + 1j, 0.5 + gap/2 + 1j, 0.5 + gap/2 + 0.25j,
                     0.5 - gap/2 + 0.25j, 0.5 - gap/2 + 1j, 1j])


def _geometry(z):
    e = np.roll(z, -1) - z
    l = np.abs(e)
    return e, l, l.sum(), np.r_[0.0, np.cumsum(l)[:-1]]


def _brute_pair(z, i, j, d, a=rp.A_EXP, b=rp.B_EXP, c=rp.C_EXP):
    """Ordered edge pair on uniform panels narrower than the gap: expensive and unconditionally
    resolved, sharing no grading with the code under test."""
    e, l, P, cum = _geometry(z)
    x8, w8 = np.polynomial.legendre.leggauss(8)

    def nodes(L):
        m = int(np.ceil(2*L/d))
        h = L/m
        return ((np.arange(m)[:, None] + 0.5*(x8[None, :] + 1))*h).ravel(), np.tile(0.5*h*w8, m)
    s, ws = nodes(l[i])
    t, wt = nodes(l[j])
    x = z[i] + s*e[i]/l[i]
    y = z[j] + t*e[j]/l[j]
    r = y[None, :] - x[:, None]
    ar = np.abs(r)
    D = rp._chord(np.abs((cum[i] + s)[:, None] - (cum[j] + t)[None, :]), P)
    k = np.abs(((e[i]/l[i]).conjugate()*r).imag)**a/ar**b*(D/ar)**c
    return ws @ k @ wt


def test_near_pair_matches_brute_force_on_close_nearly_parallel_edges():
    a, b, c = rp.A_EXP, rp.B_EXP, rp.C_EXP
    for z in (np.array([0, 1, 1 + 0.01j, 0.01j, -0.3 + 0.5j]),         # parallel, gap d
              np.array([0, 1, 1 + 0.015j, 0.1 + 0.01j, -0.3 + 0.5j]),  # tilted
              np.array([0, 1, 1 + 0.01j, 0.6 + 0.01j, 0.2 + 0.7j])):   # one edge shorter
        e, l, P, cum = _geometry(z)
        d = rp._closest(z[0], e[0], z[2], e[2])[2]
        for i, j in ((0, 2), (2, 0)):
            near = rp._near_pair_w(z, e, l, cum, P, i, j, d, a, b, c)
            assert abs(near/_brute_pair(z, i, j, d) - 1) < 1e-11


def test_the_two_pair_rules_agree_where_energy_switches():
    a, b, c = rp.A_EXP, rp.B_EXP, rp.C_EXP
    z = np.array([0, 1, 1 + 0.5j, 0.5j, -0.3 + 0.7j])                  # gap = half the edge
    e, l, P, cum = _geometry(z)
    s0, t0, d = rp._closest(z[0], e[0], z[2], e[2])
    tensor = rp._pair_w(z, e, l, cum, P, 0, 2, s0, t0, d, a, b, c)
    near = rp._near_pair_w(z, e, l, cum, P, 0, 2, d, a, b, c)
    assert abs(tensor/near - 1) < 1e-9


def test_a_closing_slit_costs_gap_to_the_minus_three_halves():
    """d^(a-b-c+1); the old tensor rule gave d^(a-b-c), one power too many."""
    gaps = 2.0**-np.array([14, 17, 20])
    E = np.array([rp.energy(_slit(g)) for g in gaps])
    assert np.all(np.abs(np.diff(np.log(E))/np.diff(np.log(gaps)) + 1.5) < 2e-2)


def test_energy_does_not_depend_on_distance_from_the_origin():
    """Rounding must follow the polygon's own size. Vertices are dyadic and the shifts exact, so
    these polygons are identical and any difference is evaluation rounding."""
    spike = np.array([0, 1, 1 + 1j, 0.5625 + 1j, 0.5 + 2.0**-20*1j, 0.4375 + 1j, 1j])
    for z in (_slit(2.0**-20), _slit(2.0**-5), spike, np.array([0, 1, 0.3125 + 0.5j])):
        E = rp.energy(z)
        for c in (2.0**10 + 2.0**9*1j, 2.0**20, -2.0**20 + 2.0**21*1j):
            assert abs(rp.energy(z + c)/E - 1) < 1e-14
