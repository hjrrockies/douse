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
