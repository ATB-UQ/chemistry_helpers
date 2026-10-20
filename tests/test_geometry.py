"""chemistry_helpers.geometry: Kabsch superposition, and its equivalence with every copy
listed in the 2026-10 duplication review (group 6).

The ``copy_*`` functions below are the other copies' bodies reproduced verbatim (read-only
snapshot, 2026-10-20) so the shared function can be shown to agree with them before any
consumer migrates. Rotation convention: column vector (``R @ x``) except
protein_validation_analysis, whose ``kabsch_rotation`` is row-vector (``p @ M``): there
``M == R.T``, asserted explicitly.
"""
import numpy as np
import pytest

from chemistry_helpers.geometry import kabsch_rmsd, kabsch_rotation, superpose


def rot(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return rz @ ry @ rx


# ---- verbatim copies ------------------------------------------------------------------

def copy_atom_reorder_transform(p, q):          # atom_reorder.mapping._kabsch_transform
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    u, _s, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T, pc, qc


def copy_atom_reorder_rmsd(query_coords, reference_coords):   # atom_reorder.mapping.kabsch_rmsd
    import math
    if query_coords.shape != reference_coords.shape:
        raise ValueError("coordinate arrays must have the same shape")
    if len(query_coords) == 0:
        return 0.0
    p = query_coords - query_coords.mean(axis=0)
    q = reference_coords - reference_coords.mean(axis=0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    rotated = (rotation @ p.T).T
    diff = rotated - q
    return float(math.sqrt(float((diff * diff).sum()) / len(p)))


def copy_anchored_rotation(ref_points, ref_anchor, query_points, query_anchor):  # atom_reorder.place_hydrogens
    a = np.asarray(ref_points, dtype=float) - np.asarray(ref_anchor, dtype=float)
    b = np.asarray(query_points, dtype=float) - np.asarray(query_anchor, dtype=float)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def copy_website_kabsch(query_points, reference_points):   # website structure_rmsd._kabsch == remap._kabsch
    from numpy import array, diag, sign
    from numpy.linalg import det, svd
    p = array(query_points, dtype=float)
    q = array(reference_points, dtype=float)
    (p_centroid, q_centroid) = (p.mean(axis=0), q.mean(axis=0))
    (u, _s, vt) = svd((p - p_centroid).T @ (q - q_centroid))
    d = sign(det(vt.T @ u.T))
    return (vt.T @ diag([1.0, 1.0, d]) @ u.T, p_centroid, q_centroid)


def copy_microstates_rmsd(a, b):                # atb_microstates.conformer_symmetry.kabsch_rmsd
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pa = a - a.mean(axis=0)
    pb = b - b.mean(axis=0)
    h = pb.T @ pa
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    moved = pb @ rot.T
    dev = np.linalg.norm(moved - pa, axis=1)
    return float(np.sqrt((dev ** 2).mean())), float(dev.max())


def copy_microstates_script_rmsd(x, y):          # hessian_substitute_test._kabsch_rmsd / _node.kabsch_rmsd
    x = np.asarray(x, float) - np.mean(x, axis=0)
    y = np.asarray(y, float) - np.mean(y, axis=0)
    u, s, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((x @ r - y) ** 2).sum() / len(x)))


def copy_molgraph_rmsd(ref_xyz, query_xyz, mapping):     # atb_molgraph/benchmarks/mapping.kabsch_rmsd
    keys = sorted(mapping)
    p = np.array([ref_xyz[i] for i in keys], dtype=float)
    q = np.array([query_xyz[mapping[i]] for i in keys], dtype=float)
    p -= p.mean(axis=0); q -= q.mean(axis=0)
    u, s, vt = np.linalg.svd(q.T @ p)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((q @ rotation - p) ** 2).sum() / len(keys)))


def copy_scan_kabsch(mobile, target):            # dihedral_scan_service.capped_residue.kabsch
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    covariance = (mobile - mc).T.dot(target - tc)
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T.dot(u.T)))
    rotation = vt.T.dot(np.diag([1.0, 1.0, d])).dot(u.T)
    return rotation, tc - rotation.dot(mc)


def copy_pva_kabsch_rotation(p, q):              # protein_validation_analysis.geometry (row-vector; p, q pre-centred)
    v, s, wt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(v @ wt))
    v[:, -1] *= d
    return v @ wt


def copy_pva_rmsd_after_fit(ref, x):             # protein_validation_analysis.geometry.rmsd_after_fit
    p = x - x.mean(axis=0)
    q = ref - ref.mean(axis=0)
    p = p @ copy_pva_kabsch_rotation(p, q)
    return float(np.sqrt(np.mean(np.sum((p - q) ** 2, axis=1))))


def copy_rotamer_identity_rmsd(a, b):            # rotamer_ladder.identity.kabsch_rmsd (batched, singular-value form)
    a = np.asarray(a, dtype=float)
    bs = np.asarray(b, dtype=float)[None]
    n = a.shape[0]
    a0 = a - a.mean(axis=0)
    b0 = bs - bs.mean(axis=1, keepdims=True)
    h = np.einsum("pni,nj->pij", b0, a0)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.matmul(u, vt)))
    s = s.copy()
    s[:, -1] *= np.where(d == 0, 1.0, d)
    msd = ((a0 * a0).sum() + (b0 * b0).sum(axis=(1, 2)) - 2.0 * s.sum(axis=1)) / n
    return float(np.sqrt(np.clip(msd, 0.0, None))[0])


def copy_rotamer_gmx_rmsd(a, b):                 # rotamer_ladder.mm.gmx.kabsch_rmsd
    a = np.asarray(a, dtype=float) - np.mean(a, axis=0)
    b = np.asarray(b, dtype=float) - np.mean(b, axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ rotation - b) ** 2).sum(-1).mean()))


# ---- inputs ---------------------------------------------------------------------------

def _cases():
    rng = np.random.default_rng(20261020)
    cases = {}
    for n in (3, 4, 10, 60):
        base = rng.normal(size=(n, 3))
        cases[f"random_n{n}"] = (base @ rot(*rng.uniform(-3, 3, 3)).T + rng.normal(size=3),
                                 base + 0.05 * rng.normal(size=(n, 3)))
    base = rng.normal(size=(8, 3))
    cases["rigid"] = (base @ rot(0.3, 0.9, -0.4).T + [1.0, 2.0, 3.0], base)
    cases["identical"] = (base.copy(), base.copy())
    cases["reflected"] = (base * [1.0, 1.0, -1.0], base)
    cases["reflected_noisy"] = (base * [-1.0, 1.0, 1.0] + 0.01 * rng.normal(size=base.shape), base)
    t = rng.normal(size=(6, 1))
    cases["collinear"] = (t * [1.0, 2.0, -1.0], rng.normal(size=(6, 1)) * 0 + t * [0.0, 1.0, 1.0])
    plane = np.c_[rng.normal(size=(7, 2)), np.zeros(7)]
    cases["planar"] = (plane, plane @ rot(0.7, -0.2, 1.1).T)
    cases["planar_mirror"] = (plane * [1.0, -1.0, 1.0], plane)
    cases["planar_to_3d"] = (plane, rng.normal(size=(7, 3)))
    cases["n3_triangle"] = (rng.normal(size=(3, 3)), rng.normal(size=(3, 3)))
    cases["single_point"] = (np.array([[1.0, 2.0, 3.0]]), np.array([[-1.0, 0.0, 4.0]]))
    cases["all_coincident"] = (np.ones((4, 3)), np.zeros((4, 3)))
    return cases


CASES = _cases()
DEGENERATE = {"collinear", "single_point", "all_coincident", "identical"}  # rotation not unique (or p==q: SVD freedom)
NAMES = list(CASES)


def _is_proper(r):
    return np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


# ---- tests of the new module ----------------------------------------------------------

def test_blind_to_rigid_motion_and_not_to_reflection():
    """Ported from atom_reorder tests/test_mapping.py::test_kabsch_is_blind_to_rigid_motion_and_not_to_reflection."""
    coords = np.random.default_rng(1).normal(size=(10, 3))
    rotated = coords @ rot(0.3, 0.9, -0.4).T + np.array([1.0, 2.0, 3.0])
    assert kabsch_rmsd(rotated, coords) == pytest.approx(0.0, abs=1e-9)
    mirrored = coords * np.array([1.0, 1.0, -1.0])
    assert kabsch_rmsd(mirrored, coords) > 0.1


def test_column_vector_convention_and_superpose_triple():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(9, 3))
    true = rot(0.4, -1.0, 2.0)
    p = q @ true.T + [3.0, -1.0, 0.5]          # p = true @ q + t, so superposing p onto q needs true.T
    r, pc, qc = superpose(p, q)
    assert _is_proper(r)
    np.testing.assert_allclose(r, true.T, atol=1e-12)
    moved = (r @ (p - pc).T).T + qc            # R @ (x - pc) + qc
    np.testing.assert_allclose(moved, q, atol=1e-12)
    np.testing.assert_allclose(kabsch_rotation(p, q), r)


def test_empty_and_shape_errors():
    z = np.zeros((0, 3))
    assert kabsch_rmsd(z, z) == 0.0
    r, pc, qc = superpose(z, z)
    np.testing.assert_array_equal(r, np.eye(3))
    with pytest.raises(ValueError):
        kabsch_rmsd(np.zeros((3, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        kabsch_rmsd(np.zeros((3, 2)), np.zeros((3, 2)))


@pytest.mark.parametrize("name", NAMES)
def test_always_proper_rotation_and_never_a_reflection(name):
    p, q = CASES[name]
    assert _is_proper(kabsch_rotation(p, q))


def test_reflected_input_has_nonzero_rmsd():
    p, q = CASES["reflected"]
    assert kabsch_rmsd(p, q) > 0.1


# ---- equivalence with every copy ------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_rmsd_equals_every_copy(name):
    p, q = CASES[name]
    ours = kabsch_rmsd(p, q)
    n = len(p)
    ident = {i: i for i in range(n)}
    got = {
        "atom_reorder.kabsch_rmsd": copy_atom_reorder_rmsd(p, q),
        "website/remap/atom_reorder transform": float(np.sqrt(
            (((p - copy_website_kabsch(p, q)[1]) @ copy_website_kabsch(p, q)[0].T
              - (q - q.mean(axis=0))) ** 2).sum() / n)),
        "microstates.conformer_symmetry": copy_microstates_rmsd(q, p)[0],     # (a, b): b onto a
        "microstates scripts": copy_microstates_script_rmsd(p, q),
        "atb_molgraph benchmarks": copy_molgraph_rmsd(q, p, ident),            # (ref, query, mapping)
        "protein_validation_analysis": copy_pva_rmsd_after_fit(q, p),          # (ref, x)
        "rotamer_ladder.identity": copy_rotamer_identity_rmsd(q, p),
        "rotamer_ladder.mm.gmx": copy_rotamer_gmx_rmsd(p, q),
    }
    for label, value in got.items():
        if label == "rotamer_ladder.identity":
            # singular-value form msd = (|a|^2 + |b|^2 - 2 sum s)/n cancels catastrophically:
            # near RMSD 0 it is only good to ~sqrt(eps) = 1e-8 in RMSD (1e-16 in msd).
            # Documented migration difference; equal to 1e-12 whenever the RMSD is not ~0.
            assert abs(value ** 2 - ours ** 2) < 1e-12, (name, label, value, ours)
            if ours > 1e-3:
                assert abs(value - ours) < 1e-12, (name, label, value, ours)
            continue
        assert abs(value - ours) < 1e-12, (name, label, value, ours)
    # rmsd is symmetric in its arguments
    assert abs(kabsch_rmsd(q, p) - ours) < 1e-12


@pytest.mark.parametrize("name", NAMES)
def test_microstates_max_deviation_is_reproducible_from_superpose(name):
    p, q = CASES[name]
    r, pc, qc = superpose(q, p)               # microstates moves b onto a
    dev = np.linalg.norm((r @ (q - pc).T).T + qc - p, axis=1)
    _rmsd, dmax = copy_microstates_rmsd(p, q)
    assert abs(dev.max() - dmax) < 1e-12


@pytest.mark.parametrize("name", [n for n in NAMES if n not in DEGENERATE])
def test_rotation_equals_column_vector_copies(name):
    p, q = CASES[name]
    r, pc, qc = superpose(p, q)
    for rr, a, b in (copy_atom_reorder_transform(p, q), copy_website_kabsch(p, q)):
        np.testing.assert_allclose(rr, r, atol=1e-12)
        np.testing.assert_allclose(a, pc, atol=1e-12)
        np.testing.assert_allclose(b, qc, atol=1e-12)
    ref = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(copy_anchored_rotation(p, p.mean(axis=0), q, q.mean(axis=0)), r, atol=1e-12)
    rs, ts = copy_scan_kabsch(p, q)
    np.testing.assert_allclose(rs, r, atol=1e-12)
    np.testing.assert_allclose(ts, qc - r @ pc, atol=1e-12)
    del ref


@pytest.mark.parametrize("name", [n for n in NAMES if n not in DEGENERATE])
def test_pva_rotation_is_the_transpose(name):
    p, q = CASES[name]
    pc, qc = p - p.mean(axis=0), q - q.mean(axis=0)
    np.testing.assert_allclose(copy_pva_kabsch_rotation(pc, qc), kabsch_rotation(p, q).T, atol=1e-12)


@pytest.mark.parametrize("name", sorted(DEGENERATE))
def test_degenerate_rotations_are_optimal_even_when_not_unique(name):
    """Collinear / single-point / coincident / identical: copies may pick different
    optimal rotations (the SVD's free directions); only the RMSD is contractual."""
    p, q = CASES[name]
    for rr in (kabsch_rotation(p, q), copy_atom_reorder_transform(p, q)[0]):
        assert _is_proper(rr)
        moved = (p - p.mean(axis=0)) @ rr.T
        d = np.sqrt(((moved - (q - q.mean(axis=0))) ** 2).sum() / len(p))
        assert abs(d - kabsch_rmsd(p, q)) < 1e-12
