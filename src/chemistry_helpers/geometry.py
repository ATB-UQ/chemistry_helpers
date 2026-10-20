"""Pure-numpy rigid-body superposition (Kabsch), the one shared implementation.

Convention (read before swapping a local copy for this): **column vectors**. ``R`` is
applied as ``R @ x`` to a single point ``x`` (shape (3,)); for an (n, 3) array of points
that is ``points @ R.T``. A proper rotation only (det +1): the reflection correction
(sign of ``det(V U^T)`` on the smallest singular direction) means a mirror image never
superposes, so an enantiomer does not score a small RMSD.

``superpose(p, q)`` returns ``(R, centroid_p, centroid_q)`` such that
``R @ (x - centroid_p) + centroid_q`` moves a point of the ``p`` frame into the ``q`` frame.
A row-vector caller (``p_centred @ M``) uses ``M = R.T``.

Degenerate input: collinear points leave the rotation about the line free, planar points
are fine, three points are fine; the RMSD is still the optimum, the rotation is then one of
the optimal ones. Empty input gives the identity and zero centroids / RMSD 0.0.
"""
import numpy as np


def _as_points(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("coordinate arrays must have the same shape")
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3)")
    return p, q


def superpose(p, q):
    """``(R, centroid_p, centroid_q)``: the proper rotation (column-vector convention)
    that best superposes ``p`` onto ``q`` (same atom order) after centring both."""
    p, q = _as_points(p, q)
    if len(p) == 0:
        return np.eye(3), np.zeros(3), np.zeros(3)
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    u, _s, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T, pc, qc


def kabsch_rotation(p, q):
    """The 3x3 proper rotation ``R`` with ``R @ (p_i - mean p) ~= q_i - mean q``."""
    return superpose(p, q)[0]


def kabsch_rmsd(p, q):
    """RMSD of ``p`` onto ``q`` (same atom order) after optimal translation and proper
    rotation, in the input unit. 0.0 for empty input."""
    p, q = _as_points(p, q)
    if len(p) == 0:
        return 0.0
    rotation, pc, qc = superpose(p, q)
    diff = (p - pc) @ rotation.T - (q - qc)
    return float(np.sqrt((diff * diff).sum() / len(p)))
