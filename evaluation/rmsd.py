import numpy as np

def compute_rmsd(a, b, aligned=False):
    """
    RMSD between corresponding 3D coordinates.

    a: reference coordinates, shape (N, 3)
    b: generated coordinates, shape (N, 3)

    aligned=False: calculate coordinate RMSD directly
    aligned=True: rigidly superpose b onto a using Kabsch
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Both inputs must have shape (N, 3)")
    if len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Coordinates must be nonempty and finite")

    if aligned:
        a_center = a.mean(axis=0)
        b_center = b.mean(axis=0)

        A = a - a_center
        B = b - b_center

        # Kabsch optimal rigid-body rotation
        H = B.T @ A
        U, _, Vt = np.linalg.svd(H)

        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[-1, -1] = -1

        R = U @ D @ Vt

        # Rotate and translate generated onto reference
        b = B @ R + a_center

    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))