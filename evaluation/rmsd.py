import numpy as np

def compute_rmsd(a, b, aligned=False, invalid="nan"):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Both inputs must have shape (N, 3)")

    # Find valid coordinate pairs
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)

    if not valid.all():
        if invalid == "raise":
            raise ValueError("Coordinates contain NaN or Inf")
        elif invalid == "nan":
            return float("nan")
        elif invalid == "omit":
            a = a[valid]
            b = b[valid]
        else:
            raise ValueError("invalid must be 'raise', 'nan', or 'omit'")

    # Ensure enough valid atoms remain
    if len(a) < (3 if aligned else 1):
        return float("nan")

    if aligned:
        a_center = a.mean(axis=0)
        b_center = b.mean(axis=0)

        A = a - a_center
        B = b - b_center

        H = B.T @ A
        U, _, Vt = np.linalg.svd(H)

        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[-1, -1] = -1

        R = U @ D @ Vt
        b = B @ R + a_center

    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))