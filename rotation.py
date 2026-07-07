import numpy as np
from dataclasses import dataclass
# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi

@dataclass
class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle
    
   
def to_bloch(g: np.ndarray) -> Bloch:
    """Recover Bloch form of 2x2 unitary g."""

    g = np.asarray(g, dtype=DTYPE)

    det_g = np.linalg.det(g)
    alpha = float(0.5 * np.angle(det_g))

    # Remove global phase
    g_dash = np.exp(-1j * alpha) * g

    tr = np.trace(g_dash)
    cos_half_theta = np.clip((0.5 * tr).real, -1.0, 1.0)

    theta = float(2 * np.arccos(cos_half_theta))

    s = np.sin(theta / 2)

    if np.isclose(s, 0.0):
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n_x = (0.5j * np.trace(sigma[0] @ g_dash) / s).real
        n_y = (0.5j * np.trace(sigma[1] @ g_dash) / s).real
        n_z = (0.5j * np.trace(sigma[2] @ g_dash) / s).real

        n = np.array([n_x, n_y, n_z], dtype=float)

        norm = np.linalg.norm(n)
        if not np.isclose(norm, 0.0):
            n = n / norm

    return Bloch(alpha, n, theta)


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
n1 = np.array([1, 0, 0])
n2 = np.array([0,1,0])

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor Bloch rotation into Rn1(alpha) Rn2(beta) Rn1(gamma)."""

    global_phase = b.alpha

    w = np.cos(b.theta / 2)
    x = np.sin(b.theta / 2) * np.dot(b.n, a1)
    y = np.sin(b.theta / 2) * np.dot(b.n, a2)
    z = np.sin(b.theta / 2) * np.dot(b.n, a3)

    p = np.arctan2(z, y)  # gamma - alpha
    q = np.arctan2(x, w)  # gamma + alpha

    gamma = 0.5 * (p + q)
    alpha = 0.5 * (q - p)

    # Much safer than beta = arcsin(z / sin(...))
    beta = np.arctan2(
        np.sqrt(y * y + z * z),
        np.sqrt(w * w + x * x),
    )

    alpha = float(np.mod(alpha, TWO_PI))
    beta = float(np.mod(beta, TWO_PI))
    gamma = float(np.mod(gamma, TWO_PI))

    return alpha, beta, gamma, global_phase



def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    ang_modded=np.mod(angle,2*np.pi)
    def angular_dist(a:float, b:float) -> float:
        return(min(abs(a-b), 2*np.pi-abs(a-b)))
    
    if angular_dist(0.0, ang_modded) <= tolerance:
        return 0

    k = 1

    while True:
        check_value = np.mod(k * LAMBDA_PI, TWO_PI)

        if angular_dist(check_value, ang_modded) <= tolerance:
            return k

        k += 1
    # TODO(student): implement using the hint above.
    raise NotImplementedError("approx_angle_with_tolerance is not implemented yet")


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """

    alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))

    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)

    return(k,l,m)
    # TODO(student): implement using the steps above.
    raise NotImplementedError("decompose_2x2 is not implemented yet")

sigma = np.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=DTYPE,
)

def from_axis_angle(b: Bloch) -> np.ndarray:
    """Build 2x2 unitary from Bloch form."""

    n_dot_sigma = (
        b.n[0] * sigma[0]
        + b.n[1] * sigma[1]
        + b.n[2] * sigma[2]
    )

    return np.exp(1j * b.alpha) * (
        np.cos(b.theta / 2) * np.eye(2, dtype=DTYPE)
        - 1j * np.sin(b.theta / 2) * n_dot_sigma
    )

def Rz(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ],
        dtype=complex,
    )


def Ry(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )



def euler_angles_zyz(u: np.ndarray):
    """Return alpha, beta, gamma, delta such that

        U = exp(i alpha) Rz(beta) Ry(gamma) Rz(delta)

    for a 2x2 unitary U.
    """

    u = np.asarray(u, dtype=complex)

    if u.shape != (2, 2):
        raise ValueError("u must be 2x2")

    det_u = np.linalg.det(u)
    alpha = 0.5 * np.angle(det_u)

    # Remove global phase.
    su = np.exp(-1j * alpha) * u

    a = su[0, 0]
    b = su[0, 1]

    cos_half_gamma = np.clip(abs(a), 0.0, 1.0)
    gamma = 2 * np.arccos(cos_half_gamma)

    # Special case: gamma approximately 0
    if np.isclose(np.sin(gamma / 2), 0.0, atol=1e-12):
        beta = -2 * np.angle(a)
        delta = 0.0

    # Special case: gamma approximately pi
    elif np.isclose(np.cos(gamma / 2), 0.0, atol=1e-12):
        beta = -2 * np.angle(-b)
        delta = 0.0

    else:
        # su[0,0] = exp(-i(beta+delta)/2) cos(gamma/2)
        # su[0,1] = -exp(-i(beta-delta)/2) sin(gamma/2)
        p = -2 * np.angle(a)       # beta + delta
        m = -2 * np.angle(-b)      # beta - delta

        beta = 0.5 * (p + m)
        delta = 0.5 * (p - m)

    return alpha, beta, gamma, delta



def unitary2_sqrt(u: np.ndarray) -> np.ndarray:
    """Matrix square root of a 2x2 unitary.

    Returns V such that V @ V ≈ u.
    """

    u = np.asarray(u, dtype=complex)

    if u.shape != (2, 2):
        raise ValueError("u must be 2x2")

    eigvals, eigvecs = np.linalg.eig(u)

    sqrt_diag = np.diag(np.sqrt(eigvals))

    return eigvecs @ sqrt_diag @ np.linalg.inv(eigvecs)



# ---------------------------------------------------------------------------
# H/T word machinery for approximating a 2x2 unitary in {H, T} (see cpp/src/HT.h).
#
# M1, M2 are short H/T words that realize rotations by THETA_M = 2*LAMBDA_PI about
# the axes a1, a2. A word is a flat string of 'H'/'T' characters, read left-to-right
# as a matrix product (leftmost char = leftmost/outermost factor).
# ---------------------------------------------------------------------------

# alternating (T-power, H-power, ...) exponents, starting with T
M1_WORD = [7, 1, 1, 1]
M2_WORD = [2, 1, 1, 1, 6, 1, 7, 1, 5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 7, 1, 6]



def expand_word(word: list[int]) -> str:
    """Flatten alternating T-power, H-power exponents."""

    result = ""

    for i, power in enumerate(word):
        if i % 2 == 0:
            result += "T" * power
        else:
            result += "H" * power

    return result
    # TODO: implement.
    raise NotImplementedError("expand_word is not implemented yet")

# flat H/T strings for the two building-block words (computed once expand_word works)
# M1_STR = expand_word(M1_WORD)
# M2_STR = expand_word(M2_WORD)

M1_STR = expand_word(M1_WORD)
M2_STR = expand_word(M2_WORD)


def gates_to_unitary(gates: str) -> np.ndarray:
    """The 2x2 unitary of a flat H/T gate string."""

    T = np.array(
        [
            [1, 0],
            [0, np.exp(1j * np.pi / 4)],
        ],
        dtype=DTYPE,
    )

    result = np.eye(2, dtype=DTYPE)

    for ch in gates:
        if ch == "H":
            result = result @ H
        elif ch == "T":
            result = result @ T
        else:
            raise ValueError(f"Unknown gate character: {ch}")

    return result

    # TODO: implement (multiply H / T for each char, starting from I).
    raise NotImplementedError("gates_to_unitary is not implemented yet")


def invert_gates(gates: str) -> str:
    """Inverse of H/T word."""

    result = ""

    for ch in reversed(gates):
        if ch == "H":
            result += "H"
        elif ch == "T":
            result += "T" * 7
        else:
            raise ValueError(f"Unknown gate character: {ch}")

    return result
    # TODO: implement.
    raise NotImplementedError("invert_gates is not implemented yet")

def power_gates(base: str, k: int) -> str:
    """The k-th power of a flat H/T word."""

    if k >= 0:
        return base * k

    return invert_gates(base) * (-k)
    # TODO: implement.
    raise NotImplementedError("power_gates is not implemented yet")

def ht_matrix(ch: str) -> np.ndarray:
    """Return matrix for one H/T gate."""

    if ch == "H":
        return (1 / np.sqrt(2)) * np.array(
            [
                [1, 1],
                [1, -1],
            ],
            dtype=complex,
        )

    if ch == "T":
        return np.array(
            [
                [1, 0],
                [0, np.exp(1j * np.pi / 4)],
            ],
            dtype=complex,
        )

    raise ValueError(f"Unknown gate: {ch}")


def gates_to_unitary(word: str) -> np.ndarray:
    """Convert an H/T word into a 2x2 unitary.

    Leftmost char = leftmost matrix factor.

    Example:
        word = "HTH" means H @ T @ H
    """

    result = np.eye(2, dtype=complex)

    for ch in word:
        result = result @ ht_matrix(ch)

    return result


def error_up_to_phase_2x2(a: np.ndarray, b: np.ndarray) -> float:
    """Compare two 2x2 matrices ignoring global phase."""

    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)

    if not np.all(np.isfinite(a)):
        return np.inf

    if not np.all(np.isfinite(b)):
        return np.inf

    overlap = np.vdot(b, a)

    if np.isclose(abs(overlap), 0.0):
        return float(np.max(np.abs(a - b)))

    phase = overlap / abs(overlap)

    return float(np.max(np.abs(a - phase * b)))


def approximate_in_ht(u: np.ndarray, error: float) -> str:
    """Approximate a 2x2 unitary using H/T words.

    Uses:
        U ≈ M1^k M2^l M1^m
    """

    k, l, m = decompose_2x2(u, error)

    return (
        power_gates(M1_STR, k)
        + power_gates(M2_STR, l)
        + power_gates(M1_STR, m)
    )