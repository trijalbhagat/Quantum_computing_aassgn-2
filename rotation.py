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
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    sigma = np.array([
    [[0, 1], [1, 0]],       
    [[0, -1j], [1j, 0]], 
    [[1, 0], [0, -1]]       
    ], dtype=DTYPE)
    n: np.ndarray
    

    det_g = np.linalg.det(g)
    alpha= float(0.5 * np.angle(det_g))
    theta=float(2 * np.arccos(0.5 * np.trace(g/alpha)).real)
    g_dash=g* np.exp(-1j* alpha)
    if np.isclose(np.sin(theta/2), 0.0):
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n_x=0.5j*np.trace(sigma[0]@g_dash)/np.sin(theta/2)
        n_y=0.5j*np.trace(sigma[1]@g_dash)/np.sin(theta/2)
        n_z=0.5j*np.trace(sigma[2]@g_dash)/np.sin(theta/2)
        n = np.array([n_x, n_y, n_z], dtype=float)
    return Bloch(alpha,n,theta)


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
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    # TODO(student): implement using the steps above.
    global_phase=b.alpha
    # z=np.sin(beta)*np.sin(gama-alpha)
    # y=np.sin(beta)*np.cos(gama-alpha)
    # x=np.cos(beta)*np.sin(gama+alpha)
    # w=np.cos(beta)*np.cos(gama+alpha)
    w=np.cos(b.theta/2)
    x=np.sin(b.theta/2)*np.dot(b.n,a1)
    y=np.sin(b.theta/2)*np.dot(b.n,a2)
    z=np.sin(b.theta/2)*np.dot(b.n,a3)
   # gama-alpha=np.arctan2(z/y)
   # gama + alpha= np.arctan2(x/w)
    gama= 0.5*(np.arctan2(z,y)+np.arctan2(x,w))
    alpha= 0.5*(-np.arctan2(z,y)+np.arctan2(x,w))
    beta=np.arcsin(z/np.sin(gama-alpha))
    alpha = float(np.mod(alpha, TWO_PI))
    beta = float(np.mod(beta, TWO_PI))
    gamma = float(np.mod(gama, TWO_PI))
    return(alpha,beta,gamma,global_phase)


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
sigma = np.array([
    [[0, 1], [1, 0]],       
    [[0, -1j], [1j, 0]], 
    [[1, 0], [0, -1]]       
    ], dtype=DTYPE)
def from_axis_angle(b: Bloch) -> np.ndarray:
    """Build a 2x2 unitary from its Bloch form: a global phase times a rotation
    by angle b.theta about axis b.n (inverse of to_bloch).

        G = e^{i b.alpha} (cos(b.theta/2) I - i sin(b.theta/2) (b.n . sigma))

    where (b.n . sigma) = n_x X + n_y Y + n_z Z. Assumes b.n is a unit vector.
    """
   
    G=np.exp(1j*b.alpha)*(np.cos(0.5*b.theta)*np.eye-1j*np.sin(0.5*b.theta)*np.dot(b.n,sigma))
    return G
    # TODO: implement using the formula above.
    raise NotImplementedError("from_axis_angle is not implemented yet")


def Rz(theta: float) -> np.ndarray:
    """Rotation about the z axis (no global phase):

    Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}).
    """
    R_z=np.array([np.exp(-1j*theta),0],[0,np.exp(1j*theta)], dtype=DTYPE)
    return R_z
    # TODO: implement (hint: from_axis_angle about axis [0, 0, 1]).
    raise NotImplementedError("Rz is not implemented yet")


def Ry(theta: float) -> np.ndarray:
    """Rotation about the y axis (no global phase):

    Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]].
    """
    R_y=np.array([np.cos(0.5*theta),-np.sin(0.5*theta)],[np.sin(0.5*theta),np.cos(0.5*theta)],dtype=DTYPE)
    return R_y
    # TODO: implement (hint: from_axis_angle about axis [0, 1, 0]).
    raise NotImplementedError("Ry is not implemented yet")


def euler_angles_zyz(u: np.ndarray) -> tuple[float, float, float, float]:
    """ZYZ Euler decomposition of a 2x2 unitary: angles (alpha, beta, gamma, delta)
    with

        u = e^{i alpha} Rz(beta) Ry(gamma) Rz(delta).

    alpha is the global phase (arg(det u)/2); the rest come from S = e^{-i alpha} u
    in SU(2), where s00 = cos(gamma/2) e^{-i(beta+delta)/2} and
    s10 = sin(gamma/2) e^{i(beta-delta)/2}. When gamma = 0 (s10 = 0), beta/delta are
    split arbitrarily (gimbal lock); the identity still holds.
    """
    z=np.array(0,0,-1)
    y=np.array(0,-1,0)
    z_y=np.cross(a1,a2)
    b=to_bloch(u)
    global_phase=b.alpha
    # z=np.sin(beta)*np.sin(gama-alpha)
    # y=np.sin(beta)*np.cos(gama-alpha)
    # x=np.cos(beta)*np.sin(gama+alpha)
    # w=np.cos(beta)*np.cos(gama+alpha)
    w=np.cos(b.theta/2)
    x=np.sin(b.theta/2)*np.dot(b.n,z)
    y=np.sin(b.theta/2)*np.dot(b.n,y)
    z=np.sin(b.theta/2)*np.dot(b.n,z_y)
   # gama-alpha=np.arctan2(z/y)
   # gama + alpha= np.arctan2(x/w)
    gama= 0.5*(np.arctan2(z,y)+np.arctan2(x,w))
    alpha= 0.5*(-np.arctan2(z,y)+np.arctan2(x,w))
    beta=np.arcsin(z/np.sin(gama-alpha))
    alpha = float(np.mod(alpha, TWO_PI))
    beta = float(np.mod(beta, TWO_PI))
    gamma = float(np.mod(gama, TWO_PI))
    return(global_phase,alpha,beta,gamma)
    # TODO: implement using the relations above.
    raise NotImplementedError("euler_angles_zyz is not implemented yet")


def unitary2_sqrt(u: np.ndarray) -> np.ndarray:
    """Principal square root: a 2x2 unitary V with V @ V == u, phase included.
    Take the Bloch form of u and halve both alpha and theta (same axis); squaring
    back doubles them, reproducing u exactly.
    G = e^{i b.alpha} (cos(b.theta/2) I - i sin(b.theta/2) (b.n . sigma))
    """
    b=to_bloch(u)
    V_alpha=0.5*b.alpha
    V_theta=0.5*b.theta
    v=np.exp(1j*V_alpha)*np.cos(0.5*V_theta)*np.eye-1j*np.sin(0.5*V_theta)*(np.dot(b.n,sigma))
    return v
    # TODO: implement (hint: to_bloch, halve alpha and theta, from_axis_angle).
    raise NotImplementedError("unitary2_sqrt is not implemented yet")


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
    """Flatten an alternating (T-power, H-power, ...) exponent list into a literal
    string of 'H'/'T' gates (left-to-right). Even indices are T, odd indices are H.
    """
    T="T"
    H="H"
    string=""
    for i in range(len(word)):
        if(i%2==0):
            string=string+T*word[i]
        else:
            string=string+H*word[i]
    return string
    # TODO: implement.
    raise NotImplementedError("expand_word is not implemented yet")

# flat H/T strings for the two building-block words (computed once expand_word works)
# M1_STR = expand_word(M1_WORD)
# M2_STR = expand_word(M2_WORD)


def gates_to_unitary(gates: str) -> np.ndarray:
    """The 2x2 unitary of a flat H/T gate string (left-to-right product)."""
    T=np.array([[1,0],[0,np.exp(0.25j*np.pi)]],dtype=DTYPE)
    aray=np.array([[1,0],[0,1]],dtype=DTYPE)
    lis=list(gates)
    for i in range(len(gates)):
        if(lis[i]=='H'):
            aray=aray@H
        else:
            aray=aray@T
    return aray
    # TODO: implement (multiply H / T for each char, starting from I).
    raise NotImplementedError("gates_to_unitary is not implemented yet")


def invert_gates(gates: str) -> str:
    """Inverse of a flat H/T word: reverse the gate order and invert each gate.
    H^-1 = H; the {H, T} basis has no T-dagger, so T^-1 must be spelled as T^7.
    """
    sring=gates[::-1]
    sring = sring.replace('T', 'TTTTTTT')
    return sring
    # TODO: implement.
    raise NotImplementedError("invert_gates is not implemented yet")

def power_gates(base: str, k: int) -> str:
    """The k-th power of a flat H/T word: base repeated k times. Negative k uses the
    inverse word (invert_gates).
    """
    if(k>=0):
        base=base*k
    else:
        base=invert_gates(base)*-k
    return base
    # TODO: implement.
    raise NotImplementedError("power_gates is not implemented yet")

def approximate_in_ht(u: np.ndarray, error: float) -> str:
    """Approximate a 2x2 unitary `u` by a flat H/T word (up to global phase) to the
    angular tolerance `error` (smaller -> longer, more accurate).

    Use decompose_2x2 to get the powers (k, l, m) with u ~= M1^k M2^l M1^m, then
    assemble the word:

        power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m).
    """
    b=decompose_2x2(u,error)
    sring=power_gates(M1_STR,b.k)+power_gates(M2_STR,b.l)+power_gates(M1_STR,b.m)
    # TODO: implement using decompose_2x2 and power_gates.
    raise NotImplementedError("approximate_in_ht is not implemented yet")