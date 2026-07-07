"""Decompose an arbitrary unitary into the universal basis {H, T, CNOT}.

A unitary is
lowered in stages:

    1. twolevel_decomposition:  Unitary     -> list[TwoLevel]
    2. decompose_twolevel:      TwoLevel    -> SingleQubitGate + ControlledU
    3. decompose_controlledU:   ControlledU -> CU + CNOT
    4. decompose_cu:            CU          -> SingleQubitGate + CNOT
    5. decompose_to_ht:         SingleQubitGate -> H / T words (using rotation.py)

Numpy types: a `Unitary` (N x N) and a 2x2 gate block are
both np.ndarray (complex128); a `ComplexVec` is a 1-D np.ndarray. A `Circuit` is a
Python list of gate objects, each exposing `to_unitary()`; gates are stored in
order of application (the first gate is applied first, i.e. the rightmost matrix
factor).

Every function/method below is a stub for you to implement; See "03 - Completing the Decomposition.pdf" for the recommended order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

import rotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def num_qubits(N: int) -> int:
    """Number of qubits n such that N == 2^n (N is the unitary / two-level size)."""
    n = int(np.log2(N))

    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    return n
    # TODO: implement.
    raise NotImplementedError("num_qubits is not implemented yet")


# ---------------------------------------------------------------------------
# Gate representations
#
# Each is a sparse description of an operation with a `to_unitary()` returning the
# full N x N matrix. As the decomposition progresses, gates get rewritten into
# simpler ones. The 2x2 block `unitary` is a (2, 2) complex ndarray.
# ---------------------------------------------------------------------------


@dataclass
class TwoLevel:
    """A two-level unitary: acts as the 2x2 `unitary` on the two basis states
    `level0`, `level1` of a size-`size` register, and as identity everywhere else.
    """

    size: int
    level0: int
    level1: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Expand to the full `size` x `size` matrix: identity except the 2x2 block
        placed at rows/cols (level0, level1).
        """
        matrix = np.eye(self.size, dtype=complex)

        matrix[self.level0, self.level0] = self.unitary[0, 0]
        matrix[self.level0, self.level1] = self.unitary[0, 1]
        matrix[self.level1, self.level0] = self.unitary[1, 0]
        matrix[self.level1, self.level1] = self.unitary[1, 1]
        return matrix
        # TODO: implement.
        raise NotImplementedError("TwoLevel.to_unitary is not implemented yet")


@dataclass
class SingleQubitGate:
    """A single-qubit gate acting as the 2x2 `unitary` on `qubit` of an n-qubit
    register (N = 2^n), identity on the other qubits.
    """

    n: int
    qubit: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Expand the 2x2 gate to the full 2^n x 2^n matrix."""

        I = np.eye(2, dtype=complex)
        full_matrix = np.array([[1]], dtype=complex)

        for q in range(self.n):
            if q == self.qubit:
                full_matrix = np.kron(full_matrix, self.unitary)
            else:
                full_matrix = np.kron(full_matrix, I)

        return full_matrix


@dataclass
class ControlledU:
    """A fully-controlled single-qubit gate C^k(U): apply the 2x2 `unitary` to
    `target` iff every other qubit is 1. Controls are always conditioned on 1, so
    their positions need not be stored.
    """

    n: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity everywhere except the single controlled block: the pair (all
        ones except the target bit, all ones).
        """
        matrix=np.identity(2**self.n)
        c=2**self.n-1-2**(self.n - 1 - self.target)
        t=2**self.n-1
        matrix[c,c] = self.unitary[0, 0]
        matrix[c,t] = self.unitary[0, 1]
        matrix[t,c] = self.unitary[1, 0]
        matrix[t,t] = self.unitary[1, 1]
        return matrix
        # TODO: implement.
        raise NotImplementedError("ControlledU.to_unitary is not implemented yet")


@dataclass
class CU:
    """Singly-controlled single-qubit gate C(U)."""

    n: int
    control: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where `unitary` acts on `target`."""

        I = np.eye(2, dtype=complex)

        P0 = np.array(
            [
                [1, 0],
                [0, 0],
            ],
            dtype=complex,
        )

        P1 = np.array(
            [
                [0, 0],
                [0, 1],
            ],
            dtype=complex,
        )

        branch_0 = np.array([[1]], dtype=complex)
        branch_1 = np.array([[1]], dtype=complex)

        for q in range(self.n):
            if q == self.control:
                branch_0 = np.kron(branch_0, P0)
                branch_1 = np.kron(branch_1, P1)

            elif q == self.target:
                branch_0 = np.kron(branch_0, I)
                branch_1 = np.kron(branch_1, self.unitary)

            else:
                branch_0 = np.kron(branch_0, I)
                branch_1 = np.kron(branch_1, I)

        return branch_0 + branch_1


@dataclass
class CNOT:
    """A controlled-NOT: flip `target` iff `control` is 1."""

    n: int
    control: int
    target: int

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where X swaps the target amplitudes."""

        X = np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=complex,
        )

        I = np.eye(2, dtype=complex)

        P0 = np.array(
            [
                [1, 0],
                [0, 0],
            ],
            dtype=complex,
        )

        P1 = np.array(
            [
                [0, 0],
                [0, 1],
            ],
            dtype=complex,
        )

        branch_0 = np.array([[1]], dtype=complex)
        branch_1 = np.array([[1]], dtype=complex)

        for q in range(self.n):
            if q == self.control:
                branch_0 = np.kron(branch_0, P0)
                branch_1 = np.kron(branch_1, P1)

            elif q == self.target:
                branch_0 = np.kron(branch_0, I)
                branch_1 = np.kron(branch_1, X)

            else:
                branch_0 = np.kron(branch_0, I)
                branch_1 = np.kron(branch_1, I)

        return branch_0 + branch_1


@dataclass
class Swap:
    """A multi-controlled NOT (generalized Toffoli): flip `target` iff every other
    qubit equals its entry in `control_vals`. `control_vals` has size n and is
    indexed by qubit; control_vals[target] is unused.
    """

    target: int
    control_vals: list[bool]


# A gate is any of the sparse representations above; a circuit is a list of gates.
Gate = Union[TwoLevel, SingleQubitGate, ControlledU, CU, CNOT]
Circuit = list  # list[Gate]
TwoLevels = list  # list[TwoLevel]


def circuit_to_unitary(circuit: Circuit) -> np.ndarray:
    """Full N x N unitary of a whole circuit. Gates are stored in order of
    application, so the product premultiplies (first gate is the rightmost factor):
    result = g_last @ ... @ g_1. Assumes the circuit is non-empty.
    """

    if len(circuit) == 0:
        raise ValueError("circuit_to_unitary expects a non-empty circuit")

    # ------------------------------------------------------------
    # Infer number of qubits / size
    # ------------------------------------------------------------
    n = None

    for gate in circuit:
        if hasattr(gate, "n"):
            n = gate.n
            break
        elif isinstance(gate, TwoLevel):
            n = num_qubits(gate.size)
            break
        elif isinstance(gate, Swap):
            n = len(gate.control_vals)
            break

    if n is None:
        raise ValueError("Could not infer number of qubits from circuit")

    size = 2 ** n
    result = np.eye(size, dtype=complex)

    # ------------------------------------------------------------
    # Build circuit unitary
    # ------------------------------------------------------------
    for gate in circuit:

        # ========================================================
        # Special case: Swap / generalized Toffoli
        # ========================================================
        if isinstance(gate, Swap):
            gate_matrix = np.zeros((size, size), dtype=complex)

            target = gate.target
            control_vals = gate.control_vals

            for i in range(size):
                controls_match = True

                for q in range(n):
                    if q == target:
                        continue

                    # q = 0 is the leftmost / most significant qubit
                    bit_val = bool((i >> (n - 1 - q)) & 1)

                    if bit_val != control_vals[q]:
                        controls_match = False
                        break

                if controls_match:
                    j = i ^ (1 << (n - 1 - target))
                else:
                    gate_matrix[j, i] = 1.0

        # ========================================================
        # Every other gate uses its own to_unitary()
        # ========================================================
        else:
            gate_matrix = gate.to_unitary()

        if gate_matrix.shape != (size, size):
            raise ValueError(
                f"Gate {gate} has shape {gate_matrix.shape}, expected {(size, size)}"
            )

        result = gate_matrix @ result

    return result


    # TODO: implement.
    raise NotImplementedError("circuit_to_unitary is not implemented yet")


def to_circuit(two_levels: TwoLevels) -> Circuit:
    """Wrap a two-level sequence as a circuit, so decompose_unitary /
    twolevel_decomposition output flows straight into a Circuit.
    """
    return list(two_levels)
    # TODO: implement.
    raise NotImplementedError("to_circuit is not implemented yet")


def error_up_to_phase(a: np.ndarray, b: np.ndarray) -> float:
    """Compare two same-size matrices ignoring global phase."""

    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if not np.all(np.isfinite(a)):
        raise ValueError("Matrix a contains nan or inf")

    if not np.all(np.isfinite(b)):
        raise ValueError("Matrix b contains nan or inf")

    overlap = np.vdot(b, a)

    if np.isclose(abs(overlap), 0.0):
        return float(np.max(np.abs(a - b)))

    phase = overlap / abs(overlap)
    b_aligned = phase * b

    return float(np.max(np.abs(a - b_aligned)))



# ---------------------------------------------------------------------------
# Stage 1: Unitary -> two-level unitaries (see cpp/src/TwoLevel.h)
# ---------------------------------------------------------------------------


def align(x: complex, y: complex, norm: float) -> np.ndarray:
    """The 2x2 unitary [[conj(x), conj(y)], [-y, x]] / norm. Premultiplying it onto
    a column with entries (x, y) at two levels rotates the amplitude at the second
    level onto the first, leaving the real `norm` there and 0 below.
    """

    if np.isclose(norm, 0.0):
        raise ValueError("norm must be nonzero")

    return np.array(
        [
            [np.conj(x), np.conj(y)],
            [-y, x],
        ],
        dtype=complex,
    ) / norm

    # TODO: implement.
    raise NotImplementedError("align is not implemented yet")


def decompose_vector(vec: np.ndarray) -> TwoLevels:
    """Given the first column of a unitary, return a sequence of two-levels which,
    when premultiplied onto the unitary, make its first column be (1, 0, 0, ...).
    Walk from the bottom up, using `align` at each pivot to zero out one entry; the
    running pivot holds the accumulated real norm after the first rotation.
    """

   
    vec = np.asarray(vec, dtype=complex).copy()
    N = len(vec)

    if N == 0:
        raise ValueError("Vector must be non-empty")

    two_levels = []

    for i in range(N - 1, 0, -1):
        x = vec[i - 1]
        y = vec[i]

        norm = np.sqrt(abs(x) ** 2 + abs(y) ** 2)

        if np.isclose(norm, 0.0):
            continue

        U = align(x, y, norm)

        two_levels.append(
            TwoLevel(
                size=N,
                level0=i - 1,
                level1=i,
                unitary=U,
            )
        )

        vec[i - 1] = norm
        vec[i] = 0.0

    return two_levels


    # TODO: implement.
    raise NotImplementedError("decompose_vector is not implemented yet")


def expand_twolevels(input: TwoLevels, n: int) -> TwoLevels:
    """Expand each TwoLevel to n dimensions by shifting size, level0, level1 up by
    the offset (n - tl.size). Used to lift a sub-block decomposition back to full n.
    """

    expanded = []

    for tl in input:
        offset = n - tl.size

        expanded.append(
            TwoLevel(
                size=n,
                level0=tl.level0 + offset,
                level1=tl.level1 + offset,
                unitary=tl.unitary,
            )
        )

    return expanded

    # TODO: implement.
    raise NotImplementedError("expand_twolevels is not implemented yet")


def two_levels_to_unitary(two_levels: TwoLevels) -> np.ndarray:
    """Full matrix of a two-level sequence: premultiply each two-level's matrix in
    order (result = tl.to_unitary() @ result), reproducing the application order.
    """
    size = two_levels[0].size
    result = np.eye(size, dtype=complex)
    for tl in two_levels:
       result = tl.to_unitary() @ result
    return result
    # TODO: implement.
    raise NotImplementedError("two_levels_to_unitary is not implemented yet")


def adjoint_twolevel(tl: TwoLevel) -> TwoLevel:
    """Adjoint of a single two-level: same levels, adjoint (conjugate transpose) of
    the 2x2 block.
    """

    return TwoLevel(
        size=tl.size,
        level0=tl.level0,
        level1=tl.level1,
        unitary=tl.unitary.conj().T,
    )

    # TODO: implement.
    raise NotImplementedError("adjoint_twolevel is not implemented yet")


def adjoint_twolevels(two_levels: TwoLevels) -> TwoLevels:
    """Adjoint of a sequence: reverse the order and take the adjoint of each, since
    (A_k ... A_1)^dagger = A_1^dagger ... A_k^dagger.
    """
    return [adjoint_twolevel(tl) for tl in reversed(two_levels)]
    # TODO: implement.
    raise NotImplementedError("adjoint_twolevels is not implemented yet")


def decompose_unitary(u: np.ndarray) -> TwoLevels:
    """Repeat decompose_vector on successive sub-columns to reduce u to identity.
    At step k, columns/rows 0..k-1 are already reduced, so work on the lower-right
    (n-k) block: clear column k below the diagonal. Finally append a phase two-level
    on the last two levels to cancel the residual phase, so the product is identity.
    Returns the sequence S with prod(S) @ u == I (i.e. prod(S) = u^dagger).
    """

    u = np.asarray(u, dtype=complex)
    N = u.shape[0]
    num_qubits(N)  

    if N < 2:
        return []

    two_levels = []
    working = u.copy()

    for k in range(N - 1):
        subvec = working[k:, k]
        sub_two_levels = decompose_vector(subvec)
        full_two_levels = expand_twolevels(sub_two_levels, N)
        for tl in full_two_levels:
            working = tl.to_unitary() @ working
        two_levels.extend(full_two_levels)
    residual_phase = working[N - 1, N - 1]
    if np.isclose(abs(residual_phase), 0.0):
        raise ValueError("Residual phase is zero; input may not be unitary")
    residual_phase = residual_phase / abs(residual_phase)
    phase_fix = TwoLevel(
        size=N,
        level0=N - 2,
        level1=N - 1,
        unitary=np.array(
            [
                [1.0, 0.0],
                [0.0, np.conj(residual_phase)],
            ],
            dtype=complex,
        ),
    )

    working = phase_fix.to_unitary() @ working
    two_levels.append(phase_fix)
    return two_levels
    # TODO: implement.
    raise NotImplementedError("decompose_unitary is not implemented yet")


def twolevel_decomposition(u: np.ndarray) -> TwoLevels:
    """The two-level decomposition of u itself: decompose_unitary returns the
    sequence S that reduces u to identity (prod(S) = u^dagger), so its adjoint is
    the sequence whose product is u.
    """
    return adjoint_twolevels(decompose_unitary(u))
    # TODO: implement (hint: adjoint_twolevels(decompose_unitary(u))).
    raise NotImplementedError("twolevel_decomposition is not implemented yet")


# ---------------------------------------------------------------------------
# ABC decomposition of a single-qubit gate (see cpp/src/ABC.h)
# ---------------------------------------------------------------------------


@dataclass
class ABC:
    """Nielsen & Chuang Corollary 4.2: every single-qubit U factors as
    U = e^{i alpha} A X B X C with A B C = I (X is Pauli-X). Building block for a
    single-controlled C(U).
    """

    alpha: float  # global phase
    A: np.ndarray  # (2, 2)
    B: np.ndarray  # (2, 2)
    C: np.ndarray  # (2, 2)


def abc_decompose(u: np.ndarray) -> ABC:
    """Build the ABC decomposition of u."""

    alpha, beta, gamma, delta = rotation.euler_angles_zyz(u)

    A = rotation.Rz(beta) @ rotation.Ry(gamma / 2)
    B = rotation.Ry(-gamma / 2) @ rotation.Rz(-(delta + beta) / 2)
    C = rotation.Rz((delta - beta) / 2)

    return ABC(
        alpha=alpha,
        A=A,
        B=B,
        C=C,
    )

def abc_reconstruct(d: ABC) -> np.ndarray:
    """Reassemble e^{i alpha} A X B X C from an ABC (inverse of abc_decompose)."""
    X = np.array([
    [0, 1],
    [1, 0]], dtype=complex)

    A=ABC.A
    B=ABC.B
    C=ABC.C
    alpha=ABC.alpha
    u=np.exp(1j*alpha)*A@X@B@X@C
    return u
    # TODO: implement.
    raise NotImplementedError("abc_reconstruct is not implemented yet")


# ---------------------------------------------------------------------------
# Gray code and controlled circuits (see cpp/src/Swap.h, cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def gray_code(tl: TwoLevel) -> list[Swap]:
    """Gray code connecting level0 and level1 of a two-level, as the sequence of
    single-qubit flips walking from level0 to level1 (one Swap per differing bit).
    At each step the Swap records which qubit flips and the current code's values on
    the other qubits (the control pattern).
    """
    n = num_qubits(tl.size)

    current = tl.level0
    target = tl.level1

    swaps = []

    for q in range(n):
        current_bit = (current >> (n - 1 - q)) & 1
        target_bit = (target >> (n - 1 - q)) & 1

        if current_bit != target_bit:
            control_vals = []

            for k in range(n):
                bit_k = (current >> (n - 1 - k)) & 1
                control_vals.append(bool(bit_k))

            swaps.append(
                Swap(
                    target=q,
                    control_vals=control_vals,
                )
            )

            current = current ^ (1 << (n - 1 - q))

    return swaps
    # TODO: implement.
    raise NotImplementedError("gray_code is not implemented yet")


def decompose_swap(swap: Swap) -> Circuit:
    """Decompose a Swap (multi-controlled NOT) into a Circuit: a controlled-X with
    the swap's arbitrary control values.
    """

    n = len(swap.control_vals)
    target = swap.target

    X = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )
    circuit = []
    for q in range(n):
        if q == target:
            continue

        if swap.control_vals[q] is False:
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=q,
                    unitary=X,
                )
            )

    circuit.append(
        ControlledU(
            n=n,
            target=target,
            unitary=X,
        )
    )

    for q in reversed(range(n)):
        if q == target:
            continue

        if swap.control_vals[q] is False:
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=q,
                    unitary=X,
                )
            )

    return circuit

    # TODO: implement (hint: controlled_circuit with Pauli-X).
    raise NotImplementedError("decompose_swap is not implemented yet")


def controlled_circuit(
    n: int, target: int, control_vals: list[bool], unitary: np.ndarray
) -> Circuit:
    """Circuit applying the 2x2 `unitary` to `target` iff every non-target qubit
    equals control_vals[q]. Realized as a fully-controlled-U core (ControlledU,
    controls = all 1) sandwiched by X gates on the qubits conditioned on 0, so those
    become 1-controls. The sandwich is symmetric (X is its own inverse).
    """

    X = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )

    circuit = []

    # Step 1: turn 0-controls into 1-controls using X
    for q in range(n):
        if q == target:
            continue

        if control_vals[q] is False:
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=q,
                    unitary=X,
                )
            )

    # Step 2: apply fully-controlled U
    circuit.append(
        ControlledU(
            n=n,
            target=target,
            unitary=unitary,
        )
    )

    # Step 3: undo the X gates
    for q in reversed(range(n)):
        if q == target:
            continue

        if control_vals[q] is False:
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=q,
                    unitary=X,
                )
            )

    return circuit

    # TODO: implement.
    raise NotImplementedError("controlled_circuit is not implemented yet")


# ---------------------------------------------------------------------------
# Stage 2-5: the decomposition pipeline (see cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def decompose_twolevel(tl: TwoLevel) -> Circuit:
    """Lower a TwoLevel to a Circuit (Nielsen-Chuang 4.5.2): walk a gray code so
    level0 becomes adjacent to level1, apply the controlled-U on that last
    transition, then undo the walk. Orient the 2x2 so a00 (level0's corner) sits on
    the target value the second-to-last code has.
    """

    circuit = []
    n = num_qubits(tl.size)
    swaps = gray_code(tl)

    for s in swaps[:-1]:
        circuit.extend(decompose_swap(s))


    last = swaps[-1]
    target = last.target
    control_vals = last.control_vals

    second_last_target_bit = control_vals[target]

    X = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )

    if second_last_target_bit is False:
        oriented_u = tl.unitary
    else:
        oriented_u = X @ tl.unitary @ X

    circuit.extend(
        controlled_circuit(
            n=n,
            target=target,
            control_vals=control_vals,
            unitary=oriented_u,
        )
    )
    for s in reversed(swaps[:-1]):
        circuit.extend(decompose_swap(s))

    return circuit

    # TODO: implement using gray_code, decompose_swap, controlled_circuit.
    raise NotImplementedError("decompose_twolevel is not implemented yet")


def decompose_controlled(
    n: int, controls: list[int], target: int, u: np.ndarray
    ) -> Circuit:
    """Decompose C^k(U) (k = len(controls)) into singly-controlled gates C(U) and
    CNOTs (Nielsen-Chuang fig 4.8). Base cases: no control -> a plain SingleQubitGate;
    one control -> a CNOT if U == X else a CU. Otherwise, with V = sqrt(U):
        a. C(V) on target
        b. C^{k-1}(X) onto the pivot control
        c. C(V dagger) on target
        d. repeat b
        e. C^{k-1}(V) on target
    Phases are kept throughout.
    """

    u = np.asarray(u, dtype=complex)

    X = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )

  
    if u.shape != (2, 2):
        raise ValueError("u must be a 2x2 matrix")

    if target < 0 or target >= n:
        raise ValueError("target qubit out of range")

    if target in controls:
        raise ValueError("target cannot also be a control")

    if len(set(controls)) != len(controls):
        raise ValueError("controls must not contain duplicates")

    for c in controls:
        if c < 0 or c >= n:
            raise ValueError("control qubit out of range")

 
    if len(controls) == 0:
        return [
            SingleQubitGate(
                n=n,
                qubit=target,
                unitary=u,
            )
        ]

   
    if len(controls) == 1:
        control = controls[0]

        if np.allclose(u, X):
            return [
                CNOT(
                    n=n,
                    control=control,
                    target=target,
                )
            ]

        return [
            CU(
                n=n,
                control=control,
                target=target,
                unitary=u,
            )
        ]

   
    pivot = controls[-1]
    rest = controls[:-1]

    V = rotation.unitary2_sqrt(u)
    V_dagger = V.conj().T

    circuit = []

    circuit.extend(
        decompose_controlled(
            n=n,
            controls=[pivot],
            target=target,
            u=V,
        )
    )

    circuit.extend(
        decompose_controlled(
            n=n,
            controls=rest,
            target=pivot,
            u=X,
        )
    )

    circuit.extend(
        decompose_controlled(
            n=n,
            controls=[pivot],
            target=target,
            u=V_dagger,
        )
    )

    circuit.extend(
        decompose_controlled(
            n=n,
            controls=rest,
            target=pivot,
            u=X,
        )
    )

    circuit.extend(
        decompose_controlled(
            n=n,
            controls=rest,
            target=target,
            u=V,
        )
    )

    return circuit

    # TODO: implement (recursive; use rotation.unitary2_sqrt for V).
    raise NotImplementedError("decompose_controlled is not implemented yet")


def decompose_controlledU(g: ControlledU) -> Circuit:
    """Lower a ControlledU (controlled on all other qubits) into CNOTs + C(U): build
    the list of all non-target qubits as controls and call decompose_controlled.
    """

    controls = []

    for q in range(g.n):
        if q != g.target:
            controls.append(q)

    return decompose_controlled(
        n=g.n,
        controls=controls,
        target=g.target,
        u=g.unitary,
    )

    # TODO: implement.
    raise NotImplementedError("decompose_controlledU is not implemented yet")


def decompose_cu(g: CU) -> Circuit:
    """Lower a singly-controlled C(U) into single-qubit gates + 2 CNOTs
    (Nielsen-Chuang Corollary 4.2 / fig 4.6). With U = e^{i alpha} A X B X C and
    A B C = I, emit: C, CNOT, B, CNOT, A on the target, plus a diag(1, e^{i alpha})
    phase on the control line. control=0: CNOTs vanish, target sees A B C = I;
    control=1: CNOTs act as X, target sees A X B X C = U with phase e^{i alpha}.
    """

    abc = abc_decompose(g.unitary)

    phase = np.array(
        [
            [1.0, 0.0],
            [0.0, np.exp(1j * abc.alpha)],
        ],
        dtype=complex,
    )

    circuit = []

    # C on target
    circuit.append(
        SingleQubitGate(
            n=g.n,
            qubit=g.target,
            unitary=abc.C,
        )
    )

    # CNOT control -> target
    circuit.append(
        CNOT(
            n=g.n,
            control=g.control,
            target=g.target,
        )
    )

    # B on target
    circuit.append(
        SingleQubitGate(
            n=g.n,
            qubit=g.target,
            unitary=abc.B,
        )
    )

    # CNOT control -> target
    circuit.append(
        CNOT(
            n=g.n,
            control=g.control,
            target=g.target,
        )
    )

    # A on target
    circuit.append(
        SingleQubitGate(
            n=g.n,
            qubit=g.target,
            unitary=abc.A,
        )
    )

    # Phase diag(1, e^{i alpha}) on control
    circuit.append(
        SingleQubitGate(
            n=g.n,
            qubit=g.control,
            unitary=phase,
        )
    )

    return circuit

    # TODO: implement using abc_decompose.
    raise NotImplementedError("decompose_cu is not implemented yet")


def decompose_to_basis(u: np.ndarray) -> Circuit:
    """Fully lower a Unitary to a Circuit of only SingleQubitGate and CNOT, running
    the four stages in sequence:
        1. twolevel_decomposition: Unitary     -> TwoLevels
        2. decompose_twolevel:     TwoLevel    -> SingleQubitGate + ControlledU
        3. decompose_controlledU:  ControlledU -> CU + CNOT
        4. decompose_cu:           CU          -> SingleQubitGate + CNOT
    Each stage rewrites only its own gate type and passes the rest through unchanged.
    """

    circuit = to_circuit(twolevel_decomposition(u))

   
    stage2 = []

    for gate in circuit:
        if isinstance(gate, TwoLevel):
            stage2.extend(decompose_twolevel(gate))
        else:
            stage2.append(gate)

    circuit = stage2

    
    stage3 = []

    for gate in circuit:
        if isinstance(gate, ControlledU):
            stage3.extend(decompose_controlledU(gate))
        else:
            stage3.append(gate)

    circuit = stage3

    stage4 = []

    for gate in circuit:
        if isinstance(gate, CU):
            stage4.extend(decompose_cu(gate))
        else:
            stage4.append(gate)

    circuit = stage4

    return circuit

    # TODO: implement (run each rewrite pass over the circuit).
    raise NotImplementedError("decompose_to_basis is not implemented yet")


def ht_gates(n: int, qubit: int, word: str) -> Circuit:
    """Expand a flat H/T word into a Circuit of SingleQubitGate H/T gates on `qubit`.
    The word (leftmost char = leftmost matrix factor) is pushed in reverse so the
    circuit's application order (first gate first = rightmost factor) reproduces
    rotation.gates_to_unitary(word).
    """

    H = (1 / np.sqrt(2)) * np.array(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=complex,
    )

    T = np.array(
        [
            [1, 0],
            [0, np.exp(1j * np.pi / 4)],
        ],
        dtype=complex,
    )

    circuit = []

    for ch in reversed(word):
        if ch == "H":
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=qubit,
                    unitary=H,
                )
            )

        elif ch == "T":
            circuit.append(
                SingleQubitGate(
                    n=n,
                    qubit=qubit,
                    unitary=T,
                )
            )

        else:
            raise ValueError(f"Unsupported HT gate character: {ch}")

    return circuit

    # TODO: implement.
    raise NotImplementedError("ht_gates is not implemented yet")



def decompose_to_ht(u: np.ndarray, error: float) -> Circuit:
    """Fully lower a Unitary to a Circuit of only H, T, and CNOT gates."""

    basis_circuit = decompose_to_basis(u)

    ht_circuit = []

    for gate in basis_circuit:
        if isinstance(gate, SingleQubitGate):
            word = rotation.approximate_in_ht(gate.unitary, error)

            ht_circuit.extend(
                ht_gates(
                    n=gate.n,
                    qubit=gate.qubit,
                    word=word,
                )
            )

        elif isinstance(gate, CNOT):
            ht_circuit.append(gate)

        else:
            raise ValueError(
                f"Unexpected gate after decompose_to_basis: {type(gate).__name__}"
            )

    return ht_circuit

    # TODO: implement using decompose_to_basis, ht_gates, and rotation.approximate_in_ht.
    raise NotImplementedError("decompose_to_ht is not implemented yet")
