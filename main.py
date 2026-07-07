import numpy as np

from decompose import (
    decompose_to_basis,
    decompose_to_ht,
    circuit_to_unitary,
    error_up_to_phase,
    SingleQubitGate,
    CNOT,
)


def random_unitary(N: int, seed: int = 0) -> np.ndarray:
    """Generate a random NxN unitary using QR decomposition."""

    rng = np.random.default_rng(seed)

    Z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))

    Q, R = np.linalg.qr(Z)

    # Fix phases so Q is distributed better as a random unitary
    diag = np.diag(R)
    phases = diag / np.abs(diag)

    Q = Q @ np.diag(np.conj(phases))

    return Q


def count_gates(circuit):
    """Count total gates and gate types."""

    counts = {}

    for gate in circuit:
        name = type(gate).__name__
        counts[name] = counts.get(name, 0) + 1

    return counts


def verify_basis(u: np.ndarray):
    """Verify decompose_to_basis end-to-end."""

    print("\n==============================")
    print("Testing decompose_to_basis")
    print("==============================")

    circuit = decompose_to_basis(u)
    rebuilt = circuit_to_unitary(circuit)

    err = error_up_to_phase(u, rebuilt)

    print("Number of gates:", len(circuit))
    print("Gate counts:", count_gates(circuit))
    print("Error up to global phase:", err)

    return err, circuit


def verify_ht(u: np.ndarray, error: float):
    """Verify decompose_to_ht end-to-end."""

    circuit = decompose_to_ht(u, error)
    rebuilt = circuit_to_unitary(circuit)

    err = error_up_to_phase(u, rebuilt)

    return err, circuit


def scaling_test_ht(u: np.ndarray):
    """Check scaling of number_of_gates vs target total error."""

    print("\n==============================")
    print("Scaling test: number_of_gates vs error")
    print("==============================")

    # First get basis circuit to know how many 1-qubit gates get approximated
    basis_circuit = decompose_to_basis(u)
    num_single_qubit = sum(isinstance(g, SingleQubitGate) for g in basis_circuit)

    print("Basis circuit gates:", len(basis_circuit))
    print("Single-qubit gates to approximate:", num_single_qubit)

    target_errors = [
        1e-1,
        5e-2,
        2e-2,
        1e-2,
    ]

    print(
        f"{'target_total_error':>20} | "
        f"{'per_gate_error':>15} | "
        f"{'actual_error':>15} | "
        f"{'num_gates':>10} | "
        f"{'CNOT':>6} | "
        f"{'1Q':>6}"
    )
    print("-" * 90)

    results = []

    for total_eps in target_errors:
        # Divide total error budget across all approximated 1Q gates
        per_gate_eps = total_eps / max(1, num_single_qubit)

        circuit = decompose_to_ht(u, per_gate_eps)
        rebuilt = circuit_to_unitary(circuit)

        actual_err = error_up_to_phase(u, rebuilt)

        counts = count_gates(circuit)
        num_gates = len(circuit)
        cnot_count = counts.get("CNOT", 0)
        oneq_count = counts.get("SingleQubitGate", 0)

        results.append(
            (total_eps, per_gate_eps, actual_err, num_gates, cnot_count, oneq_count)
        )

        print(
            f"{total_eps:20.1e} | "
            f"{per_gate_eps:15.3e} | "
            f"{actual_err:15.6e} | "
            f"{num_gates:10d} | "
            f"{cnot_count:6d} | "
            f"{oneq_count:6d}"
        )

    return results

def main():
    np.set_printoptions(precision=4, suppress=True)

    # 4x4 unitary = 2-qubit unitary
    N = 4

    u = random_unitary(N, seed=42)

    print("Random 4x4 unitary U:")
    print(u)

    # First verify exact basis decomposition:
    basis_err, basis_circuit = verify_basis(u)

    # Then test HT decomposition for different approximation errors:
    results = scaling_test_ht(u)

    print("\n==============================")
    print("Explanation")
    print("==============================")
    print(
       "The number of gates increases very fast since any arbitrary SingleQubitGate "
        "generated through decompose_to_basis needs to be approximated using a word "
        "over just H and T gates. Lower errors demand longer H/T words. In addition, "
        "since a 4x4 matrix can represent a general 2-qubit unitary operation, "
        "decomposition will have already generated several arbitrary single-qubit "
        "rotations that need to be replaced with H/T words."
    )


if __name__ == "__main__":
    main()