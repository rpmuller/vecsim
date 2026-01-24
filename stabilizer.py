# stabilizer.py - Efficient stabilizer state simulator using tableau representation
#
# Based on the Aaronson-Gottesman CHP algorithm:
# https://arxiv.org/abs/quant-ph/0406196
#
# This provides O(n^2) space and polynomial time operations for Clifford circuits,
# compared to O(2^n) for full state vector simulation.

from typing import List
import numpy as np


class StabilizerState:
    """
    Quantum state represented using the stabilizer tableau formalism.

    The tableau tracks 2n generators (n destabilizers + n stabilizers) for an
    n-qubit system. Each generator is a Pauli string with a phase.

    Tableau structure:
    - Rows 0 to n-1: Destabilizers
    - Rows n to 2n-1: Stabilizers
    - x[i,j]: X bit for generator i, qubit j
    - z[i,j]: Z bit for generator i, qubit j
    - r[i]: Phase in {0,1,2,3} representing {+1, +i, -1, -i}

    Examples:
        >>> s = StabilizerState(2)
        >>> print(s)
        +ZI
        +IZ
        >>> s.X(0)
        +ZI
        -IZ
        >>> s = StabilizerState(1).H(0)
        >>> print(s)
        +X
    """

    def __init__(self, n: int) -> None:
        """
        Initialize an n-qubit stabilizer state in |0...0⟩.

        Args:
            n: Number of qubits

        Raises:
            ValueError: If n < 1
        """
        if n < 1:
            raise ValueError(f"Number of qubits must be positive, got {n}")

        self.n = n
        # x[i,j] and z[i,j] are the X and Z bits for generator i, qubit j
        # Using uint8 for boolean storage
        self.x = np.zeros((2 * n, n), dtype=np.uint8)
        self.z = np.zeros((2 * n, n), dtype=np.uint8)
        # Phase: 0=+1, 1=+i, 2=-1, 3=-i
        self.r = np.zeros(2 * n, dtype=np.uint8)

        # Initialize to |0...0⟩ state:
        # Destabilizers: X_i for each qubit
        # Stabilizers: Z_i for each qubit
        for i in range(n):
            self.x[i, i] = 1          # Destabilizer i is X_i
            self.z[n + i, i] = 1      # Stabilizer i is Z_i

    def _rowmult(self, h: int, i: int) -> None:
        """
        Multiply generator h by generator i, storing result in h.

        This implements Pauli multiplication with proper phase tracking.
        The phase contribution when multiplying Paulis on qubit j:
        - I*I, X*X, Y*Y, Z*Z -> +1 (no phase change)
        - X*Y -> +i*Z, Y*X -> -i*Z
        - Y*Z -> +i*X, Z*Y -> -i*X
        - Z*X -> +i*Y, X*Z -> -i*Y

        Args:
            h: Row index to update (result stored here)
            i: Row index to multiply with
        """
        # Calculate phase contribution from Pauli multiplication
        # Using the formula from Aaronson-Gottesman
        phase = 0
        for j in range(self.n):
            xi, zi = self.x[i, j], self.z[i, j]
            xh, zh = self.x[h, j], self.z[h, j]

            # Determine Pauli type: 0=I, 1=X, 2=Z, 3=Y (X*Z with phase)
            pi = (xi << 1) | zi  # 0=I, 2=X, 1=Z, 3=Y
            ph = (xh << 1) | zh

            if pi == 0 or ph == 0:
                # Multiplying by identity: no phase
                pass
            elif pi == ph:
                # Same Pauli: squares to I, no phase
                pass
            else:
                # Different non-identity Paulis
                # XY=iZ, YZ=iX, ZX=iY (cyclic)
                # YX=-iZ, ZY=-iX, XZ=-iY (anti-cyclic)
                # Map: X=2, Y=3, Z=1 for cyclic check
                pauli_map = {1: 1, 2: 2, 3: 3}  # Z=1, X=2, Y=3
                if pi in pauli_map and ph in pauli_map:
                    # Check if (pi, ph) is cyclic: X->Y->Z->X (2->3->1->2)
                    cyclic = [(2, 3), (3, 1), (1, 2)]
                    if (pi, ph) in cyclic:
                        phase += 1  # +i
                    else:
                        phase += 3  # -i = +3i mod 4

        # Update phase
        self.r[h] = (self.r[h] + self.r[i] + phase) % 4

        # Update X and Z bits (XOR for Pauli multiplication)
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    def _pauli_char(self, x_bit: int, z_bit: int) -> str:
        """Return Pauli character for given X and Z bits."""
        if x_bit == 0 and z_bit == 0:
            return 'I'
        elif x_bit == 1 and z_bit == 0:
            return 'X'
        elif x_bit == 0 and z_bit == 1:
            return 'Z'
        else:  # x_bit == 1 and z_bit == 1
            return 'Y'

    def _phase_char(self, r: int) -> str:
        """Return phase character for phase value."""
        phases = ['+', '+i', '-', '-i']
        return phases[r % 4]

    def _generator_string(self, row: int) -> str:
        """Format a single generator row as a Pauli string."""
        phase = self._phase_char(self.r[row])
        paulis = ''.join(
            self._pauli_char(self.x[row, j], self.z[row, j])
            for j in range(self.n - 1, -1, -1)  # MSB first
        )
        return f"{phase}{paulis}"

    def generators(self) -> List[str]:
        """
        Return list of stabilizer generator strings.

        Returns:
            List of n strings representing the stabilizer generators.
            Ordered from highest qubit to lowest (matching string representation).

        Examples:
            >>> StabilizerState(2).generators()
            ['+ZI', '+IZ']
            >>> StabilizerState(1).H(0).generators()
            ['+X']
        """
        # Return in reverse order: highest qubit first (matches string convention)
        return [self._generator_string(self.n + i) for i in range(self.n - 1, -1, -1)]

    def __str__(self) -> str:
        """Return string representation showing stabilizer generators."""
        return '\n'.join(self.generators())

    def __repr__(self) -> str:
        """Return repr showing stabilizer generators."""
        return '\n'.join(self.generators())

    def H(self, target: int) -> 'StabilizerState':
        """
        Apply Hadamard gate to target qubit.

        Hadamard swaps X and Z: X -> Z, Z -> X, Y -> -Y

        Args:
            target: Zero-indexed qubit position

        Returns:
            Self for method chaining

        Raises:
            ValueError: If target is out of range

        Examples:
            >>> s = StabilizerState(1)
            >>> s.H(0).generators()
            ['+X']
            >>> s.H(0).generators()
            ['+Z']
        """
        if not (0 <= target < self.n):
            raise ValueError(f"Invalid target qubit {target}. Must be in [0, {self.n})")

        for i in range(2 * self.n):
            # Phase update: if both X and Z are set, we get Y -> -Y
            # r += 2 * x * z (mod 4), which flips sign when both are 1
            self.r[i] = (self.r[i] + 2 * self.x[i, target] * self.z[i, target]) % 4
            # Swap X and Z
            self.x[i, target], self.z[i, target] = self.z[i, target], self.x[i, target]

        return self

    def S(self, target: int) -> 'StabilizerState':
        """
        Apply Phase (S) gate to target qubit.

        S gate: X -> Y, Y -> -X, Z -> Z
        In tableau: Z[target] ^= X[target], with phase update

        Args:
            target: Zero-indexed qubit position

        Returns:
            Self for method chaining

        Raises:
            ValueError: If target is out of range

        Examples:
            >>> s = StabilizerState(1).H(0)  # Now in |+⟩, stabilized by X
            >>> s.S(0).generators()
            ['+Y']
        """
        if not (0 <= target < self.n):
            raise ValueError(f"Invalid target qubit {target}. Must be in [0, {self.n})")

        for i in range(2 * self.n):
            # Phase: when X=1, we pick up a phase from X -> Y = iXZ
            # r += 2 * x * z (accounts for existing Y becoming -X)
            self.r[i] = (self.r[i] + 2 * self.x[i, target] * self.z[i, target]) % 4
            # Z ^= X (X -> XZ = Y, Z -> Z, Y -> YZ = -X)
            self.z[i, target] ^= self.x[i, target]

        return self

    def CNOT(self, control: int, target: int) -> 'StabilizerState':
        """
        Apply CNOT gate with given control and target qubits.

        CNOT conjugation rules:
        - X_c -> X_c X_t
        - X_t -> X_t
        - Z_c -> Z_c
        - Z_t -> Z_c Z_t

        Args:
            control: Zero-indexed control qubit
            target: Zero-indexed target qubit

        Returns:
            Self for method chaining

        Raises:
            ValueError: If control or target is out of range, or if they're equal

        Examples:
            >>> s = StabilizerState(2).H(0).CNOT(0, 1)
            >>> sorted(s.generators())
            ['+XX', '+ZZ']
        """
        if not (0 <= control < self.n):
            raise ValueError(f"Invalid control qubit {control}. Must be in [0, {self.n})")
        if not (0 <= target < self.n):
            raise ValueError(f"Invalid target qubit {target}. Must be in [0, {self.n})")
        if control == target:
            raise ValueError("Control and target must be different qubits")

        for i in range(2 * self.n):
            # Phase update from CHP paper
            # r += x[control] * z[target] * (x[target] XOR z[control] XOR 1)
            xc, zc = self.x[i, control], self.z[i, control]
            xt, zt = self.x[i, target], self.z[i, target]
            self.r[i] = (self.r[i] + 2 * xc * zt * (xt ^ zc ^ 1)) % 4

            # X_target ^= X_control
            self.x[i, target] ^= self.x[i, control]
            # Z_control ^= Z_target
            self.z[i, control] ^= self.z[i, target]

        return self

    def Z(self, target: int) -> 'StabilizerState':
        """
        Apply Pauli-Z gate to target qubit.

        Z = S^2, so apply S twice.

        Args:
            target: Zero-indexed qubit position

        Returns:
            Self for method chaining

        Raises:
            ValueError: If target is out of range

        Examples:
            >>> s = StabilizerState(1).H(0)  # |+⟩ state
            >>> s.Z(0).generators()
            ['-X']
        """
        return self.S(target).S(target)

    def X(self, target: int) -> 'StabilizerState':
        """
        Apply Pauli-X gate to target qubit.

        X = H Z H, so use Hadamard conjugation.

        Args:
            target: Zero-indexed qubit position

        Returns:
            Self for method chaining

        Raises:
            ValueError: If target is out of range

        Examples:
            >>> s = StabilizerState(1).X(0)
            >>> s.generators()
            ['-Z']
        """
        return self.H(target).Z(target).H(target)

    def Y(self, target: int) -> 'StabilizerState':
        """
        Apply Pauli-Y gate to target qubit.

        Y = i X Z, so apply X then Z (with implicit phase from Pauli algebra).
        Since we're applying the gate (not tracking a Pauli), Y = S X S^dag = S X S^3
        Or more simply: Y = i X Z, so apply Z then X (order matters for phase).

        Actually: Y |psi> = i X Z |psi>
        For stabilizer update: conjugating by Y swaps X <-> -X, Z <-> -Z
        Y X Y^dag = -X, Y Z Y^dag = -Z, Y Y Y^dag = Y

        Args:
            target: Zero-indexed qubit position

        Returns:
            Self for method chaining

        Raises:
            ValueError: If target is out of range

        Examples:
            >>> s = StabilizerState(1).Y(0)
            >>> s.generators()
            ['-Z']
        """
        # Y = i X Z as operators, but for conjugation:
        # Y P Y^dag for Pauli P
        # This is equivalent to X(target).Z(target) with a global phase
        # But global phase doesn't matter for stabilizers
        # Actually for stabilizer states, applying Y flips both X and Z phases
        return self.X(target).Z(target)

    def M(self, i: int, ntimes: int = 1) -> list:
        """
        Measure qubit `i` `ntimes` times, collapsing the state each time.

        For stabilizer states, measurement uses the Aaronson-Gottesman algorithm:
        1. Check if any stabilizer anticommutes with Z_i (has X_i = 1)
        2. If yes: outcome is random, update tableau accordingly
        3. If no: outcome is deterministic, computed from destabilizers

        Args:
            i: Zero-indexed qubit position to measure
            ntimes: Number of measurements to perform

        Returns:
            List of measurement outcomes (0 or 1)

        Raises:
            ValueError: If qubit index is out of range

        Examples:
            >>> ket('0').M(0)
            [0]
            >>> ket('1').M(0)
            [1]
            >>> import numpy as np; np.random.seed(42)
            >>> ket('+').M(0)
            [0]
            >>> # After collapse, repeated measurements give same result
            >>> s = ket('+'); np.random.seed(42)
            >>> r1 = s.M(0)
            >>> s.M(0, 3)
            [0, 0, 0]
        """
        if not (0 <= i < self.n):
            raise ValueError(f"Invalid qubit {i}. Must be in [0, {self.n})")

        results = []
        for _ in range(ntimes):
            outcome = self._measure_single(i)
            results.append(outcome)

        return results

    def _measure_single(self, i: int) -> int:
        """
        Perform a single measurement of qubit i, updating the tableau.

        Returns:
            Measurement outcome (0 or 1)
        """
        import numpy as np

        # Find if any stabilizer anticommutes with Z_i
        # A Pauli string anticommutes with Z_i if it has X_i = 1
        p = None  # Index of first anticommuting stabilizer
        for row in range(self.n, 2 * self.n):
            if self.x[row, i] == 1:
                p = row
                break

        if p is not None:
            # Random outcome case: some stabilizer anticommutes with Z_i
            return self._measure_random(i, p)
        else:
            # Deterministic outcome case: all stabilizers commute with Z_i
            return self._measure_deterministic(i)

    def _measure_random(self, i: int, p: int) -> int:
        """
        Handle measurement when outcome is random.

        Args:
            i: Qubit being measured
            p: Index of first stabilizer that anticommutes with Z_i
        """
        import numpy as np

        # For all other rows that anticommute with Z_i, multiply by row p
        # This makes them commute with Z_i
        for row in range(2 * self.n):
            if row != p and self.x[row, i] == 1:
                self._rowmult(row, p)

        # Copy stabilizer p to its corresponding destabilizer slot
        dest_row = p - self.n
        self.x[dest_row] = self.x[p].copy()
        self.z[dest_row] = self.z[p].copy()
        self.r[dest_row] = self.r[p]

        # Set stabilizer p to ±Z_i based on random outcome
        outcome = np.random.randint(0, 2)

        # Clear row p and set it to Z_i
        self.x[p] = 0
        self.z[p] = 0
        self.z[p, i] = 1
        # Phase: 0 for +Z_i (outcome 0), 2 for -Z_i (outcome 1)
        self.r[p] = 2 * outcome

        return outcome

    def _measure_deterministic(self, i: int) -> int:
        """
        Handle measurement when outcome is deterministic.

        The outcome is determined by the existing stabilizer generators.
        We use the destabilizers to compute it.
        """
        # Find destabilizers that anticommute with Z_i (have X_i = 1)
        # Multiply them together to get the effective stabilizer
        phase = 0

        # We need to find which combination of stabilizers gives us Z_i
        # Look at destabilizers: if destabilizer j has X_i = 1, then
        # stabilizer j contributes to the measurement outcome
        for j in range(self.n):
            if self.x[j, i] == 1:  # Destabilizer j anticommutes with Z_i
                # The corresponding stabilizer j+n contributes
                # Multiply into our accumulated result
                # We only need to track the phase contribution
                phase = (phase + self.r[self.n + j]) % 4

        # The phase tells us the outcome:
        # phase 0 -> +1 eigenvalue -> outcome 0
        # phase 2 -> -1 eigenvalue -> outcome 1
        outcome = 1 if phase == 2 else 0

        return outcome


def ket(vecstring: str = "0") -> StabilizerState:
    """
    Create a stabilizer state from a string specification.

    Args:
        vecstring: String of quantum states using '0', '1', '+', or '-'
                  where '+' = (|0⟩+|1⟩)/sqrt(2) and '-' = (|0⟩-|1⟩)/sqrt(2)

    Returns:
        StabilizerState object representing the quantum state

    Raises:
        ValueError: If vecstring is empty or contains invalid characters

    Examples:
        >>> ket('0').generators()
        ['+Z']
        >>> ket('1').generators()
        ['-Z']
        >>> ket('+').generators()
        ['+X']
        >>> ket('-').generators()
        ['-X']
        >>> ket('00').generators()
        ['+ZI', '+IZ']
        >>> sorted(ket('++').generators())
        ['+IX', '+XI']
    """
    if not vecstring:
        raise ValueError("vecstring cannot be empty")

    valid_chars = {'0', '1', '+', '-'}
    invalid = set(vecstring) - valid_chars
    if invalid:
        raise ValueError(
            f"Invalid characters in vecstring: {invalid}. "
            f"Valid characters are: {valid_chars}"
        )

    n = len(vecstring)
    state = StabilizerState(n)

    # Process from right to left (qubit 0 is rightmost)
    for i, char in enumerate(reversed(vecstring)):
        if char == '1':
            # |1⟩ = X|0⟩
            state.X(i)
        elif char == '+':
            # |+⟩ = H|0⟩
            state.H(i)
        elif char == '-':
            # |-⟩ = H|1⟩ = HX|0⟩ = ZH|0⟩
            state.H(i).Z(i)
        # '0' needs no operation

    return state


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
