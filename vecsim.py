# vecsim - quantum vector state simulator in python

from typing import Union, List
import numpy as np
import doctest
import math

# utilities

def nqubits(vl: int) -> int:
    """
    Return the number of qubits in a quantum register.

    Args:
        vl: Length of the state vector (must be power of 2)

    Returns:
        Number of qubits (log2 of vector length)

    Examples:
        >>> nqubits(2)
        1
        >>> nqubits(4)
        2
        >>> nqubits(16)
        4
    """
    if vl <= 0:
        raise ValueError(f"Vector length must be positive, got {vl}")
    return math.floor(math.log2(vl))

def conjugate_index(i: int, b: int) -> int:
    """
    Flip bit b in index i using XOR operation.

    Args:
        i: The state index
        b: The bit position to flip (0-indexed from right)

    Returns:
        New index with bit b flipped

    Raises:
        ValueError: If bit position is negative

    Examples:
        >>> conjugate_index(0,0) # |0> -> |1>
        1
        >>> conjugate_index(1,0) # |1> -> |0>
        0
        >>> conjugate_index(2,1) # |10> -> |00>
        0
    """
    if b < 0:
        raise ValueError(f"Bit position must be non-negative, got {b}")
    return i ^ (1 << b)

def small(a: complex, eps: float = 1e-8) -> bool:
    """
    Check if amplitude is negligibly small.

    Args:
        a: Complex amplitude to check
        eps: Threshold for considering value small

    Returns:
        True if absolute value is below threshold
    """
    return abs(a) < eps

# quantum register

class QReg:
    """\
    >>> ket('0').X(0)
    '1.0|1>'
    >>> ket('1').X(0)
    '1.0|0>'
    >>> ket('00').X(0)
    '1.0|01>'
    >>> ket('01').X(1)
    '1.0|11>'

    Can chain these together
    >>> ket('00').X(0).X(1)
    '1.0|11>'
    >>> ket('+').H(0).X(0)
    '1.0|1>'
    >>> ket('01').CNOT(0,1)
    '1.0|11>'
    >>> ket('00').CNOT(0,1)
    '1.0|00>'
    """
    def __init__(self, register: Union[List[complex], np.ndarray]) -> None:
        """
        Initialize quantum register with state vector.

        Args:
            register: State vector as list or numpy array

        Raises:
            ValueError: If register is empty or length is not power of 2
            TypeError: If register is not list or ndarray
        """
        # Validate and convert input
        if isinstance(register, list):
            if not register:
                raise ValueError("Register cannot be empty")
            self.v = np.asarray(register, dtype=complex)
        elif isinstance(register, np.ndarray):
            if register.size == 0:
                raise ValueError("Register cannot be empty")
            self.v = register.astype(complex)
        else:
            raise TypeError(
                f"Register must be list or ndarray, got {type(register).__name__}"
            )

        # Validate length is power of 2
        n = len(self.v)
        if n == 0 or (n & (n - 1)) != 0:
            raise ValueError(f"Register length must be power of 2, got {n}")

        self.n = nqubits(len(self.v))
        self.normalize()
        
    def __str__(self) -> str:
        """Return string representation of quantum state."""
        return str(self.terms())

    def __repr__(self) -> str:
        """Return representation of quantum state."""
        return repr(self.terms())

    def __add__(self, other: 'QReg') -> 'QReg':
        """Add two quantum states (superposition with normalization)."""
        return QReg((self.v + other.v) / math.sqrt(2))

    def __sub__(self, other: 'QReg') -> 'QReg':
        """Subtract two quantum states (superposition with normalization)."""
        return QReg((self.v - other.v) / math.sqrt(2))

    def __mul__(self, other: 'QReg') -> 'QReg':
        """Tensor product of two quantum states."""
        return QReg(np.kron(self.v, other.v))

    def norm(self) -> float:
        """
        Calculate norm of the quantum state vector.

        Returns:
            L2 norm of the state vector
        """
        return np.linalg.norm(self.v)

    def normalize(self) -> None:
        """
        Normalize the quantum state vector in-place.

        Raises:
            ValueError: If state vector has zero norm
        """
        norm = self.norm()
        if norm < 1e-10:
            raise ValueError("Cannot normalize zero vector")
        self.v /= norm
    def terms(self, eps: float = 1e-8) -> str:
        """
        Return string representation of significant terms in quantum state.

        Args:
            eps: Threshold for including terms in output

        Returns:
            Space-separated string of quantum state terms
        """
        return " ".join([
            qterm(i, qi, self.n)
            for (i, qi) in enumerate(self.v)
            if abs(qi) > eps
        ])

    def apply1q(self, M: np.ndarray, target: int) -> 'QReg':
        """
        Apply single-qubit gate to target qubit in-place.

        Args:
            M: 2x2 unitary matrix representing the quantum gate
            target: Zero-indexed qubit position to apply gate to

        Returns:
            Self, for method chaining

        Raises:
            ValueError: If target qubit index is out of bounds
        """
        if not (0 <= target < self.n):
            raise ValueError(
                f"Invalid target qubit {target}. Must be in [0, {self.n})"
            )

        temp = np.empty(2, dtype=complex)
        for (i, qi) in enumerate(self.v):
            j = conjugate_index(i, target)
            if i > j:
                continue
            if small(abs(qi) + abs(self.v[j])):
                continue
            temp[0], temp[1] = qi, self.v[j]
            self.v[i], self.v[j] = M @ temp
        return self

    def apply2q(self, M: np.ndarray, control: int, target: int) -> 'QReg':
        """
        Apply two-qubit entangling gate in-place.

        Args:
            M: 4x4 unitary matrix representing the two-qubit gate
            control: Zero-indexed control qubit position
            target: Zero-indexed target qubit position

        Returns:
            Self, for method chaining

        Raises:
            ValueError: If control or target qubit indices are out of bounds
                       or if control equals target
        """
        if not (0 <= control < self.n):
            raise ValueError(
                f"Invalid control qubit {control}. Must be in [0, {self.n})"
            )
        if not (0 <= target < self.n):
            raise ValueError(
                f"Invalid target qubit {target}. Must be in [0, {self.n})"
            )
        if control == target:
            raise ValueError("Control and target must be different qubits")

        temp = np.empty(4, dtype=complex)
        for (i, qi) in enumerate(self.v):
            j = conjugate_index(i, target)
            if i > j:
                continue
            k = conjugate_index(i, control)
            if i > k:
                continue
            l = conjugate_index(j, control)

            total_amplitude = abs(qi) + abs(self.v[j]) + abs(self.v[k]) + abs(self.v[l])
            if small(total_amplitude):
                continue

            temp[0], temp[1], temp[2], temp[3] = qi, self.v[j], self.v[k], self.v[l]
            self.v[i], self.v[j], self.v[k], self.v[l] = M @ temp
        return self

    def isclose(self, other: Union['QReg', List[complex], np.ndarray]) -> bool:
        """
        Check if quantum state is close to another state.

        Args:
            other: QReg, list, or numpy array to compare against

        Returns:
            True if states are element-wise close within tolerance

        Raises:
            TypeError: If other is not QReg, list, or ndarray
        """
        if isinstance(other, np.ndarray):
            register = other
        elif isinstance(other, list):
            register = np.asarray(other, dtype=complex)
        elif isinstance(other, QReg):
            register = other.v
        else:
            raise TypeError(
                f"Cannot compare QReg with {type(other).__name__}"
            )
        return bool(np.isclose(self.v, register).all())

    def X(self, target: int) -> 'QReg':
        """Apply Pauli-X (NOT) gate to target qubit."""
        return self.apply1q(X, target)

    def Y(self, target: int) -> 'QReg':
        """Apply Pauli-Y gate to target qubit."""
        return self.apply1q(Y, target)

    def Z(self, target: int) -> 'QReg':
        """Apply Pauli-Z gate to target qubit."""
        return self.apply1q(Z, target)

    def H(self, target: int) -> 'QReg':
        """Apply Hadamard gate to target qubit."""
        return self.apply1q(H, target)

    def CNOT(self, control: int, target: int) -> 'QReg':
        """Apply controlled-NOT gate with control and target qubits."""
        return self.apply2q(CNOT, control, target)

# quantum states
q0 = QReg([1,0])
q1 = QReg([0,1])
qp = QReg([1,1])
qm = QReg([1,-1])
# and two qubit states
q00 = q0*q0
q01 = q0*q1
q10 = q1*q0
q11 = q1*q1

# quantum operators
# Pauli matrices
I = np.asarray([[1, 0], [0, 1]],dtype=complex)
X = np.asarray([[0, 1], [1, 0]],dtype=complex)
Y = np.asarray([[0, -1j], [1j, 0]],dtype=complex)
Z = np.asarray([[1, 0], [ 0, -1]],dtype=complex)

# Other matrices
H = np.asarray([[1, 1], [ 1, -1]],dtype=complex)/math.sqrt(2)
S = np.asarray([[1, 0], [ 0, 1j]],dtype=complex)
CPHASE = np.asarray([[1, 0, 0, 0], [ 0, 1, 0, 0], [ 0, 0, 1, 0], [ 0, 0, 0, -1]],dtype=complex)
CNOT = np.asarray([[1, 0, 0, 0], [ 0, 1, 0, 0], [ 0, 0, 0, 1], [ 0, 0, 1, 0]],dtype=complex)

# convenience functions and utilities
def ket(vecstring: str = "0") -> QReg:
    """
    Create a quantum ket state from string specification.

    Args:
        vecstring: String of quantum states using '0', '1', '+', or '-'
                  where '+' = (|0⟩+|1⟩)/√2 and '-' = (|0⟩-|1⟩)/√2

    Returns:
        QReg object representing the quantum state

    Raises:
        ValueError: If vecstring is empty or contains invalid characters

    Examples:
        >>> ket('0')
        '1.0|0>'
        >>> ket('1')
        '1.0|1>'
        >>> ket('00')
        '1.0|00>'
        >>> ket('10')
        '1.0|10>'
        >>> ket('11')
        '1.0|11>'
        >>> ket('101')
        '1.0|101>'
        >>> ket('10').terms()
        '1.0|10>'
        >>> ket('++').isclose([0.5,0.5,0.5,0.5])
        True
        >>> ket('--').isclose([0.5,-0.5,-0.5,0.5])
        True
        >>> ket('+').isclose(ket('0') + ket('1'))
        True
        >>> ket('-').isclose(ket('0') - ket('1'))
        True
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

    qvec = {
        "0": QReg([1, 0]),
        "1": QReg([0, 1]),
        "+": QReg([1, 1]),
        "-": QReg([1, -1])
    }
    register = np.array([1], dtype=complex)
    for s in vecstring[::-1]:
        register = np.kron(qvec[s].v, register)
    return QReg(register)

# Formatting ket terms:
def qterm(i: int, qi: complex, n: int) -> str:
    """
    Format a single term of quantum state.

    Args:
        i: State index
        qi: Complex amplitude
        n: Number of qubits (for formatting width)

    Returns:
        Formatted string like "amplitude|binary_state>"
    """
    return f"{qcoef(qi)}|{i:0{n}b}>"


def qcoef(a: complex) -> Union[float, complex]:
    """
    Format complex coefficient for display.

    Args:
        a: Complex amplitude

    Returns:
        Real part if purely real, otherwise full complex number
    """
    return a.real if np.isreal(a) else a

if __name__ == '__main__': doctest.testmod(verbose=True)