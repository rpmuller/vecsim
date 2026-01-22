# vecsim

A minimal quantum vector state simulator implemented in Python using NumPy.

## Overview

vecsim provides a lightweight framework for simulating quantum circuits using state vector representation. It implements quantum registers, common quantum gates, and utilities for constructing and manipulating quantum states.

## Features

- **State Vector Simulation**: Efficient representation of quantum states as complex NumPy arrays
- **Quantum Gates**: Support for common single-qubit gates (X, Y, Z, H) and two-qubit gates (CNOT)
- **Method Chaining**: Fluent interface for building quantum circuits
- **Flexible State Construction**: Create states using computational basis ('0', '1') or superposition basis ('+', '-')
- **In-place Operations**: Memory-efficient gate application

## Installation

```bash
uv sync
```

## Usage

```python
from vecsim import ket

# Create a Bell state
bell = ket('00').H(0).CNOT(0,1)
print(bell)  # 0.7071067811865475|00> 0.7071067811865475|11>

# Chain quantum gates
result = ket('0').X(0).H(0)
print(result)  # Superposition state

# Use superposition basis states
plus = ket('+')   # (|0⟩ + |1⟩)/√2
minus = ket('-')  # (|0⟩ - |1⟩)/√2
```

## Running Tests

```bash
make test
```

Or run doctests directly:

```bash
python vecsim.py
```

## Key Concepts

- Qubit indexing is zero-based; qubit 0 is the rightmost bit in binary representation
- All gate operations modify the quantum register in-place and return `self` for chaining
- States are automatically normalized
- Complex amplitudes with magnitude < 1e-8 are considered negligible

## Requirements

- Python 3.8+
- NumPy
- pytest (dev dependency)
