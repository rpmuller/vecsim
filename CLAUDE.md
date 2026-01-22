# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vecsim is a quantum vector state simulator implemented in Python using NumPy. It provides a minimal implementation of quantum computing primitives for simulating quantum registers and gates.

## Core Architecture

The simulator uses state vector representation where quantum states are stored as complex-valued NumPy arrays:

- **QReg class** (vecsim.py:37-115): The central abstraction representing a quantum register
  - Stores the quantum state as a normalized complex vector in `self.v`
  - Uses in-place gate application for efficiency
  - Supports method chaining for gate operations

- **Gate Application Strategy**: Two distinct methods for applying quantum gates
  - `apply1q()` (vecsim.py:75-84): Single-qubit gates (X, Y, Z, H)
  - `apply2q()` (vecsim.py:86-100): Two-qubit entangling gates (CNOT, CPHASE)
  - Both use `conjugate_index()` to flip specific qubit bits via XOR operations
  - Small amplitudes are skipped for optimization (using `small()` helper)

- **State Construction**: The `ket()` function (vecsim.py:142-177) constructs states from strings
  - Supports computational basis: '0', '1', '00', '01', etc.
  - Supports superposition basis: '+' (|0⟩+|1⟩), '-' (|0⟩-|1⟩)
  - Uses Kronecker product to build multi-qubit states from right to left

## Development Commands

### Running Tests
```bash
make test
```
The module uses doctest for testing. Doctests are embedded in function docstrings throughout vecsim.py. The test command runs pytest with `--doctest-modules` flag.

Alternatively, run doctests directly:
```bash
python vecsim.py
```

### Running JupyterLab
```bash
jupyter lab
```
Project includes JupyterLab as a dependency for interactive exploration.

### Running Main Script
```bash
python main.py
```

## Dependencies

- **numpy**: Core numerical operations and linear algebra
- **jupyterlab**: For interactive development and experimentation
- Uses Python 3.8+ (specified in pyproject.toml)
- Managed via uv package manager (uv.lock present)

## Key Implementation Details

- Qubit indexing is zero-based; qubit 0 is the rightmost bit in binary representation
- All gate operations modify the quantum register in-place and return `self` for chaining
- Quantum states are automatically normalized upon creation and after operations
- Complex amplitudes with magnitude < 1e-8 are considered negligible and displayed as zero
