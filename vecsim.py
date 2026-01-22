# vecsim - quantum vector state simulator in python

import numpy as np
import doctest
import math

# utilities

def nqubits(vl): 
    """\
    Return the number of qubits in a quantum register.
    >>> nqubits(2)
    1
    >>> nqubits(4)
    2
    >>> nqubits(16)
    4
    """    
    return math.floor(math.log2(vl))

def conjugate_index(i,b):
    """\
    Could return this as a quantum bit.
    >>> conjugate_index(0,0) # |0> -> |1>
    1
    >>> conjugate_index(1,0) # |1> -> |0>
    0
    >>> conjugate_index(2,1) # |10> -> |00>
    0
    """
    return i ^ (1 << b)

def small(a,eps=1e-8): return abs(a) < eps

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
    def __init__(self,register):
        self.n = nqubits(len(register))
        self.v = np.asarray(register,dtype=complex)
        self.normalize()
        return
        
    def __str__(self): return str(self.terms())
    def __repr__(self): return repr(self.terms())

    def __add__(self,other): return QReg((self.v+other.v)/math.sqrt(2))
    def __sub__(self,other): return QReg((self.v-other.v)/math.sqrt(2))
    def __mul__(self,other): return QReg(np.kron(self.v,other.v))

    def norm(self): return np.linalg.norm(self.v)
    def normalize(self): self.v /= self.norm()
    def terms(self,eps=1e-8): return " ".join([qterm(i,qi,self.n) for (i,qi) in enumerate(self.v) if abs(qi) > eps])

    def apply1q(self,M,target):
        "in-place application of a quantum gate"
        temp = np.asarray([0,0],dtype=complex)
        for (i,qi) in enumerate(self.v):
            j = conjugate_index(i,target)
            if i > j: continue
            if small(abs(qi) + abs(self.v[j])): continue
            temp[0],temp[1] = qi,self.v[j]
            self.v[i],self.v[j] = M@temp
        return self

    def apply2q(self,M,control,target):
        "in-place application of a two-qubit entangling quantum gate"
        temp = np.asarray([0,0,0,0],dtype=complex)
        for (i,qi) in enumerate(self.v):
            j = conjugate_index(i,target)
            if i>j: continue
            k = conjugate_index(i,control)
            if i>k: continue
            l = conjugate_index(j,control)

            if small(abs(qi) + abs(self.v[j]) + abs(self.v[k]) + abs(self.v[l])): continue

            temp[0],temp[1],temp[2],temp[3] = qi,self.v[j],self.v[k],self.v[l]
            self.v[i],self.v[j],self.v[k],self.v[l] = M@temp
        return self

    def isclose(self,other):
        if type(other) == np.ndarray:
            register = other
        elif type(other) == list:
            register = np.asarray(other,dtype=complex)
        else: # assume QReg
            register = other.v
        return bool(np.isclose(self.v,register).all())

    def X(self,target): return self.apply1q(X,target)
    def Y(self,target): return self.apply1q(Y,target)
    def Z(self,target): return self.apply1q(Z,target)
    def H(self,target): return self.apply1q(H,target)
    def CNOT(self,control,target): return self.apply2q(CNOT,control,target)

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
def ket(vecstring="0"):
    """\
    Create a quantum ket state.
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
    qvec = {
        "0" : QReg([1,0]),
        "1" : QReg([0,1]),
        "+" : QReg([1,1]),
        "-" : QReg([1,-1])
    }
    register = np.array([1],dtype=complex)
    for s in vecstring[::-1]:
        register = np.kron(qvec[s].v,register)
    return QReg(register)

# Formatting ket terms:
def qterm(i,qi,n): return f"{qcoef(qi)}|{i:0{n}b}>"
def qcoef(a): return a.real if np.isreal(a) else a

if __name__ == '__main__': doctest.testmod()