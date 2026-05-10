#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
Naive Matrix Arithmetic Library

A lightweight, intuitive Python library for matrix arithmetic, supporting exact rational arithmetic, complex numbers, and various matrix decompositions.


MIT License

Copyright (c) 2026 kitanokitsune

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from fractions import Fraction
from decimal import Decimal
from ratcomplex import Complex
from gaussjordan import rank, dot, det, invert, moore_penrose, transpose, pprint_mat, spprint_mat

class NaiveMatrix:

    def __new__(cls, mat=None):
        if isinstance(mat, NaiveMatrix):
            return mat
        obj = super().__new__(cls)
        obj._shape = (None, None)
        obj._mat = None
        if mat is not None:
            ret = obj.set_mat(mat)
            if not ret:
                raise ValueError("Shape of the matrix is illegal")
        return obj

    # -------------------------------------------------
    def set_mat(self, mat):
        nrow, ncol = self._chk_shape(mat)
        if nrow is not None and nrow > 0:
            if ncol is None:
                self._mat = [ [e] for e in mat ]
                self._shape = (nrow, 1)
                return True
            elif ncol > 0:
                self._mat = mat
                self._shape = (nrow, ncol)
                return True
        return False

    # -------------------------------------------------
    def get_shape(self):
        return self._shape

    # -------------------------------------------------
    def inv(self):
        if self._mat is None:
            return None
        return NaiveMatrix(invert(self._mat))

    # -------------------------------------------------
    def pinv(self):
        if self._mat is None:
            return None
        return NaiveMatrix(moore_penrose(self._mat))

    # -------------------------------------------------
    def transpose(self):
        if self._mat is None:
            return None
        return NaiveMatrix(transpose(self._mat))

    # -------------------------------------------------
    def T(self):
        return self.transpose()

    # -------------------------------------------------
    def adjoint(self):
        if self._mat is None:
            return None
        mat = []
        for j in range(self._shape[1]):
            row = []
            for i in range(self._shape[0]):
                a = self._mat[i][j]
                if isinstance(a, (complex, Complex)):
                    row.append(Complex(a.real, -a.imag))
                else:
                    row.append(a)
            mat.append(row)
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def rank(self):
        if self._mat is None:
            return None
        return rank(self._mat)

    # -------------------------------------------------
    def det(self):
        if self._mat is None:
            return None
        return det(self._mat)

    # -------------------------------------------------
    def Frac(self, max_denominator=None):
        if self._mat is None:
            return None
        mat = []
        for r in self._mat:
            row = []
            for n in r:
                if isinstance(n, (complex, Complex)):
                    if max_denominator is None:
                        row.append(Complex(Fraction(n.real), Fraction(n.imag)))
                    else:
                        row.append(Complex(Fraction(n.real).limit_denominator(max_denominator), Fraction(n.imag).limit_denominator(max_denominator)))
                elif not isinstance(n, (Fraction, Complex)):
                    if max_denominator is None:
                        row.append(Fraction(n))
                    else:
                        row.append(Fraction(n).limit_denominator(max_denominator))
            mat.append(row)
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def Float(self):
        if self._mat is None:
            return None
        mat = []
        for r in self._mat:
            row = []
            for n in r:
                if isinstance(n, (complex, Complex)):
                    row.append(complex(n))
                elif not isinstance(n, float):
                    row.append(float(n))
            mat.append(row)
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def copy(self):
        if self._mat:
            mat = [ [n for n in r] for r in self._mat ]
            return NaiveMatrix(mat)
        else:
            return None

    # -------------------------------------------------
    def pprint(self, prefix=''):
        if self._mat:
            pprint_mat(prefix, self._mat)

    # -------------------------------------------------
    def spprint(self, prefix=''):
        if self._mat:
            return spprint_mat(prefix, self._mat)
        return None

    # -------------------------------------------------
    def __str__(self):
        return self.spprint()

    # -------------------------------------------------
    def __pos__(self):
        return self

    # -------------------------------------------------
    def __neg__(self):
        return (-1 * self)

    # -------------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, NaiveMatrix):
            raise TypeError("unsupported operand type(s) for ==: 'NaiveMatrix' and '{}'".format(type(other).__name__))
        if self._shape != other._shape:
            return False
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                if self._mat[i][j] != other._mat[i][j]:
                    return False
        return True

    # -------------------------------------------------
    def __ne__(self, other):
        return not self.__eq__(other)

    # -------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, NaiveMatrix):
            raise TypeError("unsupported operand type(s) for +: 'NaiveMatrix' and '{}'".format(type(other).__name__))
        shape1 = self._chk_shape(self._mat)
        shape2 = self._chk_shape(other._mat)
        if shape1 != shape2:
            raise ValueError("Unmatched matrix size. {}+{}".format(shape1, shape2))
        mat = []
        for i in range(shape1[0]):
            row = []
            for j in range(shape1[1]):
                row.append(self._mat[i][j] + other._mat[i][j])
            mat.append(row)
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def __sub__(self, other):
        if not isinstance(other, NaiveMatrix):
            raise TypeError("unsupported operand type(s) for -: 'NaiveMatrix' and '{}'".format(type(other).__name__))
        shape1 = self._chk_shape(self._mat)
        shape2 = self._chk_shape(other._mat)
        if shape1 != shape2:
            raise ValueError("Unmatched matrix size. {}-{}".format(shape1, shape2))
        mat = []
        for i in range(shape1[0]):
            row = []
            for j in range(shape1[1]):
                row.append(self._mat[i][j] - other._mat[i][j])
            mat.append(row)
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def __mul__(self, other):
        shape1 = self._chk_shape(self._mat)
        mat = []
        if isinstance(other, (int, float, complex, Complex, Fraction, Decimal)):
            mat = [ [ n*other for n in r ] for r in self._mat ]
        elif isinstance(other, NaiveMatrix):
            shape2 = self._chk_shape(other._mat)
            if shape1[1] != shape2[0]:
                raise ValueError("Unmatched matrix size. {}-{}".format(shape1, shape2))
            mat = dot(self._mat, other._mat)
        else:
            raise TypeError("unsupported operand type(s) for *: 'NaiveMatrix' and '{}'".format(type(other).__name__))
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def __rmul__(self, other):
        shape1 = self._chk_shape(self._mat)
        mat = []
        if isinstance(other, (int, float, complex, Complex, Fraction, Decimal)):
            mat = [ [ n*other for n in r ] for r in self._mat ]
        else:
            raise TypeError("unsupported operand type(s) for *: '{}' and 'NaiveMatrix'".format(type(other).__name__))
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def __matmul__(self, other):
        if not isinstance(other, NaiveMatrix):
            raise TypeError("unsupported operand type(s) for @: 'NaiveMatrix' and '{}'".format(type(other).__name__))
        shape1 = self._chk_shape(self._mat)
        shape2 = self._chk_shape(other._mat)
        if shape1[1] != shape2[0]:
            raise ValueError("Unmatched matrix size. {}-{}".format(shape1, shape2))
        mat = dot(self._mat, other._mat)
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def __truediv__(self, other):
        shape1 = self._chk_shape(self._mat)
        mat = []
        if isinstance(other, (int, float, complex, Complex, Fraction, Decimal)):
            mat = [ [ n/other for n in r ] for r in self._mat ]
        elif isinstance(other, NaiveMatrix):
            shape2 = self._chk_shape(other._mat)
            if shape1[1] != shape2[0] or shape1[1] != shape2[1]:
                raise ValueError("Unmatched matrix size. {}/{}".format(shape1, shape2))
            X = other.inv()
            mat = dot(self._mat, X._mat)
        else:
            raise TypeError("unsupported operand type(s) for /: 'NaiveMatrix' and '{}'".format(type(other).__name__))
        return NaiveMatrix(mat)

    # -------------------------------------------------
    def __rtruediv__(self, other):
        shape1 = self._chk_shape(self._mat)
        if shape1[0] != shape1[1]:
            raise ValueError("Unmatched matrix size. {}".format(shape1))
        if not isinstance(other, (int, float, complex, Complex, Fraction, Decimal)):
            raise TypeError("unsupported operand type(s) for /: '{}' and 'NaiveMatrix'".format(type(other).__name__))
        return (other * self.inv())

    # -------------------------------------------------
    def __pow__(self, p):
        shape1 = self._chk_shape(self._mat)
        if not isinstance(p, (int, float, complex, Complex, Fraction, Decimal)):
            raise TypeError("unsupported operand type(s) for **: 'NaiveMatrix' and '{}'".format(type(other).__name__))
        if isinstance(p, (complex, Complex)):
            if p.imag != 0:
                raise TypeError("unsupported operand type(s) for **: 'NaiveMatrix' and '{}'".format(type(other).__name__))
            else:
                p = p.real
        p = float(p)
        r = p % 1
        if r != 0.0:
            raise ValueError("Fractional power of 'NaiveMatrix' is unsupported")
        if shape1[0] != shape1[1]:
            raise ValueError("Unmatched matrix size. {}".format(shape1))
        p = int(p)
        if p < 0:
            mat = self.inv()
            p = abs(p)
        else:
            mat = self
        zero = self._mat[0][0] - self._mat[0][0]
        one = zero  + 1
        ans = NaiveMatrix([ [ one if i==j else zero for j in range(shape1[0]) ] for i in range(shape1[0]) ])
        d = [ int(c) for c in format(p, 'b') ]
        d.reverse()
        l = len(d)
        for n in d:
            l = l - 1
            if n == 1:
                ans = ans * mat
            if l > 0:
                mat = mat * mat
        return ans

    # -------------------------------------------------
    def __getitem__(self, key):
        if self._mat is None:
            return None
        return self._mat[key]

    # -------------------------------------------------
    def __setitem__(self, key, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("'NaiveMatrix' slice can not assign '{}' type".format(type(value).__name__))
        if len(value) != self._shape[1]:
            raise TypeError("'NaiveMatrix': the argument length is unmatched")
        self._mat[key] = value

    # -------------------------------------------------
    @staticmethod
    def _chk_shape(m):
        nrow = None
        ncol = None
        if isinstance(m, (list, tuple)):
            nrow = len(m)
            if nrow != 0:
                if isinstance(m[0], (list, tuple)):
                    ncol = len(m[0])
        if nrow is not None:
            if ncol is not None:
                for r in m:
                    if not isinstance(r, (list, tuple)):
                        ncol = -1
                        break
                    if len(r) != ncol:
                        ncol = -1
                        break
            else:
                for r in m:
                    if isinstance(r, (list, tuple)):
                        ncol = -1
                        break
        return (nrow, ncol)


def NaiveIdentityMatrix(n):
    if n < 1:
        return None
    mat = []
    for i in range(n):
        row = []
        for j in range(n):
            if i==j:
                row.append(1)
            else:
                row.append(0)
        mat.append(row)
    return NaiveMatrix(mat)


def NaiveZeroMatrix(m, n=None):
    if n is None:
        n = m
    if m < 1 or n < 1:
        return None
    mat = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(0)
        mat.append(row)
    return NaiveMatrix(mat)


def NaiveOneMatrix(m, n=None):
    if n is None:
        n = m
    if m < 1 or n < 1:
        return None
    mat = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(1)
        mat.append(row)
    return NaiveMatrix(mat)



if __name__ == '__main__':
    A = NaiveMatrix( [ [1,2], [1,2] ] ).Frac(max_denominator=10)
    b = NaiveMatrix( [ 0, 2 ] ).Frac(max_denominator=10)
    c = NaiveMatrix( [ [ 0.5j, 1 ], [ -1.2 + 3.1j, -2j ] ] ).Frac(10)
    d = c.copy()
    X = NaiveMatrix([[1, 2, 3, 1], [2, 4, 6, 4], [0, 0, 0, 2]]).Frac(1)
    A.pprint('A=')
    print('rank(A)=',A.rank())
    b.pprint('b=')
    print('rank(b)=',b.rank())
    (A.pinv()).pprint('A.pinv()=')
    ((A.pinv()) @ b).pprint('(A.pinv()) @ b=')
    print('A.det()=',A.det())
    (A @ ((A.pinv()) @ b)).pprint('A @ ((A.pinv()) @ b)=')
    #print(A / A)
    c.pprint('c=')
    print('c.det()=',c.det())
    (c.Float()).pprint('c.Float()=')
    d.pprint('d=')
    (1/d).pprint('1/d=')
    (pow(A, 8)).pprint('power(A, 2)=')
    (A**8).pprint('A**2=')
    (A/2).pprint('A/2=')
    ((d.adjoint()) @ d).pprint('(d.adjoint()) @ d=')
    (A+d).pprint('A+d=')
    (A-d).pprint('A-d=')
    (A@d).pprint('A@d=')
    (A/d).pprint('A/d=')
    X.pprint('X=')
    (X.T()).pprint('X.T()=')
    print((X.T())[1:3])
    (X.pinv()).pprint('X+ =')
    ((X.pinv()) @ X).pprint('X+ @ X =')
    (X @ (X.pinv()) @ X).pprint('X @ X+ @ X =')
    ((X.pinv()) @ NaiveMatrix([0,2,2])).pprint()
    (NaiveIdentityMatrix(5) / 2).Frac().pprint()
    NaiveZeroMatrix(2, 4).pprint()
    NaiveOneMatrix(4).pprint()
    print('c == d:', c == d)
    print('c != d:', c != d)
    (2*c).pprint('2*c=')
    (c*2).pprint('c*2=')

