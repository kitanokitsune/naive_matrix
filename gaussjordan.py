#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
Naive Gauss-Jordan Solver

A lightweight, pure-Python implementation of the Gauss-Jordan elimination method for linear algebra operations. This library is designed for educational purposes and for scenarios where exact rational arithmetic is required.


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

import os
from fractions import Fraction
from unicodedata import east_asian_width

_THIS_FILE = os.path.splitext(os.path.split(__file__)[1])[0]

# ------------------------------------------
def string_length(text):
    count = 0
    for c in text:
        if east_asian_width(c) in 'FWA':
            count += 2
        else:
            count += 1
    return count


# ------------------------------------------
def _getmatrixshape(M):
    if not isinstance(M, list):
        return (-1,)
    Nrow = len(M)
    if Nrow == 0:
        return (0,)
    if not isinstance(M[0], list):
        return (Nrow,)
    Ncol = len(M[0])
    if Ncol == 0:
        return (Nrow, 0)
    for row in M:
        try:
            l = len(row)
        except:
            return (Nrow, -1)
        if l != Ncol:
            return (Nrow, -1) # Shape of M is illegal
    return (Nrow, Ncol)


def _findpivot(A,n):
    Nrow = len(A)
    if n >= (Nrow-1):
        return (Nrow-1)
    p = n
    mag = 0
    for i in range(n+1, Nrow):
        c = abs(A[i][n])
        if mag < c:
            p = i
            mag = c
    return p



def _swaprows(A, n, p):
    vn = A[n]
    A[n] = A[p]
    A[p] = vn



def _divrowbyscalar(row, mag):
    N = len(row)
    for i in range(N):
        row[i] = row[i]/mag



def _subrows(row1, row2, mag):
    N = len(row1)
    for i in range(N):
        row1[i] -= (row2[i]*mag)


# ------------------------------------------
def _decomposition(Mat):

    shapeM = _getmatrixshape(Mat)
    if len(shapeM) != 2 or shapeM[1] < 1:
        raise TypeError

    Nrow = shapeM[0]
    Ncol = shapeM[1]

    # Make augmented matrix: A
    A = []
    zero = Mat[0][0] - Mat[0][0] # type cast to dtype(Mat)
    one = zero + 1
    for n in range(Nrow):
        row = [ c for c in Mat[n] ]
        for i in range(Nrow):
            row.append(one if (i==n) else zero)
        A.append(row)

    # Decomposition
    rank = 0
    c = 0
    n = 0
    while n < Nrow:
        if c == Ncol:
            break
        if A[n][c] == 0:
            p = c
            while p < Nrow:
                if A[p][c]:
                    break
                p = p + 1
            if p == Nrow:
                c = c + 1
                continue
            if n != p:
                _swaprows(A, n, p)
        rank = rank + 1
        for i in range(rank, Nrow):
            mag = A[i][c] / A[n][c]
            if mag:
                _subrows(A[i], A[n], mag)
        c = c + 1
        n = n + 1

    return rank, A


# ------------------------------------------
def _displayform(A, delimiter=', '):

    shape = _getmatrixshape(A)
    Nrow = shape[0]
    if shape[0] < 1:
        return [str(A)]
    if len(shape) == 1:
        X = [A]
        Ncol = Nrow
        Nrow = 1
    else:
        X = A
        Ncol = shape[1]

    M = [ [ '' for j in range(Ncol) ] for i in range(Nrow) ]
    W = [ 0 for j in range(Ncol) ]
    for i in range(Nrow):
        for j in range(Ncol):
            s = str(X[i][j])
            M[i][j] = s
            if len(s) > W[j]:
                W[j] = len(s)
    frm = []
    for i in range(Nrow):
        s = ''
        c = M[i][0]
        n = W[0]
        s = s + c.rjust(n)
        for j in range(1, Ncol):
            c = M[i][j]
            n = W[j]
            s = s + delimiter + c.rjust(n)
        frm.append(s)
    return frm


# ------------------------------------------
def solve(Mat, Vec):
    """Gauss-Jordan Solver
    """
    __FUNCNAME = _THIS_FILE + '.solve()'

    shapeM = _getmatrixshape(Mat)
    Nrow = shapeM[0]
    if len(shapeM) == 2:
        Ncol = shapeM[1]
    else:
        Ncol = 0
    if Nrow != Ncol or Nrow < 1 or Ncol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shapeM))

    shapeV = _getmatrixshape(Vec)
    if len(shapeV) == 2 and shapeV[0] == 1:
        Vec = Vec[0]
        shapeV = (shapeV[1],)
    elif len(shapeV) == 2 and shapeV[1] == 1:
        Vec = (transpose(Vec))[0]
        shapeV = (shapeV[0],)
    elif len(shapeV) != 1 or shapeV[0] < 1:
        raise ValueError("{}: Shape of vector is illegal: {}".format(__FUNCNAME, shapeV))

    if shapeV[0] != Ncol:
        raise ValueError("{}: Unmatched matrix and vector. {}x{}".format(__FUNCNAME, shapeM, shapeV))

    # Make augmented matrix: A
    A = []
    for n in range(Nrow):
        row = [ c for c in Mat[n] ]
        row.append(Vec[n])
        A.append(row)

    # Solve
    for n in range(Nrow):
        p = _findpivot(A, n)
        if n!=p:
            _swaprows(A, n, p)
        _divrowbyscalar(A[n], A[n][n])
        for i in range(Nrow):
            if i==n:
                continue
            mag = A[i][n]
            if mag:
                _subrows(A[i], A[n], mag)

    # Forming
    ans = [ A[n][Nrow] for n in range(Nrow) ]

    return ans


# ------------------------------------------
def invert(Mat):
    """Gauss-Jordan Eliminator
    """
    __FUNCNAME = _THIS_FILE + '.invert()'

    shapeM = _getmatrixshape(Mat)
    Nrow = shapeM[0]
    if len(shapeM) == 2:
        Ncol = shapeM[1]
    else:
        Ncol = 0
    if Nrow != Ncol or Nrow < 1 or Ncol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shapeM))

    # Make augmented matrix: A
    A = []
    zero = Mat[0][0] - Mat[0][0] # type cast to dtype(Mat)
    one = zero + 1
    for n in range(Nrow):
        row = [ c for c in Mat[n] ]
        for i in range(Ncol):
            row.append(one if (i==n) else zero)
        A.append(row)

    # Solve
    for n in range(Nrow):
        p = _findpivot(A, n)
        if n!=p:
            _swaprows(A, n, p)
        _divrowbyscalar(A[n], A[n][n])
        for i in range(Nrow):
            if i==n:
                continue
            mag = A[i][n]
            if mag:
                _subrows(A[i], A[n], mag)

    # Forming
    ans = []
    for n in range(Nrow):
        ans.append(A[n][Nrow:])

    return ans


# ------------------------------------------
def transpose(A):
    __FUNCNAME = _THIS_FILE + '.transpose()'

    shape = _getmatrixshape(A)
    Nrow = shape[0]
    if len(shape) == 1:
        X = [A]
        Ncol = Nrow
        Nrow = 1
    else:
        X = A
        Ncol = shape[1]
    if Nrow < 1 or Ncol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shape))

    Mat = [ [ 0 for i in range(Nrow) ] for j in range(Ncol) ]
    for i in range(Nrow):
        for j in range(Ncol):
            Mat[j][i] = X[i][j]
    return Mat


# ------------------------------------------
def dot(A, B):
    __FUNCNAME = _THIS_FILE + '.dot()'

    shapeA = _getmatrixshape(A)
    NArow = shapeA[0]
    if len(shapeA) == 2:
        NAcol = shapeA[1]
    else:
        NAcol = 0
    if NArow < 1 or NAcol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shapeA))

    shapeB = _getmatrixshape(B)
    NBrow = shapeB[0]
    if len(shapeB) == 2:
        NBcol = shapeB[1]
    else:
        NBcol = 0
    if NBrow < 1 or NBcol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shapeB))

    if NAcol != NBrow:
        raise ValueError("{}: Unmatched matrix size. {}x{}".format(__FUNCNAME, shapeA, shapeB))

    Mat = [ [0 for j in range(NBcol)] for i in range(NArow) ]
    for i in range(NArow):
        for j in range(NBcol):
            s = 0
            for k in range(NAcol):
                s = s + (A[i][k] * B[k][j])
            Mat[i][j] = s

    return Mat


# ------------------------------------------
def det(M):
    __FUNCNAME = _THIS_FILE + '.det()'

    shape = _getmatrixshape(M)
    Nrow = shape[0]
    if len(shape) == 2:
        Ncol = shape[1]
    else:
        Ncol = 0
    if Nrow != Ncol or Nrow < 1 or Ncol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shape))

    # deepcopy : A = deepcopy(M)
    A = [ [c for c in r] for r in M ]

    # Triangularize
    sign = 1
    for n in range(Nrow-1):
        p = _findpivot(A, n)
        if n!=p:
            _swaprows(A, n, p)
            sign = -sign
        mag = A[n][n]
        if mag == 0:
            return 0 # matrix is irregular
        for i in range(n+1,Nrow):
            _subrows(A[i], A[n], A[i][n]/mag)

    # Calculate determinant
    ans = sign
    for n in range(Nrow):
        ans = ans * A[n][n]

    return ans


# ------------------------------------------
def rank(Mat):
    """rank
    """
    __FUNCNAME = _THIS_FILE + '.rank()'

    shapeM = _getmatrixshape(Mat)
    Nrow = shapeM[0]
    if len(shapeM) == 2:
        Ncol = shapeM[1]
    else:
        Ncol = 0
    if Nrow < 1 or Ncol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shapeM))

    rank, _ = _decomposition(Mat)
    return rank


# ------------------------------------------
def moore_penrose(Mat):
    """Moore-Penrose
    Mat = B * C
    Mat+ = Ct * (Bt * Mat * Ct)^-1 * Bt
    """
    __FUNCNAME = _THIS_FILE + '.moore_penrose()'

    shapeM = _getmatrixshape(Mat)
    Nrow = shapeM[0]
    if len(shapeM) == 2:
        Ncol = shapeM[1]
    else:
        Ncol = 0
    if Nrow < 1 or Ncol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shapeM))

    rank, A = _decomposition(Mat)

    # Forming
    P = [ A[n][Ncol:] for n in range(Nrow) ]
    P = invert(P)

    B = [ P[n][:rank] for n in range(Nrow) ]

    C = [ A[n][:Ncol] for n in range(rank) ]

    Bt = transpose(B)
    Ct = transpose(C)
    D = invert(dot(dot(Bt, Mat), Ct))
    ans = dot(dot(Ct, D), Bt)

    return ans


# ------------------------------------------
def toFraction(M, limit_denom = None):
    shape = _getmatrixshape(M)
    if shape[0] < 1:
        try:
            if limit_denom is None:
                Mf = Fraction(M)
            elif isinstance(limit_denom, int):
                Mf = Fraction(m).limit_denominator(limit_denom)
            else:
                Mf = Fraction(m).limit_denominator()
        except:
            Mf = M
    else:
        Mf = []
        for row in M:
            Mf.append(toFraction(row, limit_denom))
    return Mf


# ------------------------------------------
def toReal(M):
    shape = _getmatrixshape(M)
    if shape[0] < 1:
        try:
            Mr = float(M)
        except:
            Mr = M
    else:
        Mr = []
        for row in M:
            Mr.append(toReal(row))
    return Mr


# ------------------------------------------
def spprint_mat(prefix='', mat=[]):
    __FUNCNAME = _THIS_FILE + '.spprint_mat()'

    shape = _getmatrixshape(mat)
    Nrow = shape[0]
    if shape[0] < 1:
        return str(mat)
    if len(shape) == 1:
        X = [mat]
        Ncol = Nrow
        Nrow = 1
    else:
        X = mat
        Ncol = shape[1]
    if Nrow < 1 or Ncol < 1:
        raise ValueError("{}: Shape of matrix is illegal: {}".format(__FUNCNAME, shape))

    # Define Matrix Representation Style
    #
    #  LeftUpper { 1, 0, 0 | RightUpper
    #       Left | 0, 1, 0 | Right
    #  LeftLower | 0, 0, 1 } RightLower
    #
    LeftUpper = '{ '
    RightUpper = ' |'

    Left = '| '
    Right = ' |'

    LeftLower = '| '
    RightLower = ' }'

    delimiter = ', '

    # Format Matrix as String
    MR = _displayform(X, delimiter=delimiter)
    indent = ''.rjust(string_length(prefix))
    disp = []
    for i in range(Nrow):
        s = MR[i]

        if i == 0:
            s = prefix + LeftUpper + s
        elif i == Nrow - 1:
            s = indent + LeftLower + s
        else:
            s = indent + Left + s

        if i == Nrow - 1:
            s = s + RightLower
        elif i == 0:
            s = s + RightUpper
        else:
            s = s + Right
        disp.append(s)

    return '\n'.join(disp)


# ------------------------------------------
def pprint_mat(prefix='', mat=[]):
    print(spprint_mat(prefix, mat))


