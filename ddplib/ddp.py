#Author: Deepak Patil
#Toolbox is still under development.
"""Geometric tools for the disturbance decoupling problem (DDP).

A subspace is represented by a matrix whose columns span it. By convention the
public functions represent the zero subspace {0} by a single zero column, so
every returned basis has at least one column. Inputs may be numpy arrays,
numpy matrices or nested lists; outputs are numpy arrays.
"""

import scipy.linalg as spl
import numpy as np
import control as ctrl

# Relative tolerance for all rank decisions: singular values below rtol times the
# largest singular value are treated as zero.
rtol = 1e-10


def _as_2d(M):
    '''Converts M to a 2-D float array; a 1-D input is treated as a column.'''
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
       M = M.reshape((-1, 1))
    return M

def _dim(V):
    '''Dimension of im V.'''
    V = _as_2d(V)
    if V.size == 0:
       return 0
    sv = np.linalg.svd(V, compute_uv=False)
    return int(np.sum(sv > rtol*sv[0])) if sv[0] > 0 else 0

def _null(M):
    return spl.null_space(M, rcond=rtol)

def _zero_if_empty(V):
    '''Replaces an empty (n x 0) basis by a single zero column.'''
    if V.shape[1] == 0:
       V = np.zeros((V.shape[0], 1))
    return V

def _basis(V):
    '''Orthonormal basis of im V; the zero subspace gives an (n x 0) matrix.'''
    return spl.orth(_as_2d(V), rcond=rtol)

def _sum(V1, V2):
    return _basis(np.append(_basis(V1), _basis(V2), axis=1))

def _intersect(V1, V2):
    Q1 = _basis(V1)
    Q2 = _basis(V2)
    W = _null(np.append(Q1, -Q2, axis=1))
    return _basis(Q1.dot(W[:Q1.shape[1], :]))

def _complement_within(V1, V2):
    '''Orthogonal complement of im V2 inside im V1 (assumes im V2 is in im V1).'''
    Q = _basis(V1)
    coords = Q.transpose().dot(_as_2d(V2))
    return Q.dot(_null(coords.transpose()))

def ker(S):
    """S is a matrix to be entered as an array. This function returns the nullspace of matrix S. A key difference added is that null_space function in scipy returns an empty matrix on passing a full column rank matrix as an input. However, a non-singular matrix null-space is not empty. Its a singleton set with zero vector in it. This function returns a zero vector upon encountering a matrix which is a full column rank matrix."""
    return _zero_if_empty(_null(_as_2d(S)))

def subspace_sum(V1,V2):
    '''Calculates an orthonormal basis of the sum of two subspaces im V1 and im V2.'''
    return _zero_if_empty(_sum(V1, V2))

def subspace_intersect(V1,V2):
    '''Calculates an orthonormal basis of the intersection of two subspaces im V1 and im V2.'''
    return _zero_if_empty(_intersect(V1, V2))

def subspace_sum_intersect(V1,V2):
    '''Calculates the sum and intersection of two subspaces im V1 and im V2.'''
    return subspace_sum(V1, V2), subspace_intersect(V1, V2)

def basis_completion(V):
    '''Input: matrix V whose columns span a subspace. Output: n x n matrix whose first columns are a maximal linearly independent subset of the columns of V (in their original order), completed to a basis of R^n by an orthonormal basis of the orthogonal complement.'''
    V = _as_2d(V)
    r = _dim(V)
    if r == 0:
       return np.identity(V.shape[0])
    e = spl.qr(V, pivoting=True)[2]
    v = V[:, np.sort(e[:r])]
    ortho_comple = _null(v.transpose())
    return np.append(v, ortho_comple, axis=1)

def matrix_eqn_solve(A,B):
    """AX=B all solutions X=X_ker*Z+X_part. Returns an empty list as the particular solution if the equation is not solvable."""
    A = _as_2d(A)
    B = _as_2d(B)
    X_ker = ker(A)
    if _dim(np.append(A, B, axis=1)) > _dim(A):
       return X_ker, []
    X_part = spl.lstsq(A, B)[0]
    return X_ker, X_part

def quotient(V1,V2):
    '''Returns a basis of a subspace W such that im V1 = im V2 (direct sum) W. Requires im V2 to be contained in im V1. W is chosen orthogonal to im V2.'''
    return _zero_if_empty(_complement_within(V1, V2))

def affineintersect(S_ker1, S_part1, S_ker2, S_part2):
    ''' This function takes two affine spaces and computes the intersection affine space. Affine space is specified by matrices A, B are of appropriate size such that AX+B generates points in the affine space by arbitrary varying X. Returns ([], []) if the affine spaces do not intersect.'''
    S_ker1 = _as_2d(S_ker1)
    S_part1 = _as_2d(S_part1)
    S_ker2 = _as_2d(S_ker2)
    S_part2 = _as_2d(S_part2)
    A = np.append(S_ker1, -S_ker2, axis=1)
    B = S_part2 - S_part1
    Sol_ker, Sol_part = matrix_eqn_solve(A, B)
    if len(Sol_part) == 0:
       return [], []
    n = S_ker1.shape[1]
    X_ker = Sol_ker[:n, :]
    X_part = Sol_part[:n, :]
    Fcom_part = S_ker1.dot(X_part) + S_part1
    Fcom_ker = S_ker1.dot(X_ker)
    return Fcom_ker, Fcom_part

def maximal_contrl_invariant_space(A,B,H):
    '''Computes the maximal controlled invariant subspace for xdot = Ax + Bu, contained in ker H. Recursion: V_0 = ker H, V_{k+1} = ker H intersect A^{-1}(V_k + im B).'''
    A = _as_2d(A)
    B = _as_2d(B)
    H = _as_2d(H)
    V = ker(H)
    dim_V_temp = -1
    while dim_V_temp != _dim(V):
        dim_V_temp = _dim(V)
        # Z spans the orthogonal complement of V + im B, so Ax lies in V + im B iff Z'Ax = 0
        Z = ker(np.append(V.transpose(), B.transpose(), axis=0))
        H2 = Z.transpose().dot(A)
        V = ker(np.append(H, H2, axis=0))

    return V

def matrix_equation(A,B,C,typ=None):
    '''Computes set of all solutions to BY + XA = C. The output is a list X_ker, Y_ker and particular solution X_part, Y_part. A general solution is then of the form 'linear combination of matrices in X_ker (resp. Y_ker) + X_part (resp. Y_part)'. Note that same linear combination should be used for both X_ker and Y_ker in generating one solution X and Y. X_part and Y_part are empty lists if there is no solution. The argument typ is unused and kept for backward compatibility.'''
    A = _as_2d(A)
    B = _as_2d(B)
    C = _as_2d(C)
    n, m = A.shape
    mb = B.shape[1]
    n1 = C.shape[0]
    #vec() stacks columns: vec(XA) = (A' kron I) vec X, vec(BY) = (I kron B) vec Y
    A1 = np.kron(A.transpose(), np.identity(n1))
    B1 = np.kron(np.identity(m), B)
    AA = np.append(A1, B1, axis=1)
    BB = C.transpose().reshape([n1*m, 1])
    size_of_vecX = n*n1
    x_ker, x_part = matrix_eqn_solve(AA, BB)
    vecX_ker = x_ker[:size_of_vecX, :]
    vecY_ker = x_ker[size_of_vecX:, :]
    #devectorization of kernel
    X_ker = []
    Y_ker = []
    for i in range(x_ker.shape[1]):
        if np.any(x_ker[:, i] != 0):
           X_ker.append(vecX_ker[:, i].reshape([n, n1]).transpose())
           Y_ker.append(vecY_ker[:, i].reshape([m, mb]).transpose())
    if len(x_part) == 0: #no solution case
       return X_ker, [], Y_ker, []
    #devectorization of particular solution
    X_part = x_part[:size_of_vecX, :].reshape([n, n1]).transpose()
    Y_part = x_part[size_of_vecX:, :].reshape([m, mb]).transpose()

    return X_ker, X_part, Y_ker, Y_part

def set_of_friends(V,A,B):
    '''Computes the set of friends F of a given (A,B)-invariant subspace im V, i.e. all F with (A+BF) im V contained in im V. It is returned in a parametric form F_part + linear combination of the matrices in the list F_ker. Raises ValueError if im V is not (A,B)-invariant.'''
    V = _as_2d(V)
    A = _as_2d(A)
    B = _as_2d(B)
    RHS = A.dot(V)
    LHS = np.append(V, B, axis=1)
    X_ker, X_part = matrix_eqn_solve(LHS, RHS)
    if len(X_part) == 0:
       raise ValueError('im V is not (A,B)-invariant: A im V is not contained in im V + im B')
    m = V.shape[1]
    # AV = [V B][X;U] with U = U_part + U_ker Y. F is a friend iff FV = -U, i.e. FV + U_ker Y = -U_part.
    U_part = X_part[m:, :]
    U_ker = X_ker[m:, :]
    F_ker, F_part, Y_ker, Y_part = matrix_equation(V, U_ker, U_part)
    F_ker = [F for F in F_ker if not np.allclose(F, 0)]
    return F_ker, -F_part

def is_ddp_solvable(A,B,H,E):
    '''Given xdot = Ax + Bu + Ed, y = Hx, disturbance decoupling problem is solvable if and only if image of E is in V, the maximal controlled invariant subspace in ker H. Returns (solvable, V) where solvable is True if this condition is met and False otherwise.'''
    V = maximal_contrl_invariant_space(A, B, H)
    VE = np.append(V, _as_2d(E), axis=1)
    return bool(_dim(VE) == _dim(V)), V

def kryl(A,B):
    '''Krylov (controllability) matrix [B, AB, ..., A^(n-1)B] where every column is normalised to unit 2-norm before being multiplied by A. Zero columns are left as zero.'''
    A = _as_2d(A)
    B = _as_2d(B)
    def normalise(M):
        norms = np.linalg.norm(M, 2, axis=0)
        norms[norms == 0] = 1
        return M/norms
    blocks = [normalise(B)]
    for i in range(A.shape[0]-1):
        blocks.append(normalise(A.dot(blocks[-1])))
    return np.concatenate(blocks, axis=1)

def controllability_subspace(A,B,H):
    '''Computes the supremal controllability subspace R* contained in ker H for xdot = Ax + Bu. Recursion: S_0 = 0, S_{k+1} = V* intersect (A S_k + im B), where V* is the maximal controlled invariant subspace in ker H.'''
    A = _as_2d(A)
    B = _as_2d(B)
    V = maximal_contrl_invariant_space(A, B, H)
    S = np.zeros((A.shape[0], 0))
    while True:
        S1 = _intersect(V, _sum(A.dot(S), B))
        if _dim(S1) == _dim(S):
           return _zero_if_empty(S1)
        S = S1

def ddp_place(A,B,S,roots):
    '''Input: A,B matrices, S controllability subspace, roots of desired characteristic polynomial (the first dim(S) roots are assigned to A+BF restricted to im S, the rest to the induced map on the quotient R^n/im S; complex roots must come in conjugate pairs within each group). Output: F such that eigenvalues of A+BF are roots and (A+BF) im S is contained in im S.'''
    A = _as_2d(A)
    B = _as_2d(B)
    n = A.shape[0]
    m = B.shape[1]
    roots = np.asarray(roots).ravel()
    if len(roots) != n:
       raise ValueError('expected %d roots, got %d' % (n, len(roots)))

    Rs = _basis(S)
    r = Rs.shape[1]
    p = roots[:r]
    q = roots[r:]

    RsB = _intersect(Rs, B)                 # Rs \int B
    Z = _complement_within(B, RsB)          # (Rs \int B) \ds Z = B
    W1 = _complement_within(np.identity(n), np.append(Rs, Z, axis=1)) # (Rs \ds Z) \ds W1 = R^n
    T = np.concatenate([Rs, Z, W1], axis=1) # R^n = Rs \ds Z \ds W1
    k = RsB.shape[1]
    z = Z.shape[1]

    # Input coordinates G with B G = [RsB Z 0], so that the transformed B is block diagonal.
    G = np.append(spl.lstsq(B, np.append(RsB, Z, axis=1))[0], _null(B), axis=1)

    Anew = np.linalg.solve(T, A.dot(T))
    Bnew = np.linalg.solve(T, B.dot(G))
    A11 = Anew[:r, :r]
    A21 = Anew[r:, :r]
    A22 = Anew[r:, r:]
    B1 = Bnew[:r, :k]
    B2 = Bnew[r:, k:k+z]

    # A+BF in the new coordinates is [[A11 + B1 F11, *], [A21 + B2 F21, A22 + B2 F22]].
    Fnew = np.zeros((m, n))
    if r > 0:
       Fnew[:k, :r] = -np.asarray(ctrl.place(A11, B1, p))
    if r > 0 and r < n:
       F21 = spl.lstsq(B2, -A21)[0]
       if not np.allclose(A21 + B2.dot(F21), 0, atol=1e-8*max(1, np.linalg.norm(A, 2))):
          raise ValueError('im S is not (A,B)-invariant')
       Fnew[k:k+z, :r] = F21
    if r < n:
       if z == 0:
          raise ValueError('B is contained in im S, so the eigenvalues outside im S cannot be assigned')
       Fnew[k:k+z, r:] = -np.asarray(ctrl.place(A22, B2, q))

    return G.dot(Fnew).dot(np.linalg.inv(T))
