import numpy as np
import pytest

from ddplib import ddp


def dim(V):
    sv = np.linalg.svd(np.asarray(V, dtype=float), compute_uv=False)
    return int(np.sum(sv > 1e-9*sv[0])) if sv.size and sv[0] > 0 else 0


def same_subspace(V1, V2):
    return dim(V1) == dim(V2) == dim(np.append(V1, V2, axis=1))


def contained(V1, V2):
    '''im V1 is contained in im V2'''
    return dim(np.append(V1, V2, axis=1)) == dim(V2)


# Wonham, Linear Multivariable Control: A Geometric Approach, exercises 4.2 and 4.8
A_EX1 = np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0]])
B_EX1 = np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]])
H_EX1 = np.array([[1, 0, 0, -1, 0], [1, -1, 0, 0, 0], [0, 0, 0, 1, -1]])

A_EX2 = np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
B_EX2 = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [0, 0]])
H_EX2 = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])


@pytest.mark.parametrize('A, B, H, expected', [
    (A_EX1, B_EX1, H_EX1, np.ones((5, 1))),
    (A_EX2, B_EX2, H_EX2, np.array([[0], [0], [0], [0], [1]])),
])
def test_wonham_examples(A, B, H, expected):
    V = ddp.maximal_contrl_invariant_space(A, B, H)
    assert same_subspace(V, expected)

    F_ker, F_part = ddp.set_of_friends(V, A, B)
    assert F_ker
    for F in [F_part] + [F_part + Fk for Fk in F_ker] + [F_part + sum(F_ker)]:
        assert contained((A + B.dot(F)).dot(V), V)


def test_numpy_matrix_inputs_are_accepted():
    V = ddp.maximal_contrl_invariant_space(np.matrix(A_EX1), np.matrix(B_EX1), np.matrix(H_EX1))
    assert same_subspace(V, np.ones((5, 1)))


def test_ker_of_full_column_rank_is_zero_vector():
    K = ddp.ker(np.identity(3))
    assert K.shape == (3, 1)
    assert np.all(K == 0)


def test_subspace_sum_and_intersect():
    V1 = np.array([[1., 0], [0, 1], [0, 0]])
    V2 = np.array([[0., 0], [1, 0], [0, 1]])
    ssum, ssi = ddp.subspace_sum_intersect(V1, V2)
    assert same_subspace(ssum, np.identity(3))
    assert same_subspace(ssi, np.array([[0], [1], [0]]))


def test_subspace_intersect_trivial_is_zero_vector():
    ssi = ddp.subspace_intersect(np.array([[1.], [0]]), np.array([[0.], [1]]))
    assert ssi.shape == (2, 1)
    assert np.all(ssi == 0)


def test_subspace_sum_is_scale_invariant():
    ssum = ddp.subspace_sum(1e-14*np.array([[1.], [0]]), 1e-14*np.array([[0.], [1]]))
    assert dim(ssum) == 2


def test_basis_completion_of_full_rank_matrix_has_no_zero_column():
    T = ddp.basis_completion(np.identity(2))
    assert T.shape == (2, 2)
    assert dim(T) == 2


def test_basis_completion_of_rank_deficient_matrix():
    V = np.array([[1., 2], [0, 0], [0, 0]])
    T = ddp.basis_completion(V)
    assert T.shape == (3, 3)
    assert dim(T) == 3
    assert contained(T[:, :1], V)


def test_quotient():
    V2 = np.array([[1., 1], [0, 0], [0, 0]])  # dependent columns
    W = ddp.quotient(np.identity(3), V2)
    assert dim(W) == 2
    assert same_subspace(np.append(V2, W, axis=1), np.identity(3))


def test_matrix_eqn_solve():
    A = np.array([[1., 0], [0, 0]])
    X_ker, X_part = ddp.matrix_eqn_solve(A, np.array([[2.], [0]]))
    assert np.allclose(A.dot(X_part), [[2], [0]])
    assert same_subspace(X_ker, np.array([[0], [1]]))
    assert ddp.matrix_eqn_solve(A, np.array([[0.], [1]]))[1] == []


def test_matrix_equation():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((2, 1))
    X0 = rng.standard_normal((2, 3))
    Y0 = rng.standard_normal((1, 4))
    C = B.dot(Y0) + X0.dot(A)
    X_ker, X_part, Y_ker, Y_part = ddp.matrix_equation(A, B, C)
    assert np.allclose(B.dot(Y_part) + X_part.dot(A), C)
    for Xk, Yk in zip(X_ker, Y_ker):
        assert np.allclose(B.dot(Yk) + Xk.dot(A), 0)


def test_matrix_equation_without_solution():
    X_ker, X_part, Y_ker, Y_part = ddp.matrix_equation(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
    assert X_part == [] and Y_part == []


def test_affineintersect():
    # the lines {(t, 0)} and {(1, s)} meet at (1, 0)
    ker1, part1 = ddp.affineintersect(np.array([[1.], [0]]), np.zeros((2, 1)),
                                      np.array([[0.], [1]]), np.array([[1.], [0]]))
    assert np.allclose(part1, [[1], [0]])
    assert np.allclose(ker1, 0)
    # parallel lines do not meet
    assert ddp.affineintersect(np.array([[1.], [0]]), np.zeros((2, 1)),
                               np.array([[1.], [0]]), np.array([[0.], [1]])) == ([], [])


def test_set_of_friends_rejects_non_invariant_subspace():
    A = np.array([[0., 1], [0, 0]])
    B = np.array([[1.], [0]])
    with pytest.raises(ValueError):
        ddp.set_of_friends(np.array([[1.], [0]]), A.T, np.zeros((2, 1)))
    ddp.set_of_friends(np.array([[0.], [1]]), A, B)


def test_is_ddp_solvable():
    solvable, V = ddp.is_ddp_solvable(A_EX2, B_EX2, H_EX2, np.array([[0], [0], [0], [0], [1]]))
    assert solvable is True
    solvable, V = ddp.is_ddp_solvable(A_EX2, B_EX2, H_EX2, np.array([[1], [0], [0], [0], [0]]))
    assert solvable is False


@pytest.mark.parametrize('B', [B_EX1, B_EX1[:, :1]])
def test_kryl(B):
    K = ddp.kryl(A_EX1, B)
    assert K.shape == (5, 5*B.shape[1])
    assert dim(K) == dim(np.asarray(ddp.ctrl.ctrb(A_EX1, B)))
    norms = np.linalg.norm(K, axis=0)
    assert np.allclose(norms[norms > 0], 1)


def test_controllability_subspace_without_output_constraint_is_controllable_subspace():
    A = np.array([[0., 1, 0], [0, 0, 0], [0, 0, -1]])
    B = np.array([[0.], [1], [0]])
    R = ddp.controllability_subspace(A, B, np.zeros((1, 3)))
    assert same_subspace(R, np.asarray(ddp.ctrl.ctrb(A, B)))


# two decoupled double integrators; input 1 drives both chains
A_PLACE = np.array([[0., 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]])
B_PLACE = np.array([[0., 0], [1, 0], [0, 0], [1, 1]])
H_PLACE = np.array([[1., 0, 0, 0]])


def test_controllability_subspace():
    R = ddp.controllability_subspace(A_PLACE, B_PLACE, H_PLACE)
    assert same_subspace(R, np.identity(4)[:, 2:])


def test_ddp_place():
    R = ddp.controllability_subspace(A_PLACE, B_PLACE, H_PLACE)
    roots = [-1, -2, -3, -4]
    F = ddp.ddp_place(A_PLACE, B_PLACE, R, roots)
    Acl = A_PLACE + B_PLACE.dot(F)
    assert np.allclose(np.sort(np.linalg.eigvals(Acl).real), [-4, -3, -2, -1])
    assert contained(Acl.dot(R), R)
    # eigenvalues of A+BF restricted to R are the first dim(R) roots
    Acl_R = np.linalg.lstsq(R, Acl.dot(R), rcond=None)[0]
    assert np.allclose(np.sort(np.linalg.eigvals(Acl_R).real), [-2, -1])


def test_ddp_place_decouples_disturbance():
    E = np.array([[0.], [0], [1], [0]])
    solvable, V = ddp.is_ddp_solvable(A_PLACE, B_PLACE, H_PLACE, E)
    assert solvable
    R = ddp.controllability_subspace(A_PLACE, B_PLACE, H_PLACE)
    F = ddp.ddp_place(A_PLACE, B_PLACE, R, [-1, -2, -3, -4])
    Acl = A_PLACE + B_PLACE.dot(F)
    # transfer from d to y is zero: H (A+BF)^k E = 0 for all k
    assert np.allclose(H_PLACE.dot(np.asarray(ddp.ctrl.ctrb(Acl, E))), 0)


@pytest.mark.parametrize('seed', range(20))
def test_ddp_place_in_random_coordinates(seed):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((4, 4))
    Ti = np.linalg.inv(T)
    A = Ti.dot(A_PLACE).dot(T)
    B = Ti.dot(B_PLACE).dot(rng.standard_normal((2, 2)))
    H = H_PLACE.dot(T)
    if seed % 2:
        B = np.append(B, B[:, :1] + B[:, 1:], axis=1)  # dependent input columns
    R = ddp.controllability_subspace(A, B, H)
    assert same_subspace(R, Ti[:, 2:])
    F = ddp.ddp_place(A, B, R, [-1, -2, -3, -4])
    Acl = A + B.dot(F)
    assert np.allclose(np.sort(np.linalg.eigvals(Acl).real), [-4, -3, -2, -1], atol=1e-6)
    assert contained(Acl.dot(R), R)
