from typing import Sequence

import numpy as np
import jax.numpy as jnp



def _nway_outer(vectors: Sequence, xp):
    res = vectors[0]
    for v in vectors[1:]:
        res = xp.multiply.outer(res, v)
    return res


def nway_outer_np(vectors: Sequence):
    return _nway_outer(vectors, np)


def nway_outer_jax(vectors: Sequence):
    return _nway_outer(vectors, jnp)


def _reconstruct_Q_general(lambda_vec, Q_list, Qp_list, xp):
    D = len(Q_list)
    F = lambda_vec.shape[0]
    dims = [Q_list[d].shape[0] for d in range(D)]
    Q_hat = xp.zeros(dims + dims, dtype=lambda_vec.dtype)
    for f in range(F):
        left_vecs = [Q_list[d][:, f] for d in range(D)]  # s part
        right_vecs = [Qp_list[d][:, f] for d in range(D)]  # s' part
        left_tensor = _nway_outer(left_vecs, xp)  # (I_1,...,I_D)
        right_tensor = _nway_outer(right_vecs, xp)  # (I_1,...,I_D)
        rank1 = xp.multiply.outer(left_tensor, right_tensor)  # (I_1,...,I_D,I_1,...,I_D)
        Q_hat = Q_hat + lambda_vec[f] * rank1
    return Q_hat


def reconstruct_Q_general_np(lambda_vec, Q_list, Qp_list):
    return _reconstruct_Q_general(lambda_vec, Q_list, Qp_list, np)

def reconstruct_Q_general_jax(lambda_vec, Q_list, Qp_list):
    return _reconstruct_Q_general(lambda_vec, Q_list, Qp_list, jnp)

def pack_params_general_np(lambda_vec, Q_list, Qp_list):
    return _pack_params_general(lambda_vec, Q_list, Qp_list, np)

def pack_params_general_jax(lambda_vec, Q_list, Qp_list):
    return _pack_params_general(lambda_vec, Q_list, Qp_list, jnp)

def _pack_params_general(lambda_vec, Q_list, Qp_list, xp):
    """
    Pack x = [λ; vec(Q1); ...; vec(QD); vec(Q1'); ...; vec(QD')]
    using column-major vec(·) as in the paper.
    """
    parts = [lambda_vec.ravel()]
    for Q in Q_list:
        parts.append(jnp.reshape(Q, (-1,), order="F"))  # vec(Qd)
    for Qp in Qp_list:
        parts.append(jnp.reshape(Qp, (-1,), order="F"))  # vec(Qd')
    return xp.concatenate(parts)

def unpack_params_general_np(lambda_vec, Q_list, Qp_list):
    return _unpack_params_general(lambda_vec, Q_list, Qp_list, np)

def unpack_params_general_jax(lambda_vec, Q_list, Qp_list):
    return _unpack_params_general(lambda_vec, Q_list, Qp_list, jnp)

def _unpack_params_general(x, dims, F, xp):
    """
    Inverse of pack_params_general.

    dims = (I1,...,ID)
    Returns: lambda_vec, Q_list = [Q1,...,QD], Qp_list = [Q1',...,QD']
    """
    D = len(dims)
    idx = 0
    lam = x[idx : idx + F]
    idx += F

    Q_list = []
    for d in range(D):
        I_d = dims[d]
        size = I_d * F
        Qd = xp.reshape(x[idx : idx + size], (I_d, F), order="F")
        idx += size
        Q_list.append(Qd)

    Qp_list = []
    for d in range(D):
        I_d = dims[d]
        size = I_d * F
        Qdp = xp.reshape(x[idx : idx + size], (I_d, F), order="F")
        idx += size
        Qp_list.append(Qdp)

    assert idx == x.size
    return lam, Q_list, Qp_list

def _build_E_general(dims, F, xp):
    """
    Build E as in the appendix:
      E = blockdiag( 1_F^T, I_F⊗1_{I1}^T, ..., I_F⊗1_{ID}^T, I_F⊗1_{I1}^T, ..., I_F⊗1_{ID}^T )
    Shape: ((2 D F + 1) , F (2J+1)), J = sum(dims).
    """
    D = len(dims)
    blocks = []

    # block for ψ (copy of λ)
    blocks.append(xp.ones((1, F)))  # 1_F^T

    # blocks for s_d
    for d in range(D):
        I_d = dims[d]
        blocks.append(xp.kron(xp.eye(F), xp.ones((1, I_d))))

    # blocks for s'_d
    for d in range(D):
        I_d = dims[d]
        blocks.append(xp.kron(xp.eye(F), xp.ones((1, I_d))))

    rows = sum(b.shape[0] for b in blocks)
    cols = sum(b.shape[1] for b in blocks)
    E_empty = xp.zeros((rows, cols))

    return E_empty, blocks

def build_E_jax(dims, F):
    E, blocks = _build_E_general(dims, F, jnp)

    r = 0
    c = 0
    for b in blocks:
        rr, cc = b.shape
        E = E.at[r : r + rr, c : c + cc].set(b)
        r += rr
        c += cc
    return E

def build_E_np(dims, F):
    E, blocks = _build_E_general(dims, F, np)

    r = 0
    c = 0
    for b in blocks:
        rr, cc = b.shape
        E[r:r+rr, c:c+cc] = b
        r += rr
        c += cc
    return E

def _build_A_B_b(dims, F, xp):
    """
    A = [ I_{F(2J+1)} ; 0 ]
    B = [ -I_{F(2J+1)} ; E ]
    b = [ 0 ; 1 ]
    so that Ax + By = b encodes [x - y ; Ey - 1] = 0.
    """
    J = sum(dims)
    n = F * (2 * J + 1)

    if xp == np:
        E = build_E_np(dims, F)
    elif xp == jnp:
        E = build_E_jax(dims, F)
    m = E.shape[0]

    I_n = xp.eye(n)
    A = xp.vstack([I_n, xp.zeros((m, n))])
    B = xp.vstack([-I_n, E])
    b = xp.concatenate([xp.zeros(n), xp.ones(m)])
    return E, A, B, b

def build_A_B_b_np(dims, F):
    return _build_A_B_b(dims, F, np)

def build_A_B_b_jax(dims, F):
    return _build_A_B_b(dims, F, jnp)


__all__ = [
    "nway_outer_np",
    "nway_outer_jax",
    "reconstruct_Q_general_np",
    "reconstruct_Q_general_jax",
    "pack_params_general_np",
    "pack_params_general_jax",
    "unpack_params_general_np",
    "unpack_params_general_jax",
    "build_E_jax",
    "build_E_np",
    "build_A_B_b_np",
    "build_A_B_b_jax",
]
