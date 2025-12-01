import numpy as np

from .common import nway_outer_np as nway_outer
from .common import reconstruct_Q_general_np as reconstruct_Q_general
from .common import pack_params_general_np as pack_params_general
from .common import unpack_params_general_np as unpack_params_general
from .common import build_E_np as build_E
from .common import build_A_B_b_np as build_A_B_b

# ---------- Vectorization consistent with the appendix ----------

def grad_phi_general(x, Q_tilde, dims, F, eps=1e-12):
    """
    Gradient of φ(x) = 0.5 * ||Q_hat(x) - Q_tilde||_F^2
    for general D, w.r.t. x = [λ; vec(Qd); vec(Qd')].

    Q_tilde: tensor (I1,...,ID,I1,...,ID)
    dims: (I1,...,ID)
    """
    D = len(dims)
    lam, Q_list, Qp_list = unpack_params_general(x, dims, F)
    Q_hat = reconstruct_Q_general(lam, Q_list, Qp_list)
    E_res = Q_hat - Q_tilde  # residual

    grad_lam = np.zeros_like(lam)
    grad_Q_list  = [np.zeros_like(Q_list[d])  for d in range(D)]
    grad_Qp_list = [np.zeros_like(Qp_list[d]) for d in range(D)]

    s_axes     = list(range(D))
    sprime_axes = list(range(D, 2*D))

    for f in range(F):
        left_vecs  = [Q_list[d][:,  f] for d in range(D)]
        right_vecs = [Qp_list[d][:, f] for d in range(D)]
        left_tensor  = nway_outer(left_vecs)     # shape dims
        right_tensor = nway_outer(right_vecs)    # shape dims

        # dφ/dλ_f = <E_res, L_f ⊗ R_f>
        rank1 = np.multiply.outer(left_tensor, right_tensor)
        grad_lam[f] = np.tensordot(E_res, rank1, axes=E_res.ndim)

        # Contractions used for Q and Q'
        temp  = np.tensordot(E_res, right_tensor, axes=(sprime_axes, s_axes))  # (I1,...,ID)
        temp2 = np.tensordot(E_res, left_tensor,  axes=(s_axes,     s_axes))   # (I1,...,ID)

        for d in range(D):
            I_d = dims[d]

            # --- gradient w.r.t Q_d(:,f) ---

            Qd_col = Q_list[d][:, f]          # (I_d,)
            shape = [1]*D; shape[d] = I_d
            Qd_bc = np.broadcast_to(Qd_col.reshape(shape), dims)

            # product over all dims except d: L_f / Q_d
            B = np.zeros_like(left_tensor)
            mask = Qd_bc > eps
            B[mask] = left_tensor[mask] / Qd_bc[mask]

            w = temp * B  # (I1,...,ID)

            axes_sum = tuple(ax for ax in range(D) if ax != d)
            grad_Q_d_col = lam[f] * w.sum(axis=axes_sum)
            grad_Q_list[d][:, f] += grad_Q_d_col

            # --- gradient w.r.t Q'_d(:,f) ---

            Qdp_col = Qp_list[d][:, f]
            Qdp_bc = np.broadcast_to(Qdp_col.reshape(shape), dims)
            Bp = np.zeros_like(right_tensor)
            maskp = Qdp_bc > eps
            Bp[maskp] = right_tensor[maskp] / Qdp_bc[maskp]

            w2 = temp2 * Bp
            grad_Qp_d_col = lam[f] * w2.sum(axis=axes_sum)
            grad_Qp_list[d][:, f] += grad_Qp_d_col

    return pack_params_general(grad_lam, grad_Q_list, grad_Qp_list)


def admm_lrt_general(Q_tilde, F, dims,
                      max_iters=200,
                      beta=1.0,
                      inner_steps=20,
                      step_size=0.05,
                      verbose=False,
                      seed=0):
    """
    General-D ADMM demo following the vectorized formulation (10)-(11).

    Q_tilde: empirical joint tensor, shape (I1,...,ID, I1,...,ID)
    F      : CP rank
    dims   : tuple/list (I1,...,ID)
    """
    np.random.seed(seed)
    D = len(dims)
    J = sum(dims)
    n = F * (2 * J + 1)    # length of x and y

    # --- Initialize λ, Q_d, Q'_d on simplices (nonnegative, col sums = 1) ---

    lam = np.random.rand(F)
    lam /= lam.sum()

    def rand_factor(I, F_):
        M = np.random.rand(I, F_)
        M /= M.sum(axis=0, keepdims=True)
        return M

    Q_list  = [rand_factor(dims[d], F) for d in range(D)]
    Qp_list = [rand_factor(dims[d], F) for d in range(D)]

    x = pack_params_general(lam, Q_list, Qp_list)
    assert x.size == n

    # --- Build E and matrices for y-step ---

    E = build_E(dims, F)
    m = E.shape[0]                 # = 2 D F + 1
    M_mat = np.eye(n) + E.T @ E    # (I + E^T E)

    # y and duals (scaled)
    y = x.copy()
    u1 = np.zeros_like(x)
    u2 = np.zeros(m)
    one_m = np.ones(m)

    def phi(xvec):
        lam, Q_list, Qp_list = unpack_params_general(xvec, dims, F)
        Q_hat = reconstruct_Q_general(lam, Q_list, Qp_list)
        return 0.5 * np.sum((Q_hat - Q_tilde)**2)

    history = []

    for k in range(max_iters):
        # ----- x-step: projected gradient descent on φ(x) + (β/2)||x - y + u1||^2 -----
        c = y - u1
        xk = x.copy()
        for t in range(inner_steps):
            g = grad_phi_general(xk, Q_tilde, dims, F)
            g_total = g + beta * (xk - c)
            xk = xk - step_size * g_total
            xk = np.maximum(xk, 0.0)  # enforce x >= 0 (indicator term)
        x = xk

        # ----- y-step: solve (I + E^T E) y = x + u1 + E^T (1 - u2) -----
        rhs = x + u1 + E.T @ (one_m - u2)
        y = np.linalg.solve(M_mat, rhs)

        # ----- dual updates (scaled) -----
        r1 = x - y
        r2 = E @ y - one_m
        u1 = u1 + r1
        u2 = u2 + r2

        r_norm = np.sqrt(np.linalg.norm(r1)**2 + np.linalg.norm(r2)**2)
        obj = phi(x)
        history.append((obj, r_norm))

        if verbose and (k % 20 == 0 or k == max_iters - 1):
            print(f"Iter {k:4d}  obj={obj:.3e}  r={r_norm:.2e}")

        if r_norm < 1e-7:
            if verbose:
                print("Converged", k)
            break

    lam_est, Q_list_est, Qp_list_est = unpack_params_general(x, dims, F)
    Q_est = reconstruct_Q_general(lam_est, Q_list_est, Qp_list_est)
    return lam_est, Q_list_est, Qp_list_est, Q_est, history


def rand_factor(I, F):
    """Random column-stochastic factor with shape (I, F)."""
    M = np.random.rand(I, F)
    M /= M.sum(axis=0, keepdims=True)
    return M


def run_sanity_test(dims=(3,3), F_true=1, seed=0,
                    max_iters=200, inner_steps=30, step_size=0.05, verbose=True):
    np.random.seed(seed)
    D = len(dims)

    lam_true = np.random.rand(F_true)
    lam_true /= lam_true.sum()
    Q_list_true  = [rand_factor(dims[d], F_true) for d in range(D)]
    Qp_list_true = [rand_factor(dims[d], F_true) for d in range(D)]
    Q_true = reconstruct_Q_general(lam_true, Q_list_true, Qp_list_true)

    lam_est, Q_list_est, Qp_list_est, Q_est, history = admm_lrt_general(
        Q_tilde=Q_true,
        F=F_true,
        dims=dims,
        max_iters=max_iters,
        beta=1.0,
        inner_steps=inner_steps,
        step_size=step_size,
        verbose=verbose,
        seed=seed+1,
    )

    rel_err = np.linalg.norm(Q_est - Q_true) / np.linalg.norm(Q_true)
    return {
        "lam_est": lam_est,
        "Q_list_est": Q_list_est,
        "Qp_list_est": Qp_list_est,
        "Q_est": Q_est,
        "history": history,
        "rel_err": rel_err,
        "Ey": build_E(dims, F_true) @ pack_params_general(lam_est, Q_list_est, Qp_list_est),
    }
