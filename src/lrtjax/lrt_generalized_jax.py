import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .common import nway_outer_jax as nway_outer
from .common import reconstruct_Q_general_jax as reconstruct_Q_general
from .common import pack_params_general_jax as pack_params_general
from .common import unpack_params_general_jax as unpack_params_general
from .common import build_E_jax as build_E
from .common import build_A_B_b_jax as build_A_B_b

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

    grad_lam = jnp.zeros_like(lam)
    grad_Q_list = [jnp.zeros_like(Q_list[d]) for d in range(D)]
    grad_Qp_list = [jnp.zeros_like(Qp_list[d]) for d in range(D)]

    s_axes = list(range(D))
    sprime_axes = list(range(D, 2 * D))

    for f in range(F):
        left_vecs = [Q_list[d][:, f] for d in range(D)]
        right_vecs = [Qp_list[d][:, f] for d in range(D)]
        left_tensor = nway_outer(left_vecs)  # shape dims
        right_tensor = nway_outer(right_vecs)  # shape dims

        rank1 = jnp.multiply.outer(left_tensor, right_tensor)
        grad_lam = grad_lam.at[f].set(jnp.tensordot(E_res, rank1, axes=E_res.ndim))

        temp = jnp.tensordot(E_res, right_tensor, axes=(sprime_axes, s_axes))  # (I1,...,ID)
        temp2 = jnp.tensordot(E_res, left_tensor, axes=(s_axes, s_axes))  # (I1,...,ID)

        for d in range(D):
            I_d = dims[d]

            Qd_col = Q_list[d][:, f]  # (I_d,)
            shape = [1] * D
            shape[d] = I_d
            Qd_bc = jnp.broadcast_to(Qd_col.reshape(shape), dims)
            B = jnp.where(Qd_bc > eps, left_tensor / Qd_bc, 0.0)

            w = temp * B  # (I1,...,ID)
            axes_sum = tuple(ax for ax in range(D) if ax != d)
            grad_Q_d_col = lam[f] * w.sum(axis=axes_sum)
            grad_Q_list[d] = grad_Q_list[d].at[:, f].add(grad_Q_d_col)

            Qdp_col = Qp_list[d][:, f]
            Qdp_bc = jnp.broadcast_to(Qdp_col.reshape(shape), dims)
            Bp = jnp.where(Qdp_bc > eps, right_tensor / Qdp_bc, 0.0)

            w2 = temp2 * Bp
            grad_Qp_d_col = lam[f] * w2.sum(axis=axes_sum)
            grad_Qp_list[d] = grad_Qp_list[d].at[:, f].add(grad_Qp_d_col)

    return pack_params_general(grad_lam, grad_Q_list, grad_Qp_list)


def admm_lrt_general(
    Q_tilde,
    F,
    dims,
    max_iters=200,
    beta=1.0,
    inner_steps=20,
    step_size=0.05,
    verbose=False,
    seed=0,
):
    """
    General-D ADMM demo following the vectorized formulation (10)-(11).

    Q_tilde: empirical joint tensor, shape (I1,...,ID, I1,...,ID)
    F      : CP rank
    dims   : tuple/list (I1,...,ID)
    """
    key = jax.random.PRNGKey(seed)
    D = len(dims)
    J = sum(dims)
    n = F * (2 * J + 1)  # length of x and y

    # --- Initialize λ, Q_d, Q'_d on simplices (nonnegative, col sums = 1) ---
    key, lam_key = jax.random.split(key)
    lam = jax.random.uniform(lam_key, (F,))
    lam = lam / lam.sum()

    def rand_factor(k, I, F_):
        M = jax.random.uniform(k, (I, F_))
        return M / jnp.sum(M, axis=0, keepdims=True)

    key, q_key = jax.random.split(key)
    q_keys = jax.random.split(q_key, 2 * D)
    Q_list = [rand_factor(q_keys[d], dims[d], F) for d in range(D)]
    Qp_list = [rand_factor(q_keys[D + d], dims[d], F) for d in range(D)]

    x = pack_params_general(lam, Q_list, Qp_list)
    assert x.size == n

    # --- Build E and matrices for y-step ---
    E = build_E(dims, F)
    m = E.shape[0]  # = 2 D F + 1
    M_mat = jnp.eye(n) + E.T @ E  # (I + E^T E)

    # y and duals (scaled)
    y = x.copy()
    u1 = jnp.zeros_like(x)
    u2 = jnp.zeros(m)
    one_m = jnp.ones(m)

    def phi(xvec):
        lam_, Q_list_, Qp_list_ = unpack_params_general(xvec, dims, F)
        Q_hat = reconstruct_Q_general(lam_, Q_list_, Qp_list_)
        return 0.5 * jnp.sum((Q_hat - Q_tilde) ** 2)

    # history = []

    def inner_step_fn(carry, _):
        xk, c = carry
        g = grad_phi_general(xk, Q_tilde, dims, F)
        g_total = g + beta * (xk - c)
        xk = xk - step_size * g_total
        xk = jnp.maximum(xk, 0.0)  # enforce x >= 0 (indicator term)
        return (xk, c), None

    tol = 1e-7

    def outer_step_fn(carry, k):
        x, y, u1, u2, stop, obj_prev, r_prev = carry

        def do_step(_):
            # ----- x-step: projected gradient descent on φ(x) + (β/2)||x - y + u1||^2 -----
            c = y - u1
            (xk, _), _ = jax.lax.scan(inner_step_fn, (x, c), None, length=inner_steps)
            x_new = xk

            # ----- y-step: solve (I + E^T E) y = x + u1 + E^T (1 - u2) -----
            rhs = x_new + u1 + E.T @ (one_m - u2)
            y_new = jnp.linalg.solve(M_mat, rhs)

            # ----- dual updates (scaled) -----
            r1 = x_new - y_new
            r2 = E @ y_new - one_m
            u1_new = u1 + r1
            u2_new = u2 + r2

            r_norm = jnp.sqrt(jnp.linalg.norm(r1) ** 2 + jnp.linalg.norm(r2) ** 2)
            obj = phi(x_new)
            stop_new = jnp.logical_or(stop, r_norm < tol)

            return (x_new, y_new, u1_new, u2_new, stop_new, obj, r_norm), (obj, r_norm)

        def skip_step(_):
            return (x, y, u1, u2, stop, obj_prev, r_prev), (obj_prev, r_prev)

        return jax.lax.cond(stop, skip_step, do_step, operand=None)

    init_obj = phi(x)
    init_r = jnp.inf
    carry = (x, y, u1, u2, False, init_obj, init_r)
    final_carry, history_scan = jax.lax.scan(outer_step_fn, carry, jnp.arange(max_iters))
    x, y, u1, u2, _, _, _ = final_carry
    obj_hist, r_hist = history_scan  # each shape (max_iters,)
    history = [(float(obj_hist[i]), float(r_hist[i])) for i in range(max_iters)]

    lam_est, Q_list_est, Qp_list_est = unpack_params_general(x, dims, F)
    Q_est = reconstruct_Q_general(lam_est, Q_list_est, Qp_list_est)
    return lam_est, Q_list_est, Qp_list_est, Q_est, history


def run_sanity_test(
    dims=(3, 3),
    F_true=1,
    seed=0,
    max_iters=200,
    inner_steps=30,
    step_size=0.05,
    verbose=True,
):
    key = jax.random.PRNGKey(seed)
    D = len(dims)

    def rand_factor(k, I, F_):
        M = jax.random.uniform(k, (I, F_))
        return M / jnp.sum(M, axis=0, keepdims=True)

    key, lam_key = jax.random.split(key)
    lam_true = jax.random.uniform(lam_key, (F_true,))
    lam_true = lam_true / lam_true.sum()

    key, q_key = jax.random.split(key)
    q_keys = jax.random.split(q_key, 2 * D)
    Q_list_true = [rand_factor(q_keys[i], dims[i], F_true) for i in range(D)]
    Qp_list_true = [rand_factor(q_keys[D + i], dims[i], F_true) for i in range(D)]
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
        seed=seed + 1,
    )

    rel_err = jnp.linalg.norm(Q_est - Q_true) / jnp.linalg.norm(Q_true)
    print("Final obj:", history[-1][0])
    print("Relative Frobenius error on Q:", float(rel_err))

    # Check that Ey ≈ 1 and x ≈ y → simplex constraints & nonnegativity
    E = build_E(dims, F_true)
    x = pack_params_general(lam_est, Q_list_est, Qp_list_est)
    Ey = E @ x
    print("Ey (first few):", jnp.asarray(Ey[: min(10, Ey.size)]))
    print("lambda_est sum:", float(lam_est.sum()))
    for d in range(D):
        print(f"Q{d+1} col sums :", jnp.asarray(Q_list_est[d].sum(axis=0)))
        print(f"Q{d+1}' col sums:", jnp.asarray(Qp_list_est[d].sum(axis=0)))


if __name__ == "__main__":
    print("=== D=2, rank-1 ===")
    run_sanity_test(dims=(3, 3), F_true=1)

    print("\n=== D=2, rank-2 ===")
    run_sanity_test(dims=(3, 3), F_true=2, max_iters=400, inner_steps=40, verbose=False)

    print("\n=== D=3, rank-1 ===")
    run_sanity_test(dims=(2, 2, 2), F_true=1, max_iters=250, inner_steps=30, verbose=False)
