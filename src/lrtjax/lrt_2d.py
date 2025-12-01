import numpy as np


def reconstruct_Q_2d(lambda_vec, Q1, Q2, Q1p, Q2p):
    """
    Reconstruct Q(s, s') tensor for D=2 from CPD factors:
      Q(s,s') = sum_f lambda[f] * (Q1[:,f]⊗Q2[:,f]) ⊗ (Q1p[:,f]⊗Q2p[:,f])

    lambda_vec: (F,)
    Q1, Q2, Q1p, Q2p: (I_d, F)
    Returns: Q of shape (I1, I2, I1, I2)
    """
    I1, F1 = Q1.shape
    I2, F2 = Q2.shape
    assert F1 == F2 == lambda_vec.shape[0]
    Q = np.zeros((I1, I2, I1, I2), dtype=float)
    F = lambda_vec.shape[0]
    for f in range(F):
        L_f = np.outer(Q1[:, f], Q2[:, f])   # shape (I1, I2)
        R_f = np.outer(Q1p[:, f], Q2p[:, f]) # shape (I1, I2)
        Q += lambda_vec[f] * np.einsum('ab,cd->abcd', L_f, R_f)
    return Q


def pack_params(lambda_vec, Q_list, Qp_list):
    """
    x = [λ; vec(Q1); vec(Q2); vec(Q1'); vec(Q2')]
    """
    flat = [lambda_vec.ravel()]
    for Q in Q_list:
        flat.append(Q.ravel())
    for Qp in Qp_list:
        flat.append(Qp.ravel())
    return np.concatenate(flat)


def unpack_params(x, dims, F):
    """
    Inverse of pack_params for D=2.

    dims = (I1, I2)
    Returns:
      lambda_vec, [Q1, Q2], [Q1p, Q2p]
    """
    D = len(dims)
    idx = 0
    lam = x[idx:idx+F]; idx += F
    Q_list = []
    for d in range(D):
        I = dims[d]
        size = I * F
        Qd = x[idx:idx+size].reshape(I, F)
        idx += size
        Q_list.append(Qd)
    Qp_list = []
    for d in range(D):
        I = dims[d]
        size = I * F
        Qpd = x[idx:idx+size].reshape(I, F)
        idx += size
        Qp_list.append(Qpd)
    assert idx == x.size
    return lam, Q_list, Qp_list


def grad_phi_2d(x, Q_tilde, dims, F):
    """
    Gradient of φ(x) = 0.5 * ||Q_hat(x) - Q_tilde||_F^2
    for D=2.

    x: flattened params
    Q_tilde: (I1, I2, I1, I2)
    """
    I1, I2 = dims
    lam, (Q1, Q2), (Q1p, Q2p) = unpack_params(x, dims, F)
    Q_hat = reconstruct_Q_2d(lam, Q1, Q2, Q1p, Q2p)
    E = Q_hat - Q_tilde  # residual

    F_ = lam.shape[0]
    grad_lam = np.zeros_like(lam)
    grad_Q1 = np.zeros_like(Q1)
    grad_Q2 = np.zeros_like(Q2)
    grad_Q1p = np.zeros_like(Q1p)
    grad_Q2p = np.zeros_like(Q2p)

    for f in range(F_):
        L_f = np.outer(Q1[:, f], Q2[:, f])   # (I1, I2)
        R_f = np.outer(Q1p[:, f], Q2p[:, f]) # (I1, I2)

        # dφ/dλ_f = <E, L_f ⊗ R_f>
        rank1 = np.einsum('ab,cd->abcd', L_f, R_f)
        grad_lam[f] = np.tensordot(E, rank1, axes=E.ndim)

        # Contract E over s' with R_f  → temp(s1,s2)
        temp = np.tensordot(E, R_f, axes=([2, 3], [0, 1]))  # (I1, I2)

        # dφ/dQ1[:,f]
        grad_Q1[:, f] = lam[f] * (temp * Q2[:, f][None, :]).sum(axis=1)

        # dφ/dQ2[:,f]
        grad_Q2[:, f] = lam[f] * (temp * Q1[:, f][:, None]).sum(axis=0)

        # Contract E over s with L_f  → temp2(s1',s2')
        temp2 = np.tensordot(E, L_f, axes=([0, 1], [0, 1]))  # (I1, I2)

        # dφ/dQ1'[:,f]
        grad_Q1p[:, f] = lam[f] * (temp2 * Q2p[:, f][None, :]).sum(axis=1)

        # dφ/dQ2'[:,f]
        grad_Q2p[:, f] = lam[f] * (temp2 * Q1p[:, f][:, None]).sum(axis=0)

    return pack_params(grad_lam, [grad_Q1, grad_Q2], [grad_Q1p, grad_Q2p])


def proj_simplex(v):
    """
    Euclidean projection of v onto the probability simplex:
      Δ = { x ≥ 0, sum x = 1 }.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.full_like(v, 1.0 / n)
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


def project_to_constraints(z, dims, F):
    """
    Projection onto:
      λ ∈ Δ_F
      each column of Q1,Q2,Q1',Q2' ∈ simplex

    z is flattened params.
    """
    lam, Q_list, Qp_list = unpack_params(z, dims, F)

    lam_proj = proj_simplex(lam)

    Q_list_proj = []
    for Q in Q_list:
        Qp = Q.copy()
        I, F_ = Qp.shape
        for f in range(F_):
            Qp[:, f] = proj_simplex(Qp[:, f])
        Q_list_proj.append(Qp)

    Qp_list_proj = []
    for Qp in Qp_list:
        Qpp = Qp.copy()
        I, F_ = Qpp.shape
        for f in range(F_):
            Qpp[:, f] = proj_simplex(Qpp[:, f])
        Qp_list_proj.append(Qpp)

    return pack_params(lam_proj, Q_list_proj, Qp_list_proj)


def admm_lrt_2d(Q_tilde, F,
                max_iters=200,
                rho=1.0,
                inner_steps=50,
                step_size=0.05,
                verbose=False,
                seed=0):
    """
    ADMM demo for D=2 low-rank tensor factorization:

      min_{λ,Q,Q'} 0.5 || Q_tilde - [[λ, Q, Q']] ||_F^2
      s.t. λ, columns of Q1,Q2,Q1',Q2' are probability vectors.

    Q_tilde: (I1, I2, I1, I2) empirical joint tensor
    F: CP rank
    """
    np.random.seed(seed)
    I1, I2, J1, J2 = Q_tilde.shape
    assert I1 == J1 and I2 == J2
    dims = (I1, I2)

    # Random initialization on the simplices
    lam_init = np.ones(F) / F

    def rand_factor(I, F_):
        M = np.random.rand(I, F_)
        M /= M.sum(axis=0, keepdims=True)
        return M

    Q1_init = rand_factor(I1, F)
    Q2_init = rand_factor(I2, F)
    Q1p_init = rand_factor(I1, F)
    Q2p_init = rand_factor(I2, F)

    x = pack_params(lam_init, [Q1_init, Q2_init], [Q1p_init, Q2p_init])
    z = x.copy()
    u = np.zeros_like(x)

    def phi(xvec):
        lam, (Q1, Q2), (Q1p, Q2p) = unpack_params(xvec, dims, F)
        Q_hat = reconstruct_Q_2d(lam, Q1, Q2, Q1p, Q2p)
        return 0.5 * np.sum((Q_hat - Q_tilde) ** 2)

    history = []

    for k in range(max_iters):
        # x-step: gradient descent on φ(x) + (ρ/2)||x - z + u||^2
        c = z - u
        xk = x.copy()
        for _ in range(inner_steps):
            g = grad_phi_2d(xk, Q_tilde, dims, F)
            g_total = g + rho * (xk - c)
            xk = xk - step_size * g_total
        x = xk

        # z-step: prox of indicator_C  → projection onto constraints
        z = project_to_constraints(x + u, dims, F)

        # dual update
        u = u + x - z

        # diagnostics
        r_norm = np.linalg.norm(x - z)
        obj = phi(z)
        history.append((float(obj), float(r_norm)))

        if verbose and (k % 20 == 0 or k == max_iters - 1):
            print(f"Iter {k:4d}  obj={float(obj):.3e}  r={float(r_norm):.2e}")

        if r_norm < 1e-7:
            if verbose:
                print("Converged", k)
            break

    lam_est, (Q1_est, Q2_est), (Q1p_est, Q2p_est) = unpack_params(z, dims, F)
    Q_est = reconstruct_Q_2d(lam_est, Q1_est, Q2_est, Q1p_est, Q2p_est)
    return lam_est, (Q1_est, Q2_est), (Q1p_est, Q2p_est), Q_est, history


def rand_factor(I, F):
    """Random column-stochastic factor with shape (I, F)."""
    M = np.random.rand(I, F)
    M /= M.sum(axis=0, keepdims=True)
    return M
