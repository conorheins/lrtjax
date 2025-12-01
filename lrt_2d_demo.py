import numpy as np

from lrtjax.lrt_2d import (
        admm_lrt_2d,
        rand_factor,
        reconstruct_Q_2d,
    )

def print_summary(lam_est, Q_list_est, Qp_list_est, Q_true, Q_est, history):
    rel_err = np.linalg.norm(Q_est - Q_true) / np.linalg.norm(Q_true)
    print("Final obj:", history[-1][0])
    print("Relative Frobenius error on Q:", rel_err)
    print("lambda_est sum:", lam_est.sum())
    print("Q1 col sums:", Q_list_est[0].sum(axis=0))
    print("Q1' col sums:", Qp_list_est[0].sum(axis=0))


def run_rank1():
    np.random.seed(3)
    I1 = I2 = 3
    F_true = 1
    dims = (I1, I2)

    lam_true = np.array([1.0])  # rank-1; joint is normalized by the factors
    Q1_true = rand_factor(I1, F_true)
    Q2_true = rand_factor(I2, F_true)
    Q1p_true = rand_factor(I1, F_true)
    Q2p_true = rand_factor(I2, F_true)

    Q_true = reconstruct_Q_2d(lam_true, Q1_true, Q2_true, Q1p_true, Q2p_true)

    lam_est, (Q1_est, Q2_est), (Q1p_est, Q2p_est), Q_est, hist = admm_lrt_2d(
        Q_true,
        F=F_true,
        max_iters=500,
        rho=1.0,
        inner_steps=50,
        step_size=0.05,
        verbose=True,
        seed=0,
    )
    print_summary(lam_est, (Q1_est, Q2_est), (Q1p_est, Q2p_est), Q_true, Q_est, hist)


def run_rank2():
    np.random.seed(4)
    I1 = I2 = 3
    F_true = 2
    dims = (I1, I2)

    lam_true = np.random.rand(F_true)
    lam_true /= lam_true.sum()
    Q1_true = rand_factor(I1, F_true)
    Q2_true = rand_factor(I2, F_true)
    Q1p_true = rand_factor(I1, F_true)
    Q2p_true = rand_factor(I2, F_true)

    Q_true = reconstruct_Q_2d(lam_true, Q1_true, Q2_true, Q1p_true, Q2p_true)

    lam_est, (Q1_est, Q2_est), (Q1p_est, Q2p_est), Q_est, hist = admm_lrt_2d(
        Q_true,
        F=F_true,
        max_iters=800,
        rho=1.0,
        inner_steps=80,
        step_size=0.05,
        verbose=False,
        seed=0,
    )
    rel_err = np.linalg.norm(Q_est - Q_true) / np.linalg.norm(Q_true)
    print("Final obj:", hist[-1][0])
    print("Relative Frobenius error on Q:", rel_err)
    print("lambda_est sum:", lam_est.sum())
    print("Q1 col sums:", Q1_est.sum(axis=0))
    print("Q1' col sums:", Q1p_est.sum(axis=0))


if __name__ == "__main__":
    print("----- Rank-1 test -----")
    run_rank1()

    print("\n----- Rank-2 test -----")
    run_rank2()
