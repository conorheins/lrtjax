from lrtjax.lrt_generalized_numpy import (
        run_sanity_test,
    )


def print_summary(dims, result):
    history = result["history"]
    print("Final obj:", history[-1][0])
    print("Relative Frobenius error on Q:", result["rel_err"])
    Ey = result["Ey"]
    print("Ey (first few):", Ey[: min(10, Ey.size)])
    lam_est = result["lam_est"]
    Q_list_est = result["Q_list_est"]
    Qp_list_est = result["Qp_list_est"]
    print("lambda_est sum:", lam_est.sum())
    for d in range(len(dims)):
        print(f"Q{d+1} col sums :", Q_list_est[d].sum(axis=0))
        print(f"Q{d+1}' col sums:", Qp_list_est[d].sum(axis=0))


if __name__ == "__main__":
    print("=== D=2, rank-1 ===")
    res = run_sanity_test(dims=(3, 3), F_true=1)
    print_summary((3, 3), res)

    print("\n=== D=2, rank-2 ===")
    res = run_sanity_test(dims=(3, 3), F_true=2, max_iters=400, inner_steps=40, verbose=False)
    print_summary((3, 3), res)

    print("\n=== D=3, rank-1 ===")
    res = run_sanity_test(dims=(2, 2, 2), F_true=1, max_iters=250, inner_steps=30, verbose=False)
    print_summary((2, 2, 2), res)
