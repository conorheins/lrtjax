from lrtjax.lrt_generalized_jax import run_sanity_test

if __name__ == "__main__":
    print("=== D=2, rank-1 ===")
    run_sanity_test(dims=(3, 3), F_true=1)

    print("\n=== D=2, rank-2 ===")
    run_sanity_test(dims=(3, 3), F_true=2, max_iters=400, inner_steps=40, verbose=False)

    print("\n=== D=3, rank-1 ===")
    run_sanity_test(dims=(2, 2, 2), F_true=1, max_iters=250, inner_steps=30, verbose=False)
