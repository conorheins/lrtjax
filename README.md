# lrtjax

Repository that provides a JAX and NumPY implementation of the **low-rank tensor (LRT) / CP decomposition** and the **ADMM** algorithm introduced in

> [“Low-Rank Tensors for Multi-Dimensional Markov Models” (2024)](https://arxiv.org/abs/2411.02098) by Madeline Navarro, Sergio Rozada, Antonio G. Marques, Santiago Segarra

The original github repository accompanying the paper can be found here (PyTorch implementation): https://github.com/sergiorozada12/tensor-mc

The code focuses on the *transition kernel* factorization (their Eq. (2)) and the vectorized ADMM formulation around Eqs. (10)–(11), with both:

- a **2D special case** (nice and readable), and  
- a **general D-dimensional implementation** in NumPy and JAX.

---

## Installation (with `uv`)

The easiest way to set things up is with [`uv`](https://github.com/astral-sh/uv):

```bash
# Clone the repo
git clone https://github.com/conorheins/lrtjax.git
cd lrtjax

# Create and activate a virtualenv with uv
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies from pyproject.toml
uv sync
```

You can also use plain pip to install

```bash
pip install -e .
```
from the repo root (but `uv` is the recommended path).

## Running the demos

Assuming you’re in the repo root and your `uv` env is active:

### 1. 2D demos

```bash
python lrt_2d_demo.py
```

You should see output like:

```text
----- Rank-1 test -----
Iter    0  obj=...
...
Final obj: ...
Relative Frobenius error on Q: ...
lambda_est sum: ...
Q1 col sums: [...]
Q1' col sums: [...]
```

This uses the **2D-specific** ADMM with explicit simplex projections.

### 2. General-D (NumPy/JAX hybrid) demos

```bash
python lrt_generalized_demo.py
```

This runs:

* `D=2, rank=1`
* `D=2, rank=2`
* `D=3, rank=1`

and prints:

* final objective value,
* relative Frobenius error between recovered `Q` and ground-truth `Q`,
* `Ey` (to check the linear constraints),
* column sums for each factor (to check simplex constraints).

### 3. General-D (pure JAX) demos

```bash
python lrt_generalized_demo_jax.py
```

---

## Notes

* All demos currently use **synthetic data**: ground-truth λ and factor matrices sampled at random, then we run ADMM to recover them.
* Because the CPD problem is **non-convex**, the algorithms can converge to different local optima depending on:

  * rank `F`,
  * initialization (seed),
  * ADMM hyperparameters (`beta`, `inner_steps`, `step_size`, `max_iters`).
* For quick experiments, tweak these hyperparameters directly in the demo scripts.
