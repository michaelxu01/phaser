# Installing Phaser

Install Phaser from a source checkout with `pip`, and pick a
[backend](../concepts/glossary.md#backend) — the array library it computes on. There is no
PyPI or conda package yet; you clone the repository and install it in editable mode.

For most new users: install with the **JAX** backend. JAX runs on CPU or GPU and is
required by the gradient-descent engine (Torch also works there; NumPy and CuPy do not).

## 1. Clone the repository

```console
$ git clone https://github.com/hexane360/phaser
$ cd phaser
```

We recommend a conda environment or a Python virtual environment to keep dependencies
isolated, though it is not required. Phaser needs Python 3.10 or newer.

## 2. Choose a backend

| Backend | Install extra | Runs on | Gradient engine? | Notes |
| --- | --- | --- | --- | --- |
| NumPy | *(none — always installed)* | CPU | No | Simplest, slowest. Fine for small conventional (ePIE/LSQML) reconstructions. |
| JAX | `jax` | CPU or GPU | **Yes** | Recommended default. GPU builds need a CUDA-matched JAX (see below). |
| Torch | `torch` | CPU or GPU | **Yes** | Also supports the gradient engine (verified — corrects an older "JAX-only" claim). Runs uncompiled; see the [JAX guide](../architecture/jax.md). |
| CuPy | `cupy11`, `cupy12`, or `cupy13` | GPU (CUDA) | No | Pick the extra matching your CUDA toolkit version. |

The gradient-descent engine requires JAX or Torch; under NumPy or CuPy it raises a
`ValueError` at engine start (`phaser/execute.py`). The conventional engines (ePIE, LSQML)
run on any backend.

**GPU builds first.** If you want GPU acceleration with JAX or CuPy, follow those
projects' own install instructions for your CUDA version *before* installing Phaser, and
confirm they work:

```console
$ python -c "import jax; print(jax.default_backend())"   # 'gpu' on CUDA, else 'cpu'
```

JAX does not support CUDA on Windows.

## 3. Install Phaser

Editable install with the extras you chose (quote the brackets in most shells):

```console
$ python -m pip install -e ".[jax]"           # recommended default
$ python -m pip install -e ".[jax,cupy12,web]"  # multiple extras
$ python -m pip install -e .                    # NumPy only
```

Optional extras beyond the backends: `web` (the reconstruction server and workers) and
`dev` (test tooling). For [Optuna](https://optuna.org/) hyperparameter sweeps, install it
separately with `pip install optuna`.

## 4. Confirm it works

```console
$ phaser --help
```

This should list the subcommands `run`, `serve`, `worker`, and `validate`. `phaser run`
executes a plan and `phaser validate` checks one without running it — see
[Validating a plan](validate-a-plan.md). Then run your
[first reconstruction](first-reconstruction.md) end to end.

## Maintainer sources

- `pyproject.toml`
- `README.md`
- `phaser/cli/__init__.py`
- `phaser/execute.py`
