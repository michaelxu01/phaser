# Your first reconstruction

Run one complete reconstruction end to end, on real experimental data, using a plan that
ships with Phaser. This uses the experimental PrScO₃ dataset and the LSQML conventional
solver — a good default first engine (see [Choosing a reconstruction
engine](../cookbook/engine-selection.md)).

| Metadata | Value |
| --- | --- |
| Reader level | Beginner |
| Data origin | Experimental (EMPAD) |
| Data loader | [`empad`](../cookbook/parameters/data-and-calibration.md) |
| Model | Multislice (21 slices) |
| Engine | LSQML (conventional) |
| Compute requirements | `jax` backend; a GPU is strongly recommended (this is a real reconstruction, not a toy) |
| Plan | `examples/prsco3_lsqml.yaml` |

This is the fast path. For the full parameter-by-parameter walkthrough of this same plan,
see the [LSQML reconstruction](../cookbook/reconstructions/lsqml.md) cookbook page.

## Before you start

- Phaser [installed](install.md) with the `jax` extra (`pip install -e ".[jax]"`).
- Run every command below **from the repository root**, so the plan's relative
  `sample_data/...` path resolves.

## Get the sample data

The example datasets are not in the repository — download them once from the maintainers'
Dropbox and unpack them into a `sample_data/` directory at the repository root:

```console
$ curl --output sample_data.zip -L 'https://www.dropbox.com/scl/fo/txm3k88ubrzvt541v23ir/AL-l_m6VnGlFxzHWZSSc0TA?rlkey=8qxtwnc8cwhpff6jpr5s40y6i&st=x9pbwke0&dl=1'
$ unzip sample_data.zip -d sample_data
```

The archive includes simulated and experimental MoS₂, simulated and experimental Si, and
experimental PrScO₃ data. This page uses `sample_data/experimental_prsco3/PSO.json`.

## Validate the plan

Always check the plan against the schema before running it (details in [Validating a
plan](validate-a-plan.md)):

```console
$ phaser validate examples/prsco3_lsqml.yaml
Validation of plan successful!
```

## Run it

```console
$ phaser run examples/prsco3_lsqml.yaml
```

!!! danger "Trust warning"
    This plan uses an expression schedule, which evaluates arbitrary Python via `eval`
    (`phaser/hooks/schedule.py`). A plan file that uses one is equivalent to a script —
    only run plans you trust. See [Schedules and
    flags](../cookbook/parameters/schedules-and-flags.md).

This is a full multislice reconstruction (21 object slices, 8 probe modes, 100
iterations), so it takes real time — minutes on a GPU, considerably longer on CPU. Two
ways to see progress without waiting for the end:

- The plan writes images every 10 iterations (`save_images: {every: 10}`), so
  reconstructed object-phase and probe images appear in the output directory as it runs —
  watch those.
- To just confirm it works quickly, copy the plan and lower `niter` (for example to `10`),
  then run the copy.

## What you get

While it runs, Phaser logs each iteration's detector error (it should trend downward and
level off). On completion, the plan's `save`/`save_images` settings produce, in the output
directory:

- HDF5 state files `iter010.h5` … `iter100.h5` (every 10 iterations), each a full
  `ReconsState` readable with `phaser.state.ReconsState.read_hdf5`;
- TIFF images at the same cadence: `probe`, `probe_recip`, `object_phase_sum`,
  `object_mag_sum`, `object_phase_stack`, `object_mag_stack`;
- a `finished` marker file.

A successful run shows the detector error decreasing and plateauing, and the summed
object-phase image resolving the atomic-column structure of the specimen. (Output paths
and cadence above are read from the plan; the exact error values and wall time depend on
your hardware and are not reproduced here.)

## Next steps

- [LSQML reconstruction](../cookbook/reconstructions/lsqml.md) — the full walkthrough of
  this plan: every parameter, the execution flow, and failure modes.
- [Experimental reconstruction (Si, gradient descent)](../cookbook/reconstructions/empad-experimental.md)
  — the same real-data path with the gradient engine, Poisson noise, and position
  refinement.
- [Choosing a reconstruction engine](../cookbook/engine-selection.md) and the
  [Cookbook](../cookbook/index.md) — to adapt a plan to your own data.

## Maintainer sources

- `examples/prsco3_lsqml.yaml`
- `phaser/cli/__init__.py`
- `phaser/cli/validate.py`
- `README.md`
