# Termination and diagnostics

How long an engine runs, how its error is computed and sampled, what NaN/Inf checks fire,
and how early stopping ("patience") works. All fields are declared on the shared
`EnginePlan` base and inherited by both engine families (`phaser/plan.py:43-98`); a few are
read by only one, noted below. Types and defaults are in the
[plan reference](../../generated/plan/index.md).

## `niter` and iteration counting

`niter` (`int`, default `10`, `phaser/plan.py:56`) sets one engine's iteration-loop length.
A plan chains engines (`engines:` is a list); each engine counts and resets its own `niter`.
The right value depends entirely on your data — guidance beyond
[the reconstructions](../reconstructions/index.md) is pending.

`ReconsState.iter` (`IterState`, `phaser/state.py:31-42`) separates the current engine from
the whole run:

| Field | Meaning |
| --- | --- |
| `engine_num` | 1-indexed engine number (`0` = before any engine) |
| `engine_iter` | iteration within the current engine (`0` = before any) |
| `total_iter` | iteration across the whole reconstruction |
| `n_engine_iters` | iterations in this engine (`= niter`) |
| `n_total_iters` | iterations across the whole reconstruction |

Every [flag and schedule](schedules-and-flags.md) reads `engine_iter`/`niter`, not the
totals — so a schedule in a later engine restarts from iteration 1.

```yaml
engines:
  - type: gradient
    niter: 200   # examples/smoke/single_slice_gradient.yaml
```

## Error calculation: `calc_error` and `calc_error_fraction`

- `calc_error` — flag, default `SimpleFlag(after=0, every=1)` (true every iteration from 1),
  `phaser/plan.py:81`. Whether the engine computes a reported error this iteration.
- `calc_error_fraction` — `float`, default `0.1`, `phaser/plan.py:82`. Fraction of an
  iteration's groups the error is computed over, once `calc_error` is true.

`calc_error_fraction` feeds `mask_fraction_of_groups` (`phaser/utils/misc.py:199-208`):
`n_required = max(1, ceil(n_groups * fraction))` — an evenly spaced subset (or all groups if
the fraction reaches the total). At the default `0.1`, roughly one group in ten per
iteration is sampled and averaged. The `max(1, ...)` floor means at least one group is
always sampled, so the effective range is `(0, 1]`. Evaluated per iteration (`calc_error`)
and per group within it, before the solver's per-group update.

!!! warning "Restriction — gradient engine ignores both fields"
    `phaser/engines/gradient/run.py` never reads `calc_error`/`calc_error_fraction` (verified
    by search), though `GradientEnginePlan` inherits them and `phaser validate` accepts them.
    The gradient engine always computes `total_loss` per group — that value *is* the
    objective it differentiates, with no skippable error step. Both fields matter only for
    the conventional engines, whose reported error is a squared-intensity difference computed
    separately from the noise model's wave update (see
    [Noise-model hooks](../../architecture/hooks/noise-models.md#lifecycle-point)).

```yaml
engines:
  - type: conventional
    calc_error: {every: 5}
    calc_error_fraction: 0.25
```

## Finite-value (NaN/Inf) checks

Both engines check for non-finite values every iteration and raise `ValueError` immediately,
stopping rather than continuing with corrupted state: the conventional engine calls
`check_finite(object.data, probe.data, ...)` (`phaser/engines/conventional/run.py:109`), the
gradient engine checks `numpy.isfinite(total_loss)` (`phaser/engines/gradient/run.py:301`).

`check_every_group` (`bool`, default `False`, `phaser/plan.py:97`) additionally runs the same
check after **every group** — conventional solvers on `object.data`/`probe.data`
(`solvers.py:125-126,399-400`), the gradient engine on `total_loss` (`run.py:297`) — catching
divergence one group earlier at the cost of an extra check per group. A finite-value failure
raises immediately and is never retried; it is unrelated to early termination (a deliberate
stop on a *converged*, not diverged, trend).

```yaml
engines:
  - type: gradient
    check_every_group: true
```

## Per-group observer notification: `send_every_group`

`send_every_group` (`bool`, default `False`, `phaser/plan.py:98`) forces observers (logging,
remote status, checkpointing) to be notified after every group instead of every iteration
(`observer.update_group(state, force=...)`, `run.py:299`, `solvers.py:134,408`). A
diagnostic-granularity choice with a data-movement cost, read once per group; see
[Observers](../../architecture/observers.md). `examples/mos2_epie.yaml` ships it commented
out.

## Early termination ("patience")

- `early_termination` — `int | None`, default `None`, `phaser/plan.py:88`. A positive integer
  enables stopping after that many iterations without improvement; `None` runs the full
  `niter`.
- `early_termination_smoothing` — `float`, default `0.9`, `phaser/plan.py:90`.

Setting `early_termination` wraps the engine's observer in
`PatienceObserver(...)` (`phaser/execute.py:60-65`), which watches `total_loss` each
iteration (`phaser/observer.py:191-209`): a new best resets the no-improvement counter,
otherwise it increments, and reaching `early_termination` raises `EarlyTermination`, which
`execute.py` treats as the final state. Checked once per iteration, identically on both
engines.

!!! warning "Restriction — `early_termination_smoothing` has no effect"
    `PatienceObserver` computes `smoothed_error` from the smoothing factor but the stop
    decision compares the **raw** `error` to `best_error`; `smoothed_error` is stored and
    never read (verified by search — dead bookkeeping, as is `_error_from_state`). Changing
    `early_termination_smoothing` currently does not change when the engine stops.

`early_termination` is a count of iterations; `examples/czo_grad.yaml` uses `5` for early
stages and `15` for a later higher-resolution stage (noisier error, longer patience). Early
termination in one engine defaults to continuing to the next engine, not stopping the whole
plan (`continue_next_engine=True`, not exposed as a plan field, `execute.py:60-63`).

```yaml
engines:
  - type: gradient
    niter: 100
    early_termination: 5   # from examples/czo_grad.yaml
```

## Maintainer sources

- `phaser/plan.py`
- `phaser/state.py`
- `phaser/execute.py`
- `phaser/observer.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/conventional/{run,solvers}.py`
- `phaser/utils/misc.py`
- `docs/architecture/hooks/noise-models.md`
- `examples/czo_grad.yaml`, `examples/mos2_epie.yaml`, `examples/smoke/single_slice_gradient.yaml`
