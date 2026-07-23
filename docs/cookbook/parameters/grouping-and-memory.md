# Grouping and memory

The options that batch scan positions into [groups](../../concepts/glossary.md#group) and
control how much data sits on the compute device: `grouping`, `compact`, `shuffle_groups`,
`buffer_n_groups`, `jit_unroll_slices`. All are shared `EnginePlan` fields (types in the
[plan reference](../../generated/plan/index.md#conventionalengineplan)), read identically by
both engines via `GroupManager`/`stream_patterns`
(`phaser/engines/common/simulation.py`). JIT/retracing mechanics are in
[the JAX guide](../../architecture/jax.md), linked where relevant.

A **group** is the batch of scan positions simulated and updated together in one inner-loop
pass — the practical unit of compute and device memory, like a minibatch. `grouping` sets
its size, `compact` how positions are assigned, `shuffle_groups` whether that changes between
iterations, `buffer_n_groups` how many groups' patterns sit on the device at once.

## `grouping`

| Type (generated) | Default | Units |
| --- | --- | --- |
| `int \| None` | Schema default `None`; resolves to `64` at runtime (`GroupManager.__init__`, `phaser/engines/common/simulation.py:31`, `self.grouping = grouping or 64`) | scan positions per group |

`grouping` is the largest number of scan positions simulated together in one group; the
last group in a pass is usually smaller, since the position count need not divide evenly
(`phaser/engines/common/simulation.py:36`, `GroupManager.n_groups`). This is a
memory/compute trade-off, not correctness: a larger `grouping` makes fewer, bigger groups
per iteration (more parallelism, more device memory per group); a smaller one makes more,
smaller groups (less memory per group, more overhead and, under JAX, more retracing risk
from the uneven final group — see
[Grouping changes](../../architecture/jax.md#jit-boundaries-in-the-gradient-engine)).

Read once, at the start of the engine that declares it, to build that engine's
`GroupManager`. Repository examples set `grouping` from `16` to `256`
(`examples/mos2_epie.yaml:23`: `16`; `examples/prsco3_lsqml.yaml:25`: `16`;
`examples/czo_grad.yaml:33,89`: `128`; `examples/mos2_lsqml.yaml:24`: `256`), and
`examples/czo_grad.yaml:154` records a concrete memory-driven choice: at `sim_shape: [192,
192]`, `grouping: 128` requested 41 GB and was reduced to `grouping: 64` (comment "128
request 41 Gb mem"). A formula for the memory a given `grouping` needs is not evidenced
here; see [Memory-scaling intuition](#memory-scaling-intuition) for the qualitative
picture.

## `compact`

| Type (generated) | Default |
| --- | --- |
| `bool` | `False` |

`compact` chooses how scan positions are partitioned into groups:

- **`False` (the default) — sparse/random grouping.** `create_sparse_groupings`
  (`phaser/utils/misc.py:96-115`) shuffles all position indices and splits them into
  equal-sized chunks — a group's positions are scattered arbitrarily across the scanned
  region.
- **`True` — spatially compact grouping.** `create_compact_groupings`
  (`phaser/utils/misc.py:118-179`) seeds group centroids with k-means clustering over the
  scan positions, then greedily assigns each position to its nearest centroid's group — a
  group's positions are spatially clustered together in the scanned region.

`compact` also changes `shuffle_groups`'s *effective* default (see next section).
Selection guidance beyond the mechanical difference is pending — no example, benchmark, or
test here evidences when compact grouping should be preferred, or its effect on
convergence or wall-clock time.

## `shuffle_groups`

!!! warning "Restriction"
    `shuffle_groups`'s runtime resolution is under review — see
    [checklist blocker B21](../../design/implementation-checklist.md). Do not follow
    advice to explicitly set `shuffle_groups: false`; it does not do what that value would
    suggest, as described below.

| Type (generated) | Default |
| --- | --- |
| `bool \| SimpleFlag \| FlagHook \| None` | Schema default `None`; see the runtime resolution below |

`shuffle_groups` controls whether groups are rebuilt (new random/compact partition) or just
re-ordered between iterations — passed to `GroupManager.iter`
(`phaser/engines/common/simulation.py:44-52`): true builds a fresh partition for that
iteration; otherwise the *same* groups from the previous partition are simply visited in a
shuffled order (`shuffled()`, not repartitioned).

The runtime value is **not** the schema default directly — both engines resolve it as
`props.shuffle_groups or not props.compact`
(`phaser/engines/conventional/run.py:31`, `phaser/engines/gradient/run.py:179`, generated
runtime notes on the
[`ConventionalEnginePlan`](../../generated/plan/index.md#conventionalengineplan)/[`GradientEnginePlan`](../../generated/plan/index.md#gradientengineplan)
reference). Because this is a plain `or`, **any falsy `shuffle_groups` value — including an
explicit `False`, not only the unset `None` default — is silently overridden to `not
compact`**:

| `shuffle_groups` you set | `compact` | Effective behavior |
| --- | --- | --- |
| unset (`None`) or `False` | `False` | `True` — groups are rebuilt every iteration |
| unset (`None`) or `False` | `True` | `False` — the same compact groups are reused, only re-ordered |
| any truthy value (e.g. `True`) | either | `True` — groups are rebuilt every iteration |

So there is currently no way to make `shuffle_groups: false` take effect under
`compact: false` — it is silently replaced by `True` (blocker B21, likely an unintended `or`
consequence). Consulted once per iteration.

## `buffer_n_groups`: tri-state device-transfer control

| Type (generated) | Default | Units |
| --- | --- | --- |
| `int \| None` | `2` | groups |

`buffer_n_groups` (`phaser/plan.py:60-65`) controls how many groups' worth of diffraction
patterns transfer to the compute device at once, read identically by both engines via the
shared `stream_patterns` helper
(`phaser/engines/common/simulation.py:58-86`; called from
`phaser/engines/gradient/run.py:195-207` and `phaser/engines/conventional/run.py:44-49`).
It has **three** distinct states:

| Value | Behavior | Memory/throughput trade-off |
| --- | --- | --- |
| `0` | **Synchronous, no prefetch.** Transfers one group, blocking (`block_until_ready`) before yielding it — the next transfer doesn't start until this group is consumed (`stream_patterns`, `phaser/engines/common/simulation.py:62-66`). | Least device memory; no transfer/compute overlap. |
| a positive integer `N` (default `2`) | **Prefetch `N` groups ahead.** A bounded queue holds up to `N` in-flight transfers, feeding a new one as soon as the oldest is consumed, so transfer for group *k+N* can overlap compute on group *k* (`phaser/engines/common/simulation.py:68-86`). | Moderate device memory (`N` groups' worth); overlaps transfer with compute. |
| `None` (`~` in YAML) | **Load the entire dataset up front.** Both engines check `buffer_n_groups is None` before the main loop and transfer the whole patterns array once, indexing groups directly out of the resident array instead of calling `stream_patterns` at all. | Most device memory (the whole dataset); no per-group transfer overhead. |

Consulted once per group ([exact code paths](../../architecture/jax.md#buffer_n_groups-tri-state-device-transfer-semantics)).
A memory/throughput trade-off, not correctness: `0` on a memory-constrained device with a
large dataset, a positive `N` (default `2`) when memory allows prefetch, `None` only when the
dataset fits comfortably and you want to drop transfer overhead. No example evidences a
specific `N`; all leave it unset.

## `jit_unroll_slices`

!!! warning "Restriction"
    JAX backend only — this field has no effect under `numpy`, `cupy`, or `torch`.

| Type (generated) | Default (schema) | Default (runtime, per engine) |
| --- | --- | --- |
| `None \| bool \| int` | `None` | Conventional: resolves to `False` (no unrolling, `phaser/engines/conventional/solvers.py:37,324`). Gradient: resolves to `10` (`phaser/engines/gradient/run.py:162`). |

`jit_unroll_slices` controls the `unroll` argument passed to `jax.lax.scan`/`fori_loop`
during multislice propagation: "`True` or `0` unrolls all slices, `False` or `1` disables
unrolling... larger unrolling may be faster, at the expense of increased compilation time"
(`EnginePlan.jit_unroll_slices` docstring, `phaser/plan.py:67-74`). Only affects multislice
objects under JAX — see
[Simulation geometry: `slices`](simulation-geometry.md#slices) (more object slices means
more to unroll or loop over either way) and
[Multislice traversal: `jax.lax.scan`/`fori_loop` and `jit_unroll_slices`](../../architecture/jax.md#multislice-traversal-jaxlaxscanfori_loop-and-jit_unroll_slices)
for the full mechanism, including why it has no effect on Torch, NumPy, or CuPy (Torch
always runs multislice as a plain unrolled Python loop — no traced graph to control).
Guidance on choosing a value other than the two engines' defaults is not evidenced here.

## Memory-scaling intuition

The dominant, code-verified memory drivers for one group's simulation are the simulated
detector shape (`sim_shape`, see
[Simulation geometry](simulation-geometry.md#sim_shape-and-resize_method)), probe-mode
count (`probe_modes`), object-slice count (`slices`), and `grouping` itself (see the
`czo_grad.yaml:151,154` case under [`grouping`](#grouping) above). A precise formula
relating these fields to bytes of device memory is not derived here — this page states
the qualitative direction (more pixels, modes, slices, or positions per group means more
device memory) rather than a numeric model.

## Minimal example

Reducing per-group memory on a large simulated shape by lowering `grouping` and disabling
prefetch, based on the pattern in `examples/czo_grad.yaml`:

```yaml
engines:
  - type: gradient
    sim_shape: [192, 192]
    probe_modes: 8
    grouping: 64
    buffer_n_groups: 0
    # ...
```

*Verification pending:* illustrates a memory-constrained configuration drawn from
`examples/czo_grad.yaml:151-154`; not re-validated for this page (omits required fields for
brevity), and `buffer_n_groups: 0` specifically was not exercised — only the `grouping`
reduction is an observed, recorded outcome.

## Maintainer sources

- `phaser/plan.py`
- `phaser/engines/common/simulation.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/gradient/run.py`
- `phaser/utils/misc.py`
- `tests/test_misc.py`
- `examples/czo_grad.yaml`, `examples/mos2_epie.yaml`, `examples/mos2_lsqml.yaml`, `examples/prsco3_lsqml.yaml`
- `docs/design/implementation-checklist.md` (B21)
