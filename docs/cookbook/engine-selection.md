# Choosing a reconstruction engine

Phaser reconstructs with one of two [**engine**](../concepts/glossary.md#engine) families
— **gradient descent**, or **conventional** (solvers **LSQML** and **ePIE**) — and a
[plan](../concepts/glossary.md#plan) can chain several engine stages in sequence (see
[Engine-boundary reshaping](../architecture/lifecycle.md#engine-boundary-reshaping)). This
page is the curated compatibility matrix for picking a starting engine, plus a short
decision guide. Read it before copying a
[minimal complete reconstruction](reconstructions/index.md) or a
[recipe](recipes/index.md).

!!! note "Curated, not yet generated"
    This matrix is **curated and verified 2026-07-22** directly against
    `phaser/plan.py`, `phaser/engines/`, the hook registries, and every plan under
    `examples/`, per the [sources of truth](../design/documentation-architecture.md#sources-of-truth).
    It will be **generated from explicit registry metadata in a later phase (WP-3)** once
    that metadata exists (see [Delivery phases](../design/documentation-architecture.md#delivery-phases));
    until then, treat this page, not the schema, as the compatibility authority — the
    plan schema accepts several combinations (for example `noise_model: poisson` on a
    conventional engine) that fail at runtime.

## Legend

Every cell uses one of these words — never color alone:

| Value | Meaning |
| --- | --- |
| **Yes** | Supported by the current code, and shown to run (an example, or this documentation's own executed code). |
| **No** | Not supported: rejected at validation time, raises at runtime, or has no code path at all. |
| **Under review** | Implemented, but a specific, verified code issue is open with a code owner (see the footnoted blocker). Do not present as fully supported. |
| **Unverified** | Could not be confirmed from code, tests, or an example without running a reconstruction; do not assume either way. |

"Example-demonstrated" means a plan under `examples/` or this documentation's own executed
code was shown to run it, with no automated test asserting correctness; "test-covered"
means a test in `tests/` does. Per
[verified blocker B12](../design/implementation-checklist.md#blocker-triage-phase-1-do-first)
and [Testing](../architecture/testing.md#example-demonstrated-versus-test-covered-b12),
**no engine, noise model, or position-correction path currently has a correctness test** —
only [one bounded reconstruction](reconstructions/simulated-single-slice-gradient.md) has
been executed and checked against a known synthetic answer ("smoke-tested" below);
everything else engine-related is example-demonstrated only.

## Backends, model support, and verification status

| Engine | Backends allowed by code | Backends demonstrated | Single-slice | Multislice | Mixed-state probes | Noise models | Verification status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Gradient descent** | JAX or Torch only[^b1] | JAX[^demo-grad] (no Torch example exists) | Yes | Yes[^multi-demo] | Yes[^modes-demo] | amplitude, anscombe, poisson[^noise-grad] | Smoke-tested (single-slice, JAX)[^smoke]; otherwise example-demonstrated[^b12] |
| **LSQML** (conventional) | numpy, cupy, jax, torch — no restriction in code[^no-restrict] | JAX only[^demo-lsqml] | Yes | Yes[^multi-demo] | Yes[^modes-demo] | amplitude, anscombe (poisson **fails**)[^b6] | Example-demonstrated only[^b12] |
| **ePIE** (conventional) | numpy, cupy, jax, torch — no restriction in code[^no-restrict] | Torch only[^demo-epie] | Yes | Yes, code-verified — no example demonstrates it[^multi-epie] | Yes[^modes-demo] | amplitude, anscombe (poisson **fails**)[^b6] | Example-demonstrated only[^b12] |

## Refinable variables

| Engine | Object | Probe | Positions | Tilt |
| --- | --- | --- | --- | --- |
| **Gradient descent** | Yes[^b12] | Yes[^b12] | Yes[^pos-grad] | Yes — the **only** engine that refines tilt[^tilt] |
| **LSQML** | Yes[^b12] | Yes[^b12] | Under review[^pos-review] — demonstrated only with the `momentum` position solver[^pos-lsqml] | No — forward-applies a fixed tilt only[^tilt] |
| **ePIE** | Yes[^b12] | Yes[^b12] | Under review[^pos-review] — **not demonstrated by any example**[^pos-epie] | No — forward-applies a fixed tilt only[^tilt] |

## Regularizer categories and restart

| Engine | Cost regularizers | Group constraints | Iteration constraints | Restart |
| --- | --- | --- | --- | --- |
| **Gradient descent** | Yes — the only engine with this field[^reg-cost] | Yes[^reg-group] | Yes[^reg-group] | Yes[^restart] |
| **LSQML** | No — no such field on `ConventionalEnginePlan`[^reg-cost] | Yes[^reg-group] | Yes[^reg-group] | Yes[^restart] |
| **ePIE** | No — no such field on `ConventionalEnginePlan`[^reg-cost] | Yes[^reg-group] | Yes[^reg-group] | Yes[^restart] |

## Known limitations

- **Gradient descent** — requires JAX or Torch[^b1]; on Torch, every JIT boundary in
  Phaser is JAX-specific, so it runs fully eager (uncompiled) — a performance, not
  correctness, limitation (see [the JAX guide](../architecture/jax.md)) — and no
  repository example exercises the Torch path. Only engine that can refine tilt or use
  the Poisson noise model. No correctness test exists (B12).
- **LSQML** — Poisson noise raises `NotImplementedError` at the first group[^b6]; no cost
  regularizers; no tilt refinement (`update_tilt` is silently inert)[^tilt]. Position
  refinement is under code-owner review[^pos-review], though every LSQML example enabling
  it uses the `momentum` position solver, unaffected by the `steepest_descent`
  bug[^b8]. No correctness test exists (B12).
- **ePIE** — same Poisson, cost-regularizer, and tilt restrictions as LSQML. Position
  refinement is weaker still: `EPIESolver.run_iteration` never passes `update_position` to
  the position-gradient computation, so it always runs regardless of the
  `update_positions` flag (wasted compute, not wrong results — application is still
  correctly gated)[^b7]; no repository example configures ePIE position refinement at
  all. No correctness test exists (B12).
- **All three engines** — an explicit `shuffle_groups: false` is silently overridden back
  to shuffling whenever `compact` is also `false`: the runtime default
  `shuffle_groups or not compact` treats any falsy value, not just `None`, as
  "unset"[^b21]. Affects [grouping and memory](parameters/grouping-and-memory.md) choices
  identically on every engine.

## Decision guide

- **Start with LSQML** for a typical, general-purpose reconstruction (amplitude or
  Anscombe noise, single- or multislice, mixed-state probes) — it runs on any backend
  without a JAX or Torch install, and works well as a coarse, fast first stage before
  gradient descent (see the
  [conventional-to-gradient recipe](recipes/conventional-to-gradient.md)). It has the
  broader demonstrated feature set of the two conventional solvers (multislice, position
  refinement) and is the solver used in this documentation's multislice examples.
- **Choose ePIE** if you want its specific update rule (to match a workflow or paper) or
  need Torch for a conventional engine today — the only demonstrated
  conventional-engine-on-Torch example uses ePIE. Do not rely on its position refinement
  yet; it is under code-owner review and undemonstrated by any example.
- **Choose gradient descent** for the Poisson noise model, tilt refinement, cost
  regularizers (a differentiable prior alongside the detector loss), or fine control over
  per-variable optimizers (Adam, SGD, Polyak-SGD, independently per
  `object`/`probe`/`positions`/`tilt`). Requires JAX or Torch. Also the only engine with
  an executed, checked smoke test in this documentation
  ([Simulated single-slice reconstruction](reconstructions/simulated-single-slice-gradient.md)).
- **A staged plan is common and supported**: run a conventional engine first (fast, any
  backend, no tilt/regularizer needs), then hand off to gradient descent for tilt,
  Poisson, or regularized refinement — `prepare_for_engine` reshapes the shared state
  automatically at that boundary (see
  [Engine-boundary reshaping](../architecture/lifecycle.md#engine-boundary-reshaping)).

## Where to go next

- [Solver hooks](../architecture/hooks/solvers.md) — full signatures, properties, and the
  position-solver restriction.
- [Noise-model hooks](../architecture/hooks/noise-models.md) — the Poisson restriction,
  with executed evidence.
- [Cost-regularizer hooks](../architecture/hooks/cost-regularizers.md), [group-constraint
  hooks](../architecture/hooks/group-constraints.md), and [iteration-constraint
  hooks](../architecture/hooks/iteration-constraints.md) — the three regularizer
  lifecycles.
- [Engine hooks](../architecture/hooks/engines.md) — the engine hook mechanism.
- [Minimal complete reconstructions](reconstructions/index.md) — one worked plan per
  engine and data source, including the executed [gradient
  smoke test](reconstructions/simulated-single-slice-gradient.md).
- [Solvers and learning rates](parameters/solvers-and-learning-rates.md) and [noise
  models](parameters/noise-models.md) parameter pages.

[^b1]: `prepare_for_engine` (`phaser/execute.py:386-387`) raises `ValueError("The gradient descent engine requires the 'jax' or 'torch' backend.")` for any backend other than JAX or Torch when the engine is a `GradientEnginePlan`. Verified blocker [B1](../design/implementation-checklist.md#blocker-triage-phase-1-do-first): this accepts **either** backend, correcting an older, JAX-only claim.
[^no-restrict]: No call in `phaser/engines/conventional/run.py` or `phaser/engines/conventional/solvers.py` checks the active backend; `prepare_for_engine`'s backend check (`phaser/execute.py:386`) only fires for `GradientEnginePlan`. `jit_unroll_slices` only emits a warning if set on a non-JAX backend (`phaser/engines/conventional/solvers.py:34,321`) — it does not reject the plan.
[^demo-grad]: Every `examples/*grad*.yaml` plan sets `backend: jax` (`czo_grad.yaml`, `mos2_grad.yaml`, `prsco3_grad.yaml`, `si_grad.yaml`, `si_grad_exp.yaml`); no repository example sets `backend: torch` for a gradient engine.
[^demo-lsqml]: `examples/mos2_lsqml.yaml`, `examples/si_lsqml.yaml`, and `examples/prsco3_lsqml.yaml` all set `backend: jax`; no repository example runs LSQML on Torch, NumPy, or CuPy.
[^demo-epie]: `examples/mos2_epie.yaml` sets `backend: torch`; no repository example runs ePIE on JAX, NumPy, or CuPy.
[^multi-demo]: `examples/si_grad.yaml`, `examples/si_grad_exp.yaml`, `examples/prsco3_grad.yaml`, `examples/czo_grad.yaml` (gradient) and `examples/si_lsqml.yaml`, `examples/prsco3_lsqml.yaml` (LSQML) all set a top-level `slices:` block (for example `si_grad.yaml:17-19`, `n: 10, total_thickness: 200`) and both `epie_run`/`lsqml_run` and the gradient `run_model` loop over `n_slices = sim.state.object.data.shape[0]` with per-slice propagation (`phaser/engines/conventional/solvers.py:214,467`; `phaser/engines/gradient/run.py:451-456`).
[^multi-epie]: `epie_run` (`phaser/engines/conventional/solvers.py:463-523`) uses the identical multislice loop structure as `lsqml_run` (`slice_forwards`/`slice_backwards` over `n_slices`) — code-verified support — but `examples/mos2_epie.yaml`, the only ePIE example, sets no `slices:` field (single-slice only).
[^modes-demo]: All conventional solvers sum over an incoherent-mode axis before comparing to measured intensity (`phaser/engines/conventional/solvers.py:179,249,440,487`) and share the same `probe_modes`/`base_mode_power` schema fields (`phaser/plan.py`, both `ConventionalEnginePlan` and `GradientEnginePlan`) with the gradient engine, which sums modes analogously inside `run_model` (`phaser/engines/gradient/run.py`). Demonstrated with `probe_modes: 4` or `8` in `examples/mos2_epie.yaml` (ePIE), `examples/mos2_lsqml.yaml`/`si_lsqml.yaml`/`prsco3_lsqml.yaml` (LSQML), and every `*_grad.yaml` (gradient).
[^noise-grad]: The gradient engine calls only `calc_loss` (`phaser/engines/gradient/run.py:460`), which all three built-in noise models implement; `poisson` is demonstrated in `examples/mos2_grad.yaml`, `czo_grad.yaml`, `prsco3_grad.yaml`, `si_grad.yaml`, `si_grad_exp.yaml`.
[^b6]: Verified blocker [B6](../design/implementation-checklist.md#blocker-triage-phase-1-do-first): `PoissonNoiseModel.calc_wave_update` raises `NotImplementedError()` unconditionally (`phaser/engines/common/noise_models.py:104-112`); the conventional solvers call only `calc_wave_update` (`phaser/engines/conventional/solvers.py:255,490`), so `noise_model: poisson` crashes at the first group on either conventional solver, even though the plan schema accepts it for both engine types. See [Noise-model hooks](../architecture/hooks/noise-models.md) for the full restriction and executed evidence. No repository example configures `poisson` for a conventional engine.
[^smoke]: [Simulated single-slice reconstruction (gradient descent)](reconstructions/simulated-single-slice-gradient.md) is executed end to end (`phaser run`) against synthesized data and its result checked against the known synthetic object — the only reconstruction in this documentation effort verified this way, as opposed to validated-only or example-only.
[^b12]: Verified blocker [B12](../design/implementation-checklist.md#blocker-triage-phase-1-do-first): `tests/` covers utilities and data-loading/initialization only; no test exercises `phaser/engines/conventional/`, `phaser/engines/gradient/`, `phaser/engines/common/noise_models.py`, or `phaser/engines/common/position_correction.py`. See [Testing](../architecture/testing.md#example-demonstrated-versus-test-covered-b12).
[^pos-grad]: The gradient engine optimizes `positions` by assigning it an ordinary gradient solver (`sgd`/`adam`) in the `solvers` mapping — demonstrated in `examples/si_grad.yaml:46-48`, `si_grad_exp.yaml:46-48`, `prsco3_grad.yaml:41-43,93-95`, and `czo_grad.yaml:49-51,105-107`, each with `update_positions` enabled. Not affected by blockers B7/B8, which are specific to the conventional-engine position-correction code path (`phaser/engines/common/position_correction.py`).
[^tilt]: Tilt refinement is a gradient-solver assignment to the `tilt` variable and is exercised only by `examples/czo_grad.yaml`. `phaser/engines/conventional/run.py` never reads `update_tilt`, even though `ConventionalEnginePlan` inherits the field from `EnginePlan` — see [Engine families and backends](../architecture/overview.md#engine-families-and-backends) and verified blocker [B11](../design/implementation-checklist.md#blocker-triage-phase-1-do-first).
[^pos-review]: Verified blockers [B7 and B8](../design/implementation-checklist.md#blocker-triage-phase-1-do-first), detailed on [Solver hooks](../architecture/hooks/solvers.md#built-in-implementations) with executed evidence for B8. Position solving is documented here honestly, not as fully supported, pending code-owner disposition.
[^pos-lsqml]: `examples/mos2_lsqml.yaml:39-43,80-84`, `si_lsqml.yaml:43-47`, and `prsco3_lsqml.yaml:45-49` all configure `position_solver: {type: momentum, ...}` with `update_positions` enabled. `MomentumPositionSolver` correctly reads `max_step_size` (`phaser/engines/common/position_correction.py:38-41`) — only `SteepestDescentPositionSolver` has the confirmed B8 bug, and no LSQML example uses it.
[^pos-epie]: `examples/mos2_epie.yaml` sets neither `position_solver` nor `update_positions` (verified by search) — ePIE position refinement, unlike LSQML's, is not exercised by any repository example, on top of the B7 flag issue described in [Known limitations](#known-limitations).
[^b7]: Verified blocker [B7](../design/implementation-checklist.md#blocker-triage-phase-1-do-first): `EPIESolver.run_iteration` never passes `update_position` to `epie_run` (`phaser/engines/conventional/solvers.py:389-398`), unlike `LSQMLSolver`, which does (`solvers.py:118`) — `epie_run`'s `update_position` therefore always defaults to `True`, so the position-gradient step is always computed; it is still correctly gated from being *applied* by `update_positions` (`solvers.py:404-406`).
[^b8]: Verified blocker [B8](../design/implementation-checklist.md#blocker-triage-phase-1-do-first): `SteepestDescentPositionSolver.__init__` assigns `self.max_step_size = props.step_size` instead of `props.max_step_size` (`phaser/engines/common/position_correction.py:14-16`) — the per-step cap is always `step_size` itself. See [Solver hooks](../architecture/hooks/solvers.md#built-in-implementations) for executed evidence.
[^reg-cost]: `GradientEnginePlan.regularizers: t.List[CostRegularizerHook]` (`phaser/plan.py:154`); `ConventionalEnginePlan` declares no such field. Confirmed directly with `phaser validate`: a `regularizers:` key under a conventional engine fails with `Unexpected field`, the same key under a gradient engine validates — see [Cost-regularizer hooks](../architecture/hooks/cost-regularizers.md).
[^reg-group]: `group_constraints`/`iter_constraints` are declared, identically typed (`list` of `GroupConstraintHook`/`IterConstraintHook`), on both `ConventionalEnginePlan` and `GradientEnginePlan` (`phaser/plan.py`; see the [generated plan reference](../generated/plan/index.md)) and are applied on both engine families — see [Group-constraint hooks](../architecture/hooks/group-constraints.md) and [Iteration-constraint hooks](../architecture/hooks/iteration-constraints.md).
[^restart]: Restart reuses a prior state's probe/scan/tilt/object independent of which engine reconstructs next — see [Initialization merge semantics](../architecture/lifecycle.md#raw-data-loading-and-the-initialization-merge). Not itself engine-restricted; a plan may restart into a different engine than the one that produced the saved state.
[^b21]: Verified blocker [B21](../design/implementation-checklist.md#blocker-triage-phase-1-do-first): `props.shuffle_groups or not props.compact` (`phaser/engines/conventional/run.py:31`, `phaser/engines/gradient/run.py:179`, identical in both engines) treats an explicit `shuffle_groups: false` the same as unset whenever `compact` is also `false`. Confirm intent before recommending `shuffle_groups: false` in [grouping and memory](parameters/grouping-and-memory.md) guidance.

## Maintainer sources

- `phaser/plan.py`
- `phaser/execute.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/conventional/solvers.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/common/noise_models.py`
- `phaser/engines/common/position_correction.py`
- `phaser/hooks/solver.py`
- `phaser/hooks/regularization.py`
- `examples/`
- `docs/design/implementation-checklist.md`
