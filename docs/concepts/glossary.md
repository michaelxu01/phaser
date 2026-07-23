# Glossary

This glossary gives one canonical definition for each term, verified against the current
implementation. Where a word is used differently by microscopists and by the Python
implementation, both meanings are given and distinguished explicitly. Every other page in
this documentation should link here at first use of a defined term rather than redefining
it locally.

Terms are listed alphabetically.

## backend

An array-computation library Phaser can run on: `numpy`, `cupy`, `jax`, or `torch`
(`BackendName` in `phaser/types.py`). A plan selects a backend with the top-level
`backend` field, or the `xp=` argument to the Python API; `phaser.utils.num.get_backend_module`
resolves the name to the corresponding module. The gradient-descent engine specifically
requires the `jax` or `torch` backend — `phaser/execute.py`'s `prepare_for_engine` raises
a `ValueError` if a `GradientEnginePlan` is run under `numpy` or `cupy`. The conventional
engines (ePIE, LSQML) are not backend-restricted in the same code path.

## constraint

A hook that mutates reconstruction state directly, on a fixed cadence, rather than
contributing a differentiable term to an objective (contrast **regularizer** below, with
which constraints share several implementations). Two families exist, both in
`phaser/hooks/regularization.py`: a **group constraint** (`GroupConstraintHook`, calling
`apply_group`) runs after every group; an **iteration constraint** (`IterConstraintHook`,
calling `apply_iter`) runs after every iteration. Both conventional and gradient engine
plans accept `group_constraints` and `iter_constraints` lists (`phaser/plan.py`).

## engine

**Scientific meaning:** one stage of a reconstruction — a specific algorithm (a
conventional solver such as ePIE or LSQML, or gradient descent) that repeatedly updates
the object, probe, and optionally positions or tilt from the diffraction data.

**Implementation meaning:** an `EngineHook` (`phaser/hooks/__init__.py`) resolved to a
run function — `phaser.engines.conventional.run:run_engine` or
`phaser.engines.gradient.run:run_engine` — and configured by a `ConventionalEnginePlan` or
`GradientEnginePlan` (`phaser/plan.py`). A plan's `engines` field is an ordered list;
`phaser.execute.execute_plan` runs them in sequence against one shared state. At each
engine boundary, `prepare_for_engine` (`phaser/execute.py`) may reshape that shared
state: resampling the probe and patterns to a new `sim_shape`, padding or resampling the
object, changing the probe mode count, reslicing the object into a different number of
slices, and creating a zeroed tilt map the first time a gradient solver targets `tilt`.

## flag

A boolean-valued hook controlling whether something happens on a given iteration:
either a plain `bool`, a `SimpleFlag(after=, every=, before=)` (`phaser/types.py`), or a
`FlagHook` (`phaser/hooks/schedule.py`). Contrast **schedule**, which produces a number
instead of a boolean.

## group

A batch of scan positions processed together within one engine iteration. The `grouping`
field of `EnginePlan` sets the (target) group size; `compact` and `shuffle_groups`
control how positions are assigned to groups (`phaser/engines/common/simulation.py` uses
`create_compact_groupings`/`create_sparse_groupings`). `buffer_n_groups` controls how many
groups' patterns are held on the compute device at once — `0` transfers each group
synchronously, a positive integer prefetches that many groups ahead, and `None` (`~` in
YAML) loads the entire dataset onto the device up front (`phaser/plan.py`). A **group
constraint** (see **constraint**) runs once per group.

## hook

A configuration-driven construction mechanism: a `Hook[T, U]` (`phaser/hooks/hook.py`)
that resolves either to a short, registered name (looked up in that hook family's `known`
registry) or to an external `"package.module:function"` reference, and is called with
properties parsed from the plan's YAML/JSON. Hooks are how plans stay declarative:
raw-data loading, initialization, post-load and post-init processing, noise models,
solvers, regularizers, constraints, and engines are all hook families. A hook is
distinct from a **solver** or **constraint** object (which carries state across calls
inside a running engine) and from an **observer** (which reacts to lifecycle events and
is not part of the plan schema at all).

## iteration

**Scientific meaning:** one pass of a reconstruction engine over the data (or, with
grouping, over all groups), after which the object/probe/other variables have been
updated once.

**Implementation meaning:** tracked by `IterState` (`phaser/state.py`), whose
`engine_iter` and `total_iter` fields are 1-indexed with `0` meaning "before any
iterations have run." `EnginePlan.niter` sets how many iterations an engine performs.

## JIT

Just-in-time compilation. `phaser.utils.num.jit` wraps a function so that, when called
with JAX arrays, it is compiled via `jax.jit`; called with other array types, it runs the
plain Python function directly (`phaser/utils/num.py`, `_JitKernel.__call__`). This is
distinct from `jit_unroll_slices` (an `EnginePlan` field), which controls how many
multislice propagation steps are unrolled into the JIT-traced computation graph rather
than executed as a traced Python loop, and only affects the JAX backend.

## mixed-state

A partially coherent probe modeled as an incoherent sum of several orthogonal modes
(`EnginePlan.probe_modes > 1`), each carrying a fraction of the total probe intensity set
by `base_mode_power`. Implemented as `ProbeState.data` having shape `(modes, y, x)`;
`phaser.utils.optics.make_hermetian_modes` creates additional modes when `probe_modes`
increases at an engine boundary (`phaser/execute.py`, `prepare_for_engine`).

## mode

Short for a probe mode: one of the orthogonal components of a (possibly mixed-state)
probe, indexed along the leading axis of `ProbeState.data`. See **mixed-state**.

## noise model

**Scientific meaning:** the statistical model relating a measured diffraction pattern to
the simulated model intensity, used to compute both a scalar loss and a wave-domain
update.

**Implementation meaning:** a `NoiseModelHook` resolving to an object implementing
`calc_loss` and `calc_wave_update` (`phaser/hooks/solver.py`); the registered options are
`amplitude`, `anscombe`, and `poisson` (`phaser/plan.py`). Both `ConventionalEnginePlan`
and `GradientEnginePlan` declare `noise_model` as a single hook, not a list — each engine
uses exactly one noise model.

## object

**Scientific meaning:** the specimen's reconstructed complex transmission function; its
amplitude corresponds to absorption and its phase is approximately proportional to
projected potential.

**Implementation meaning:** `ObjectState` (`phaser/state.py`); `data` has shape
`(z, y, x)`, and `thicknesses` gives each slice's physical thickness along the beam
direction (length `< 2` for a single-slice object). Constructed by an `ObjectHook`
(for example `'random'`, `phaser/hooks/__init__.py`).

## observer

An object implementing the `Observer` interface (`phaser/observer.py`) that reacts to
reconstruction lifecycle events — initialization, engine start, per-group and
per-iteration updates, engine/reconstruction finish, and close — typically for logging,
checkpointing, or forwarding state elsewhere. Observers are supplied as Python objects to
`execute_plan`/`initialize_reconstruction` via the `observers=` argument (appended to the
built-in defaults, `phaser/execute.py`'s `_normalize_observers`) or `override_observers=`
(replaces the defaults entirely); they are not part of the plan's YAML schema, unlike
hooks.

## plan

A validated `ReconsPlan` (`phaser/plan.py`) describing one reconstruction end to end: how
to load raw data, optional post-load processing, initialization (`init`), optional
post-init processing, and an ordered list of engines. Plans are parsed and validated from
YAML/JSON by `pane`. Because expression schedules use unrestricted `eval` and external
hooks execute arbitrary importable code, running a plan is equivalent to running a
script — see the trust model described in the [authoring guide](../design/authoring-guide.md#trust-model-and-warning-conventions).

## probe

**Scientific meaning:** the illumination wavefunction incident on the specimen at each
scan position; may be a single coherent mode or a mixed state (see **mixed-state**).

**Implementation meaning:** `ProbeState` (`phaser/state.py`); `data` has shape
`(modes, y, x)` in real space, paired with a `Sampling` describing its coordinate system.
Constructed by a `ProbeHook` (for example `'focused'`, `phaser/hooks/__init__.py`).

## pytree

A nested container structure that JAX (and, separately, PyTorch) can flatten into leaf
arrays and reassemble, so automatic differentiation and JIT compilation can operate
through ordinary Python objects. Phaser's `@tree_dataclass` decorator
(`phaser/utils/tree.py`) registers a dataclass as a pytree for whichever of JAX and
PyTorch are loaded (`jax.tree_util.register_pytree_with_keys`,
`torch.utils._pytree.register_pytree_node`), separating declared `static_fields` (kept as
hashable metadata rather than traced, such as a `Sampling`) from the remaining fields
(treated as leaves).

## registry

The `known` class-level dictionary on each `Hook` subclass (`phaser/hooks/hook.py`),
mapping a short YAML type name (for example `'empad'`, `'adam'`, `'obj_tv'`) to an
importable function reference and its properties dataclass. Built-in hooks are
registered in their defining module or, for engines and solvers, in `phaser/plan.py`.

## regularizer

The schema term — **not** "regularization" — for a hook that encodes a prior belief
about the object, probe, or tilt. Three distinct hook families share this vocabulary but
differ in lifecycle, all defined in `phaser/hooks/regularization.py`:

- a **cost regularizer** (`CostRegularizerHook`) adds a differentiable term to the
  gradient engine's objective (for example `obj_l1`, `obj_tv`);
- a **group constraint** (`GroupConstraintHook`) mutates state after every group;
- an **iteration constraint** (`IterConstraintHook`) mutates state after every iteration.

`GradientEnginePlan` declares a `regularizers` field; `ConventionalEnginePlan` does not —
conventional engines accept group and iteration constraints but not cost regularizers
(`phaser/plan.py`).

## schedule

A numeric value that can vary across an engine's iterations: a plain float, or a
`ScheduleHook` resolving to `constant`, `piecewise`, or `expr`
(`phaser/hooks/schedule.py`). The `expr` schedule evaluates its `expr` string with
Python's `eval`, given `i`, `iter`, `state`, `niter`, and `np` — this executes arbitrary
Python and must always carry the trust warning described in the
[authoring guide](../design/authoring-guide.md#trust-model-and-warning-conventions).

## slice / multislice

**Scientific meaning:** dividing the specimen along the beam direction into a stack of
thin transmission functions connected by free-space propagation between them
(multislice), as opposed to representing the whole specimen with one 2D transmission
function (single slice).

**Implementation meaning:** `ObjectState.data` always carries a leading slice axis
`(z, y, x)`; a single-slice object simply has fewer than two `thicknesses`
(`ObjectState.zs` in `phaser/state.py`). The plan-level `slices` field (`SliceList`,
`SliceStep`, or `SliceTotal` in `phaser/types.py`) sets slice thicknesses; changing slice
count between engines triggers reslicing in `prepare_for_engine`
(`phaser/execute.py`).

## solver

An implementation-only term covering three distinct, stateful protocols
(`phaser/hooks/solver.py`):

- a **conventional solver** (`ConventionalSolver`, e.g. ePIE, LSQML) drives one full
  engine iteration, including its own presolve and per-iteration logic;
- a **gradient solver** (`GradientSolver`, e.g. SGD, Adam, Polyak-SGD) is a per-parameter
  optimizer applied to a declared, disjoint set of `ReconsVar`s — gradient solvers
  separate per-group variables (`object`, `probe`) from per-iteration variables
  (`positions`, `tilt`) and are rejected if a single solver's variable set mixes the two;
- a **position solver** (`PositionSolver`, e.g. `steepest_descent`, `momentum`) updates
  scan positions from position gradients.

All three carry algorithm state across calls (an `init_state`/state-update pattern),
which distinguishes them from stateless hooks such as loaders or post-processing steps.

## state

**Scientific meaning:** the physical quantities being reconstructed together with their
bookkeeping — object, probe, scan positions, optional tilt, iteration counters, and error
history.

**Implementation meaning:** `ReconsState` (`phaser/state.py`), whose fields are all
required, versus `PartialReconsState`, whose fields are all optional (used while merging
a restart file or metadata before defaults are filled in via `to_complete()`). The unit
an engine actually receives is `PreparedRecons` (patterns, state, name, observer). State
can be written to and read from HDF5 — see **serialization**.

## serialization

Writing a `ReconsState` or `PartialReconsState` to, and reading it from, an HDF5 file via
`write_hdf5`/`read_hdf5` (`phaser/state.py`, implemented in `phaser/utils/io.py`). Written
files carry a `type` marker (`'phaser_state'`) and a `version` marker, checked on read.

## Maintainer sources

- `phaser/state.py`
- `phaser/plan.py`
- `phaser/execute.py`
- `phaser/observer.py`
- `phaser/types.py`
- `phaser/hooks/hook.py`
- `phaser/hooks/__init__.py`
- `phaser/hooks/schedule.py`
- `phaser/hooks/solver.py`
- `phaser/hooks/regularization.py`
- `phaser/utils/num.py`
- `phaser/utils/tree.py`
- `phaser/utils/io.py`
