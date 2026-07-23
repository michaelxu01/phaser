# JAX implementation guide

This page is for contributors writing or reviewing backend-compatible code — hooks,
solvers, constraints, or shared kernels — that must run correctly whether the active
[backend](../concepts/glossary.md#backend) is NumPy, CuPy, JAX, or PyTorch. It assumes the
[overview](overview.md) and [reconstruction lifecycle](lifecycle.md) and does not repeat
plan-level configuration already documented there.

!!! warning "Restriction"
    JAX acceleration is not a separate engine. Both engine families —
    [conventional](hooks/engines.md) (ePIE, LSQML) and gradient descent — call the same
    shared simulation kernels in `phaser/engines/common/simulation.py`, and both wrap
    their inner per-group kernels with the same `phaser.utils.num.jit` helper
    (`phaser/engines/conventional/solvers.py:146,186,417,446` and
    `phaser/engines/gradient/run.py:354,416`). What differs between backends is what that
    shared code *does* when traced or differentiated, described below — the gradient
    engine additionally requires JAX or Torch specifically because it needs automatic
    differentiation, which NumPy and CuPy do not provide
    (`phaser/execute.py:386-387`; see [Engine families and backends](overview.md#engine-families-and-backends)).

## The `xp` compatibility layer

Phaser code is written against `xp`, a stand-in for whichever array module is active, so
the same function can run on NumPy, CuPy, JAX, or PyTorch arrays. `phaser/utils/num.py`
implements this layer:

- **Loading a backend.** `_BackendLoader` (`phaser/utils/num.py:43-129`) lazily imports
  each backend's module the first time it is requested, and caches the result — importing
  `phaser.utils.num` does not eagerly import JAX, Torch, or CuPy. `get_backend_module`
  (line 132) returns the `xp` module for a given `BackendName` (`'numpy'`, `'cupy'`,
  `'jax'`, or `'torch'`, `phaser/types.py:78`), raising if that backend is not installed.
- **Auto-preference.** `get_default_backend()` (`phaser/utils/num.py:159-181`) is used
  when neither the Python API's `xp=` argument nor the plan's `backend` field selects one
  (`phaser/execute.py`, `initialize_reconstruction`). In order: it first checks whether
  JAX is installed and reports a GPU or TPU device, returning `'jax'` if so; then whether
  Torch is installed and its default device is not `'cpu'`, returning `'torch'` if so;
  otherwise it returns the first of `'jax'`, `'torch'`, `'cupy'` that is importable, and
  falls back to `'numpy'` if none are. In effect: prefer a GPU/TPU-backed JAX or Torch
  installation, then prefer JAX, Torch, or CuPy on CPU over plain NumPy, and use NumPy
  only if nothing else is available.
- **Dispatch by argument inspection**, not by a global setting. `get_array_module(*arrs)`
  (`phaser/utils/num.py:268-286`) inspects the actual arrays passed to it — including
  arrays nested inside a [pytree](../concepts/glossary.md#pytree) — and returns the
  corresponding module; `is_jax`/`is_torch`/`is_cupy` and `xp_is_jax`/`xp_is_torch`/`xp_is_cupy`
  (lines 365-397) make the same check against a value or a module object respectively.
  Shared kernel code calls `get_array_module` on its inputs rather than importing a fixed
  backend, which is what lets the same function body run under any backend.
- **Backend-specific fallbacks stay local.** Functions like `fft2`/`ifft2`, `pad`, and
  `abs2` (`phaser/utils/num.py`) branch internally on `xp_is_torch`/`is_jax` only where the
  underlying libraries genuinely disagree (for example, Torch's FFT functions take a
  `dim=` keyword where NumPy/JAX/CuPy take `axes=`) — most Phaser code never needs to
  check the backend itself, because `xp.<function>` already means the right thing.

## Pytrees and `@tree_dataclass`

A [pytree](../concepts/glossary.md#pytree) is a nested container JAX (and, separately,
PyTorch) can flatten into leaf arrays and reassemble, which is what lets `jax.jit` and
`jax.grad` (and their Torch equivalents) operate through ordinary Python objects instead
of only raw arrays. Phaser's `@tree_dataclass` decorator (`phaser/utils/tree.py:379-442`)
registers a dataclass as a pytree with whichever of JAX and PyTorch are actually loaded —
registration is deferred via `_BackendLoader._schedule_on_load`, so a dataclass decorated
before either library is imported still gets registered once one is.

`tree_dataclass` splits a class's fields into three kinds:

- **Leaves** (the default): traced, differentiated, and transformed like any array.
- **`static_fields`**: kept as hashable metadata attached to the tree structure rather
  than treated as a leaf — used for values that must not be traced (they change what code
  runs, not what it computes on), such as a coordinate system.
- **`drop_fields`**: excluded from the tree entirely — neither a leaf nor metadata. On
  `unflatten`, a dropped field is simply absent from the constructor call
  (`phaser/utils/tree.py:413-418`), so it silently reverts to that field's dataclass
  default if it has one.

Verified registrations, most relevant to backend-compatible code:

| Class | Location | `static_fields` | `drop_fields` |
| --- | --- | --- | --- |
| `Sampling` | `phaser/utils/num.py:821` | (none — `shape`/`sampling` are leaves) | `extent` |
| `Patterns`, `IterState`, `ProgressState` | `phaser/state.py:17,30,129` | none | none |
| `ProbeState`, `ObjectState` | `phaser/state.py:60,96` | `sampling` | none |
| `ReconsState` | `phaser/state.py:141` | none | `progress` |
| `PartialReconsState` | `phaser/state.py:190` | `progress` | none |
| `PreparedRecons` | `phaser/state.py:239` | `name`, `observer` | none |
| `SimulationState` | `phaser/engines/common/simulation.py:89` | `xp`, `dtype`, `noise_model`, `group_constraints`, `iter_constraints` | `ky`, `kx` |
| `SolverStates` (gradient engine) | `phaser/engines/gradient/run.py:128` | none | none |

!!! warning "Restriction"
    `ReconsState` **drops** `progress` from its pytree structure — it is neither traced
    nor kept as metadata. `phaser.engines.gradient.run.run_engine` has to work around this
    explicitly: its own comment states "progress gets clobbered by the jits, so we keep
    track of it manually" (`phaser/engines/gradient/run.py:247-248`), and the function
    keeps a separate `progress` dictionary outside the JIT-traced calls, reattaching it to
    `state.progress` after each iteration (line 347) rather than trusting it to survive a
    round trip through `run_group`/`run_model`. Any new code that passes a `ReconsState`
    through a JIT boundary and expects `progress` to come back populated will be wrong for
    the same reason — this is a correctness detail specific to how `drop_fields` behaves,
    not a bug to fix locally.

`Sampling.__eq__` (`phaser/utils/num.py:830-837`) is overridden to compare `shape`/`extent`
by value; this matters because `ProbeState`/`ObjectState` keep `sampling` as **static**
pytree metadata, and JAX/Torch use structural equality (and, for JAX, hashing) on static
metadata to decide whether a JIT-compiled function can be reused or must retrace.

## JIT boundaries in the gradient engine

`phaser.utils.num.jit` (`phaser/utils/num.py:472-487`, implemented by `_JitKernel`) wraps a
function so that:

- if it is called with any JAX array argument, it dispatches to a `jax.jit`-compiled
  version (compiling on first call for a given combination of shapes and static argument
  values, then reusing that compilation);
- called with any other argument (NumPy, CuPy, or Torch arrays), it calls the plain Python
  function directly — no compilation happens at all in that case.

In the gradient engine (`phaser/engines/gradient/run.py`), the per-group step is split
into two JIT-wrapped functions:

- **`run_group`** (line 354, `static_argnames=('vars', 'xp', 'dtype', 'noise_model',
  'group_solvers', 'group_constraints', 'regularizers', 'jit_unroll_slices')`,
  `donate_argnames=('state', 'iter_grads', 'solver_states')`) computes the gradient of
  `run_model` with respect to the extracted per-group variables (`tree.grad`, itself
  dispatching to `jax.grad` or `torch.func.grad` depending on `xp` —
  `phaser/utils/tree.py:157-193`), then applies each group solver's update.
- **`run_model`** (line 416, same static-argument pattern for `xp`/`dtype`/`noise_model`/
  `regularizers`/`jit_unroll_slices`) runs the forward simulation for one group — cutting
  out the probe/object at each scan position, propagating through slices, and computing
  the noise-model loss plus any cost regularizers — and is the function actually
  differentiated.

`static_argnames` marks arguments JAX must treat as compile-time constants: a Python
object such as the resolved `noise_model`, the tuple of `group_solvers`, or `jit_unroll_slices`
cannot be a traced array, so it becomes part of the cache key instead. **A JIT boundary
retraces (recompiles) whenever a static argument's value changes, or whenever a traced
argument's pytree structure or leaf shape/dtype changes** — this is the practical
recompilation trigger to watch for:

- **Shape changes at engine boundaries.** `prepare_for_engine` (see
  [Engine-boundary reshaping](lifecycle.md#engine-boundary-reshaping)) can resize the
  probe, resample or pad the object, change probe mode count, or reslice the object
  between engines — any of these changes a leaf's shape, so the first group of a new
  engine stage retraces even if the same solver ran in a previous stage.
  `jit_unroll_slices` and the slice count together also affect how many operations the
  traced multislice loop contains (see below), so a change in slice count can affect
  compile time as well as trigger a retrace.
- **Grouping changes.** The last group in an iteration is frequently a different size
  than the others (`scan.shape[:-1]` need not be a multiple of `grouping`,
  `phaser/engines/common/simulation.py:23-56`, `GroupManager`), so a plan whose position
  count doesn't divide evenly by `grouping` retraces at least once per iteration purely
  from that final group's shape — a case to watch for when tuning `grouping` for
  wall-clock performance.
- **Static-argument identity changes**, not just value — since `group_solvers`,
  `group_constraints`, and `regularizers` are passed as `static_argnames`, JAX hashes them
  as part of the cache key; a solver, constraint, or regularizer object must be
  re-`==`/re-hash stable across calls within one engine run, or every group would
  retrace.

Outside these two JIT-wrapped functions, `run_engine` itself (the per-iteration driver)
runs as plain Python — schedule evaluation, progress bookkeeping, and solver/constraint
construction happen outside any trace, which is deliberate: the code comment at
`phaser/engines/gradient/run.py:267-268` notes that "this needs to be done outside the
JIT context, which makes this kinda hacky" for updating solver schedules once per
iteration. `dry_run` (line 479, decorated separately) is a third, smaller JIT-wrapped
kernel used once per group before the main loop, to estimate a probe-intensity rescaling
factor.

## Multislice traversal: `jax.lax.scan`/`fori_loop` and `jit_unroll_slices`

The multislice forward and backward passes are shared kernels
(`phaser/engines/common/simulation.py:266-315`, `slice_forwards`/`slice_backwards`), used
by both engine families through `SimulationState`-based conventional solvers and the
gradient engine's `run_model`. Each takes a per-slice step function `f` and a stack of
propagators, and picks its traversal strategy by inspecting the propagators array:

- **If `is_jax(props)`** (line 277), `slice_forwards` uses `jax.lax.scan` and
  `slice_backwards` uses `jax.lax.fori_loop`, both passing `unroll=jit_unroll_slices`.
  This is what lets the traced computation graph represent an arbitrary number of object
  slices without JAX unrolling every one of them into the trace by default.
- **Otherwise** (NumPy, CuPy, or **Torch** — the check is `is_jax`, not "is this backend
  traced"), both functions fall back to a plain Python `for` loop over slices
  (`phaser/engines/common/simulation.py:286-289,311-314`).

!!! warning "Restriction"
    Under the **Torch** backend, multislice traversal always uses a plain unrolled Python
    loop, never `jax.lax.scan`/`fori_loop` — the `is_jax(props)` check in
    `slice_forwards`/`slice_backwards` is what selects the scan/loop implementation, and
    it is JAX-specific. This is consistent with the rest of the Torch path: no call site
    in `phaser/` uses `torch.compile` or `torch.jit` (verified by search), and
    `phaser.utils.num.jit`'s `_JitKernel.__call__` only ever dispatches to `jax.jit` — for
    Torch arguments it always calls the wrapped Python function directly, uncompiled. Torch
    autodiff goes through `torch.func.grad`/`grad_and_value` (`phaser/utils/tree.py:170-172,220-221`),
    which does not require or use `jit_unroll_slices` at all; the field's docstring itself
    scopes it to "JAX backend only" (`phaser/plan.py:67-74`). In short: **the gradient
    engine's JAX path is JIT-compiled with an explicit unrolling tradeoff for the
    multislice loop; its Torch path is fully eager, always effectively "unrolled" as an
    ordinary Python loop, and has no equivalent tuning knob.** Whether eager per-slice
    execution is competitive with a compiled JAX trace was not measured for this page and
    is not asserted either way.

`jit_unroll_slices` (`EnginePlan.jit_unroll_slices`, `phaser/plan.py:67-74`) controls the
`unroll` argument passed to `jax.lax.scan`/`fori_loop`, per its docstring: "`True` or `0`
unrolls all slices, `False` or `1` disables unrolling," and "larger unrolling may be
faster, at the expense of increased compilation time." The gradient engine defaults it to
`10` when the plan leaves it `None` (`phaser/engines/gradient/run.py:162`,
`jit_unroll_slices = 10 if props.jit_unroll_slices is None else props.jit_unroll_slices`);
other engines/solvers may resolve the `None` default differently, and this page does not
assert a single project-wide default beyond the gradient engine's. The tradeoff: more
unrolling produces a larger traced graph (slower to compile, faster or more
memory-hungry to run, depending on the object's slice count), while less unrolling keeps
compile time and traced-graph size small at the cost of loop overhead per slice at
runtime — under JAX only; this knob has no effect at all under Torch, NumPy, or CuPy.

## `buffer_n_groups`: tri-state device-transfer semantics

`EnginePlan.buffer_n_groups` (`phaser/plan.py:60-65`, default `2`) controls how many
groups' worth of diffraction patterns are transferred to the compute device at once. It
is read identically by both engine families
(`phaser/engines/gradient/run.py:195-207`, `phaser/engines/conventional/run.py:44-49`,
using the shared `stream_patterns` helper, `phaser/engines/common/simulation.py:58-86`)
and has three distinct states, not just "on" and "off":

- **`0` — synchronous, no prefetch.** `stream_patterns`'s `buf_n == 0` branch
  (`phaser/engines/common/simulation.py:62-66`) transfers one group's patterns to the
  device and blocks on `block_until_ready` before yielding it — the next group's transfer
  does not start until the current one has been consumed. This uses the least additional
  device memory but does not overlap host-to-device transfer with compute.
- **A positive integer `N` — prefetch `N` groups ahead.** The general branch
  (lines 68-86) keeps a bounded queue (`collections.deque`) of up to `N` in-flight
  transfers, feeding a new one in as soon as the oldest is consumed — so pattern transfer
  for group *k + N* can overlap with compute on group *k*. This is the default
  (`buffer_n_groups: 2`).
- **`None` (`~` in YAML) — load everything onto the device up front.** Both engines check
  `props.buffer_n_groups is None` before the main loop and, if so, transfer the entire
  patterns array to the device once (`xp.asarray(args['data'].patterns)`) instead of
  calling `stream_patterns` at all; every group is then indexed directly out of an
  already-resident device array. This uses the most device memory (the whole dataset) but
  removes per-group transfer overhead entirely.

Choosing among these is a memory/throughput tradeoff, not a correctness one — see
[Grouping and memory](../cookbook/parameters/grouping-and-memory.md) for guidance tied to
dataset size and device memory once that page is written; this page only documents what
each value does mechanically.

## Writing JAX-compatible custom hooks

A custom hook, solver, constraint, or regularizer that must work under the gradient
engine's JAX path (and, more restrictively, under `jax.jit` at all — noise models, group
solvers, group constraints, and regularizers are all called from inside `run_group`/
`run_model`, see [JIT boundaries](#jit-boundaries-in-the-gradient-engine) above) needs to
follow the constraints JAX itself imposes on traced code:

- **Be a pure function of its traced arguments.** No Python-level side effects (mutating a
  module-level list, writing a file, printing based on array *values*) may happen inside a
  function that gets traced — JAX traces the function once per distinct shape/dtype/static
  combination and replays the compiled version afterward, so a side effect written this
  way runs once at trace time, not once per call. Logging based on static configuration
  (not on traced array values) is fine; `phaser.utils.num.debug_callback` exists
  specifically for the case where you do need to observe traced array values from Python
  during a trace (it wraps `jax.debug.callback`, falling back to calling the callback
  directly when JAX is unavailable, `phaser/utils/num.py:499-504`).
- **Keep shapes and dtypes stable across calls that must reuse one compilation.** A
  Python-level `if` on a traced array's *value* (rather than on a static argument or plain
  Python data) will fail to trace; branch on shape, dtype, or explicitly-static
  configuration instead, or use `xp.where` for a value-dependent choice inside traced code.
- **Return new state rather than mutating in place** where the code you are extending
  already threads state functionally — the group/iteration constraint and solver
  protocols (`phaser/hooks/solver.py`, `phaser/hooks/regularization.py`) pass a state
  object in and expect a (possibly new) state object back out, which is what lets them be
  donated arguments (`donate_argnames`) and pass through JIT cleanly.
- **Register any new stateful container as a pytree** with `@tree_dataclass`
  (`phaser/utils/tree.py`) if it needs to be passed through a JIT boundary — an ordinary
  (non-pytree) Python object holding arrays will not flatten correctly and will either
  raise or be silently treated as static (and therefore never updated by JIT-traced code).
- **Remember this code path must also work without JAX.** Because the same kernels run
  under NumPy/CuPy/Torch too (see [`xp` compatibility layer](#the-xp-compatibility-layer)
  above), write against `xp`/`get_array_module` rather than importing `jax.numpy`
  directly, unless the hook is explicitly documented as JAX-only.

## Maintainer sources

- `phaser/utils/num.py`
- `phaser/utils/tree.py`
- `phaser/state.py`
- `phaser/engines/common/simulation.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/conventional/solvers.py`
- `phaser/execute.py`
- `phaser/plan.py`
