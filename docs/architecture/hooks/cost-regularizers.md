# Cost-regularizer hooks

A **cost regularizer** adds a differentiable term to the gradient engine's objective,
encoding a prior belief about the object, probe, or tilt — for example, penalizing
implausibly large object amplitude or a rough (high total-variation) reconstructed
phase. This is one of three [**regularizer**](../../concepts/glossary.md#regularizer)
lifecycles defined in `phaser/hooks/regularization.py`: this page covers cost
regularizers; the other two mutate state directly rather than contributing to an
objective — [Group constraints](group-constraints.md) (after every group),
[Iteration constraints](iteration-constraints.md) (after every iteration). All three share
several built-in implementations (e.g. `clamp_object_amplitude` is a group *and* iteration
constraint, never a cost regularizer) — read all three pages together when choosing which
lifecycle fits a new hook.

!!! warning "Restriction"
    Cost regularizers are **gradient-engine only**. `GradientEnginePlan` declares a
    `regularizers: t.List[CostRegularizerHook]` field; `ConventionalEnginePlan` declares no
    such field (`phaser/plan.py:141-156`). Confirmed directly with `phaser validate`: a
    conventional-engine plan with a `regularizers:` key under `engines[0]` fails with
    `Unexpected field '0.engines.0.regularizers'`, while the same key under a `gradient`
    engine validates successfully. Conventional engines (ePIE, LSQML) accept
    [group constraints](group-constraints.md) and
    [iteration constraints](iteration-constraints.md), which mutate state directly instead
    of contributing to a loss — there is no equivalent of a differentiable cost term for a
    conventional solver, since conventional solvers don't compute or use a gradient of a
    scalar objective.

!!! warning "Terminology"
    The schema field is `regularizers` (`GradientEnginePlan.regularizers`,
    `phaser/plan.py:154`), never `regularizations` — a plan using `regularizations:` fails
    `phaser validate` with `Missing required field '...regularizers'` and
    `Unexpected field '...regularizations'`. Use *regularizer*, not *regularization*, as
    the name of this hook family; "regularization" remains fine as the general scientific
    concept.

## Lifecycle point

A cost regularizer's loss is computed **once per group**, inside `run_model`
(`phaser/engines/gradient/run.py:421-475`) — the same JIT-traced function that computes
the detector loss from the [noise model](noise-models.md) — and *before* the gradient of
the total loss is taken with respect to the group's variables
(`tree.grad(run_model, ...)`, `phaser/engines/gradient/run.py:380-385`). Each regularizer's
loss is added directly to the detector loss to form `total_loss`
(`phaser/engines/gradient/run.py:466-473`):

```python
for (reg_i, reg) in enumerate(regularizers):
    (reg_loss, solver_states.regularizer_states[reg_i]) = reg.calc_loss_group(
        group, sim, solver_states.regularizer_states[reg_i]
    )
    losses[reg.name()] = reg_loss
    loss += reg_loss
```

Because this runs inside `run_group`, which is JIT-compiled when the backend is JAX
(`@partial(jit, ...)`, `phaser/engines/gradient/run.py:354-358`), a custom regularizer's
`calc_loss_group` must have no side effects and be traceable like any other
JAX-differentiated function — see [the JAX guide](../jax.md) once written, and the
[hook anatomy page](index.md#a-resolved-hook-is-a-plain-callable) for how this two-step
call pattern differs from a schedule or flag.

## Callable signature and property schema

`CostRegularizerHook` (`Hook[None, CostRegularizer]`, `phaser/hooks/regularization.py:113`)
is a **two-step** hook: the hook itself is called once, with `None`, to construct a
`CostRegularizer` instance; that instance's own methods are then called every group.

```python
@t.runtime_checkable
class CostRegularizer(HasState[StateT], t.Protocol[StateT]):
    def name(self) -> str:
        ...

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: 'ReconsState', state: StateT
    ) -> t.Tuple['Float', StateT]:
        ...
```

(`phaser/hooks/regularization.py:27-33`.) `HasState[StateT]` (`phaser/hooks/solver.py`)
additionally requires `init_state(self, sim: ReconsState) -> StateT`, called once when the
engine starts (`SolverStates.init_state`, `phaser/engines/gradient/run.py:135-150`) to
produce each regularizer's initial carried state — most built-ins carry no state at all
(`init_state` returns `None`).

Two property schemas cover every built-in:

| Property schema | Fields |
| --- | --- |
| `CostRegularizerProps` | `cost: float` |
| `TVRegularizerProps` | `cost: float`, `eps: float = 1.0e-8` |

(`phaser/hooks/regularization.py:104-110`.) `cost` is a **dimensionless weight**: it scales
the regularizer's own loss term relative to the detector loss the
[noise model](noise-models.md) produces. There is no fixed physical unit for `cost` in the
code or its docstrings — a claim that it is denominated "in electrons" could not be verified
against `phaser/hooks/regularization.py` or `phaser/engines/common/regularizers.py` and
isn't made here. What *is* verified: because the detector loss is only meaningful in
physical-count terms when patterns are scaled to counts, a fixed `cost` value's effective
strength shifts if pattern scaling changes
([Intensity and count scaling](../state-and-conventions.md#intensity-and-count-scaling)).

## Accepted state and returned value

`calc_loss_group(group, sim, state)` receives:

- `group` — the current group's scan-position indices, `NDArray[numpy.integer]`, shape
  `(group_size,)`;
- `sim` — the full `ReconsState`, with this group's per-group variables (`object`,
  `probe`) already substituted for the current gradient-descent step via `insert_vars`
  (`phaser/engines/gradient/run.py:436`) — per-iteration variables (`positions`, `tilt`)
  are present too, but only this group's slice of them is meaningful;
- `state` — the regularizer's own carried state from the previous call (or from
  `init_state` on the first call).

It returns `(loss, new_state)`: `loss` is a scalar (`Float`, the project's backend-generic
floating alias) added directly into the total loss; `new_state` replaces the regularizer's
carried state for the next call.

## Built-in implementations

Registered in `CostRegularizerHook.known` (`phaser/hooks/regularization.py:113-130`) and
implemented in `phaser/engines/common/regularizers.py`. Every built-in scales its raw
penalty by `group.shape[-1] / prod(sim.scan.shape[:-1])` — the fraction of all scan
positions in the current group — except the probe-only regularizers, which use a fixed
`cost_scale = 1.0` since the probe doesn't vary per scan position.

| Name | Penalizes | Notes |
| --- | --- | --- |
| `obj_l1` | `sum(abs(object - 1))` | L1 distance of object amplitude/phase from the vacuum value `1` |
| `obj_l2` | `sum(abs(object - 1)^2)` | L2 (squared) distance from vacuum |
| `obj_phase_l1` | `sum(abs(angle(object)))` | L1 norm of object phase |
| `obj_recip_l1` | `sum(abs(fft2(prod(object, axis=0))))` | L1 norm of the projected object's diffracted amplitude |
| `obj_tv` | isotropic total variation of the object | `sqrt(g_y^2 + g_x^2 + eps)` summed over pixels (`TVRegularizerProps.eps`, default `1e-8`, avoids a non-differentiable point at zero gradient) |
| `obj_tikh` / `obj_tikhonov` | squared finite differences of the object along `y` and `x` | two aliases for one implementation, `ObjTikhonov` |
| `layers_tv` | total variation of the object across the **slice** axis | `0` if the object has fewer than 2 slices (single-slice) |
| `layers_tikh` / `layers_tikhonov` | squared finite differences across the slice axis | `0` if fewer than 2 slices; two aliases, one implementation, `LayersTikhonov` |
| `probe_phase_tikh` / `probe_phase_tikhonov` | squared finite differences of the probe's Fourier-space **phase** | two aliases, one implementation, `ProbePhaseTikhonov` |
| `probe_recip_tv` | isotropic total variation of the probe's Fourier-space amplitude | uses `TVRegularizerProps` (has `eps`) |
| `probe_recip_tikh` / `probe_recip_tikhonov` | squared finite differences of the probe's Fourier-space amplitude | two aliases, one implementation, `ProbeRecipTikhonov` |

The generated reference for this hook family does not exist yet (Phase 2 of the
[implementation plan](../../design/implementation-plan.md)); this table was verified
directly against `phaser/hooks/regularization.py` and
`phaser/engines/common/regularizers.py`.

## Minimal custom implementation

A cost regularizer needs only `name`, `init_state`, and `calc_loss_group` — no base class
is required because the protocol is structural (`@t.runtime_checkable`). This example
penalizes the mean object amplitude's deviation from a target value:

```python
class ObjAmplitudeTarget:
    """Penalize deviation of the mean object amplitude from a target value."""
    def __init__(self, args, props):
        self.cost = props['cost']
        self.target = props['target']

    @staticmethod
    def name():
        return 'obj_amplitude_target'

    def init_state(self, sim):
        return None

    def calc_loss_group(self, group, sim, state):
        xp = numpy
        amp = xp.abs(sim.object.data)
        loss = self.cost * xp.sum((amp - self.target) ** 2)
        return (loss, state)
```

Run against a synthetic `ReconsState` (a `(1, 16, 16)` object, uniform amplitude `1.0`,
NumPy backend) with `cost=0.5`, `target=0.9`:

```text
custom cost regularizer loss: 1.2800006 name: obj_amplitude_target
```

which matches `0.5 * 16 * 16 * (1.0 - 0.9)**2 = 1.28` computed by hand, confirming the
implementation is called and returns the expected scalar.

## YAML invocation

Built-in short name (see the example plan `examples/si_grad.yaml`, which uses this exact
pattern):

```yaml
engines:
  - type: gradient
    # ...
    regularizers:
      - type: obj_l2
        cost: 0.4
      - type: obj_tikh
        cost: 0.2
```

External reference — properties are passed through as a plain dictionary and are **not**
schema-validated, so a typo in `target` below is only discovered when the regularizer
first runs:

```yaml
    regularizers:
      - type: "my_package.my_regularizers:ObjAmplitudeTarget"
        cost: 0.5
        target: 0.9
```

Verified with `phaser validate` on a plan combining a built-in cost regularizer with a
group and an iteration constraint under a `gradient` engine
([restriction admonition](#lifecycle-point) above).

## Engine and backend restrictions

**Gradient engine only** (restriction at the top of this page). Because the gradient
engine itself requires the `jax` or `torch` backend
([overview](../overview.md#engine-families-and-backends)), every cost regularizer runs
under one of those two backends in practice, and a custom implementation should use
backend-generic operations (`get_array_module(sim.object.data)`, as every built-in does)
rather than assuming NumPy specifically.

## Optional dependencies

None. No entry in `CostRegularizerHook.known` declares an optional-dependency tuple;
every built-in uses only `numpy`-generic array operations already available through the
active backend.

## Testing pattern

A cost regularizer can be tested without running any engine or backend: construct the
object directly (`reg = ObjAmplitudeTarget(None, {...})`), build a synthetic `ReconsState`
with the object/probe shapes you care about, call `reg.init_state(sim)` once, then
`reg.calc_loss_group(group, sim, state)` and assert on the returned scalar — the pattern
used to produce the output above, working identically whether `sim.object.data` is a
NumPy or JAX array, since the implementation should only use backend-generic operations.
`tests/` currently has no coverage of `phaser/hooks/regularization.py`
([implementation checklist](../../design/implementation-checklist.md), blocker B12) — a new
custom regularizer should add its own unit test in this style rather than rely on an
end-to-end engine run, which additionally requires a JAX or Torch installation.

## Maintainer sources

- `phaser/hooks/regularization.py`
- `phaser/engines/common/regularizers.py`
- `phaser/engines/gradient/run.py`
- `phaser/plan.py`
