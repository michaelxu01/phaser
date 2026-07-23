# Observers

An [**observer**](../concepts/glossary.md#observer) is an object implementing the
`Observer` interface (`phaser/observer.py`) that reacts to reconstruction lifecycle
events — logging progress, checkpointing state and images to disk, forwarding state to a
remote server, or stopping a run early. Observers are supplied in Python to
`execute_plan`/`initialize_reconstruction`, not in a plan's YAML/JSON — see the
[overview's extension-mechanism table](overview.md#extension-mechanisms) for how this
distinguishes observers from every other extension surface, including hooks.

## Observers are not hooks

A [hook](../concepts/glossary.md#hook) is resolved from a plan field and *constructs* one
piece of behavior (a loader, an initializer, a noise model); a
[solver](../concepts/glossary.md#solver), regularizer, or constraint carries algorithm
state across calls inside one running engine. An observer does neither — it carries no
plan-schema representation at all, and it does not construct or mutate reconstruction
behavior; it only reacts to state that already changed. Passing an observer therefore
never affects reconstruction results (loss, object, probe, positions), only what is
logged, saved, or transmitted alongside them. See the
[extension-mechanism table](overview.md#extension-mechanisms) for the full comparison.

## The `Observer` interface

`Observer` (`phaser/observer.py`) is a base class whose methods all default to no-ops, so
a custom observer overrides only the events it cares about:

```python
class Observer(contextlib.AbstractContextManager):
    def init_recons(self, plan: ReconsPlan): ...
    def start_recons(self, init_state: ReconsState): ...
    def init_engine(self, init_state: ReconsState, *, recons_name: str, plan: EnginePlan, **kwargs: t.Any): ...
    def start_engine(self, init_state: ReconsState): ...
    def heartbeat(self): ...
    def update_group(self, state: t.Union[ReconsState, PartialReconsState], force: bool = False): ...
    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]): ...
    def finish_engine(self, state: ReconsState): ...
    def finish_recons(self, state: ReconsState): ...
    def close(self, exc: t.Optional[BaseException] = None): ...
```

`Observer` also implements the context-manager protocol: `__enter__` returns `self`, and
`__exit__` calls `close(exc)` — but nothing in `phaser/execute.py` currently uses an
observer as a `with`-block; `close` is called directly (see below).

### Event lifecycle and call order

The table below gives the verified call order and origin of every event, cross-referenced
to the [reconstruction lifecycle](lifecycle.md) page it belongs to.

| Order | Method | Called from | When | Frequency |
| --- | --- | --- | --- | --- |
| 1 | `init_recons(plan)` | `initialize_reconstruction` (`phaser/execute.py`) | Right after the observer set is built, before any data loads | Once per `execute_plan`/`initialize_reconstruction` call |
| 2 | `start_recons(init_state)` | `initialize_reconstruction` | After the initial `ReconsState` is fully built (probe, scan, tilt, object, `post_init` hooks all run) | Once |
| 3 | `init_engine(init_state, *, recons_name, plan, **kwargs)` | `run_engine` in `phaser/engines/conventional/run.py` or `phaser/engines/gradient/run.py` | Inside a given engine's run function, after `prepare_for_engine` has already reshaped state, before that engine's own presolve/rescaling | Once per engine in `plan.engines` |
| 4 | `start_engine(init_state)` | Same `run_engine` functions | After presolve/rescaling, immediately before the first iteration | Once per engine |
| 5 | `update_group(state, force=False)` | `EPIESolver`/`LSQMLSolver.run_iteration` (`phaser/engines/conventional/solvers.py`) or the gradient engine's per-group loop (`phaser/engines/gradient/run.py`) | After each group within an iteration finishes | Many times per iteration (once per group) |
| 6 | `update_iteration(state, i, n, errors)` | Same `run_engine` functions | After every group in an iteration has run and losses are aggregated | Once per iteration, per engine |
| 7 | `finish_engine(state)` | Same `run_engine` functions | After the last iteration of that engine | Once per engine |
| 8 | `finish_recons(state)` | `execute_plan` (`phaser/execute.py`) | After every engine in `plan.engines` has run (or the reconstruction terminated early) | Once |
| 9 | `close(exc)` | `execute_plan`, in a `finally` block | Always last, whether the reconstruction succeeded or raised; `exc` carries the raised exception, if any | Once |

`heartbeat()` is declared on `Observer` (docstring: "Called reasonably often by the
engine, to e.g. periodically send data") and forwarded by `ObserverSet`, and the built-in
`WorkerObserver` (`phaser/web/worker.py`) overrides it — but a repository-wide search finds
no call site at all: nothing under `phaser/engines/`, `phaser/execute.py`, or
`phaser/web/` actually invokes `observer.heartbeat()`. Documentation must not claim it
fires periodically during a run or during a worker's poll loop; a custom observer that
wants that behavior currently has to call `heartbeat()` itself from code it owns.

Two further call-order details:

- **`update_group` versus `update_iteration` frequency.** Within one iteration, `update_group`
  fires once per group and `update_iteration` fires exactly once, after all of that
  iteration's groups have run — so a custom observer that wants a per-iteration summary
  should implement `update_iteration`, not try to aggregate calls to `update_group` itself
  unless it specifically wants per-group granularity.
- **Early termination still reaches `finish_engine`/`finish_recons`/`close`.** If an
  `EarlyTermination` exception is raised (for example, by the built-in `PatienceObserver`,
  see below), `execute_engine` still calls `finish_engine` before re-raising or continuing,
  and `execute_plan` still calls `finish_recons` and `close` — see
  [Sequential engines](lifecycle.md#sequential-engines) for the full exception path.

## Constructing the observer set: append versus override

`_normalize_observers` (`phaser/execute.py:97-125`) turns the `observers=` and
`override_observers=` arguments of `execute_plan`/`initialize_reconstruction` into the
single `ObserverSet` actually used. Exactly one of the two arguments may be passed —
passing both raises `TypeError`.

- **`observers=` appends.** The default two observers, in this order —

    ```python
    obs = [
        SaveObserver(),
        LoggingObserver(),
    ]
    ```

  — are always constructed first; whatever is passed to `observers=` (a single `Observer`
  or an iterable of them) is appended after them. So `execute_plan(plan, observers=my_obs)`
  always keeps saving and logging active, with `my_obs` receiving every event in addition.
- **`override_observers=` replaces.** The observer set becomes exactly what is passed —
  the built-in `SaveObserver`/`LoggingObserver` defaults are **not** included. A plan run
  with `override_observers=my_obs` and nothing else will not write state or images to disk
  and will not log progress, unless `my_obs` does so itself.
- **Neither argument is part of the plan schema.** Both are Python-only keyword arguments;
  a plan YAML/JSON file cannot configure observers at all (this is what the
  [extension-mechanism table](overview.md#extension-mechanisms) means by "not part of the
  plan's YAML schema").

`ObserverSet` (`phaser/observer.py`) itself is also an `Observer`: it forwards every event
to each observer it wraps, in the order they were passed, via a small
`_fwd_to_children`-decorated method per event. `execute_engine` additionally wraps the
whole current observer set in another `ObserverSet` alongside a fresh `PatienceObserver`
whenever `plan.engines[i].early_termination` is set for that engine
(`phaser/execute.py:60-65`) — so early termination is layered on top of whatever observers
were already constructed, per engine, rather than being part of the default list above.

## Built-in observers

Three built-in observers exist, all in `phaser/observer.py`; two are installed by default
(see above) and one is opt-in via plan configuration.

### `LoggingObserver`

Installed by default whenever `observers=` (not `override_observers=`) is used, or when
neither argument is passed. Logs, via the standard `logging` module:

- `init_recons`/`start_recons`: "Initializing reconstruction..." then "Initialized
  reconstruction in MM:SS.mmm", and records wall-clock and UTC timestamps into
  `state.progress['utc']`/`state.progress['time']` the first time a reconstruction starts
  (`init_state.iter.total_iter == 0`) — this is how the `utc`/`time` progress series shown
  in saved state files gets populated.
- `init_engine`/`start_engine`: "Initializing engine..." then "Engine initialized".
- `update_iteration`: one line per iteration with elapsed time and, if present in the
  `errors` dict, `total_loss` and any other reported error keys — also appends to the
  `utc`/`time` progress series, when those keys are already present in `state.progress`.
- `finish_engine`/`finish_recons`: total engine time and total reconstruction time.

`LoggingObserver` takes no configuration; it has no constructor arguments beyond `self`.

### `SaveObserver`

Installed by default whenever `observers=` (not `override_observers=`) is used, or when
neither argument is passed. Configuration comes entirely from the running engine's plan
fields, read in `init_engine`:

- **`plan.save`** and **`plan.save_images`** — [flags](../concepts/glossary.md#flag)
  controlling whether state and images, respectively, are written on a given iteration
  (`process_flag`, evaluated in `update_iteration`).
- **`plan.save_options`** (`SaveOptions`, `phaser/plan.py`) — the output directory format
  string (`out_dir`, formatted with `engine_num`, `name`, `group`, `niter`, and any extra
  keyword arguments `init_engine` receives) and which images to write.

`SaveObserver` creates `out_dir` (if any output is configured for that engine) in
`init_engine`, writes state/images on flagged iterations in `update_iteration`, writes one
more final state/image pair in `finish_engine` whenever any output was configured for the
engine — regardless of whether the final iteration's own flag was true — and touches a
`finished` marker file in `close` if no exception occurred. If the output directory
changes between engines (a new `out_dir` format resolves to a different path), the
previous directory is closed (its `finished` marker written) before the new one is used.

### `PatienceObserver`

**Not** installed by default. `execute_engine` (`phaser/execute.py:60-65`) constructs one
automatically, wrapped around the existing observer set, whenever an engine's
`early_termination` plan field is set — a plan author enables this through
`EnginePlan.early_termination` (patience, in iterations; `None` disables it, the default)
and `EnginePlan.early_termination_smoothing` (default `0.9`), not by passing an observer
directly: `PatienceObserver(plan.early_termination, plan.early_termination_smoothing)`.
`continue_next_engine` has no plan field and is always left at its constructor default
(`True`) when built this way. The constructor signature is
`PatienceObserver(patience, smoothing=0.1, continue_next_engine=True)`:

- tracks an exponential moving average of `errors['total_loss']` (weight `smoothing` on
  the newest value) purely for its own bookkeeping — the moving average is not currently
  read back out anywhere;
- counts iterations since the best (lowest) raw `total_loss` seen so far;
- raises `EarlyTermination(state, continue_next_engine)` once that count reaches
  `patience`, which `execute_engine` catches, calls `finish_engine`, and either continues
  to the next engine (`continue_next_engine=True`, the default) or re-raises to end the
  whole reconstruction.

## A minimal custom observer

The observer below implements only the events it needs (every other `Observer` method
keeps its no-op default) and carries no external dependency. It records the reported
`total_loss` at every iteration and counts how many engines and groups it saw — a useful
skeleton for a custom observer that collects a metric over a run without touching
`SaveObserver`/`LoggingObserver`'s behavior, since it is used with `observers=`, which
keeps both active.

```python
class LossHistoryObserver(Observer):
    """Minimal custom observer: records the reported total_loss at every
    iteration and counts engines/groups seen. Implements only the events it
    needs; every other `Observer` method keeps its no-op default."""

    def __init__(self):
        self.losses: t.List[float] = []
        self.engines_started: int = 0
        self.groups_seen: int = 0
        self.finished: bool = False

    def init_engine(self, init_state, *, recons_name, plan, **kwargs):
        self.engines_started += 1

    def update_group(self, state, force: bool = False):
        self.groups_seen += 1

    def update_iteration(self, state, i: int, n: int, errors: t.Dict[str, float]):
        if (loss := errors.get('total_loss')) is not None:
            self.losses.append(loss)

    def finish_recons(self, state):
        self.finished = True
```

Used with `execute_plan`, this keeps the default `SaveObserver`/`LoggingObserver` active
alongside it:

```python
from phaser.execute import execute_plan

history = LossHistoryObserver()
execute_plan(plan, observers=history)
# after the run: history.losses is the per-iteration total_loss series
```

This is the same implementation exercised by
`tests/test_doc_examples.py::test_loss_history_observer_event_sequence`, which drives it
through a synthetic event sequence matching the call order in
[Event lifecycle and call order](#event-lifecycle-and-call-order) above, without running a
real reconstruction — see [Extension testing](testing.md#testing-a-custom-observer-a-synthetic-event-sequence) for
the pattern.

## Advanced example: Optuna integration

`examples/optuna_study.py` defines `OptunaObserver` (around line 115), used to drive
Optuna hyperparameter search over reconstruction plans. It overrides only
`update_iteration`: every `MEASURE_EVERY` iterations it computes a structural-similarity
error against a ground-truth image, reports that error to the active `optuna.Trial` via
`trial.report(error, i)`, writes a comparison plot, and raises `optuna.TrialPruned()` if
`trial.should_prune()` — so Optuna's pruning decision is driven entirely by the reported
error series, using the same `update_iteration` event every other observer sees. This
example calls `execute_engine` directly (not `execute_plan`) and attaches the observer
with `PreparedRecons.with_observer(...)`, which appends to whatever observer set the
reconstruction already has (`phaser/state.py`) — the same append semantics as
`observers=`, just expressed on an already-prepared reconstruction rather than at
`execute_plan` call time. See `examples/optuna_study.py` for the full trial-reporting and
plotting logic, which this page does not duplicate.

## Advanced example: the web-worker observer

`WorkerObserver` (`phaser/web/worker.py`, around line 43) forwards reconstruction state to
a remote server over HTTP, for the web manager and local/Slurm worker deployments
documented on [Interfaces and deployment](interfaces.md). It overrides:

- `init_engine` — sends the initial state for that engine (`send_update`);
- `heartbeat` — implemented to send a `PingMessage` if no message has been sent in the
  last 5 seconds, but — consistent with the note above — no code in `run_worker` or
  elsewhere in `phaser/web/worker.py` actually calls `observer.heartbeat()`; the method is
  defined but currently unreachable in this codebase;
- `update_group` — sends the current state if `force` is set or more than 30 seconds have
  passed since the last message, so per-group updates are rate-limited rather than sent
  every group;
- `update_iteration` — sends the current state unconditionally, every iteration.

Every send goes through `send_message`, which raises `SignalException` if the server
responds with a `signal` message — this is how a server-initiated cancel or shutdown
request reaches a running reconstruction from inside an observer callback. `run_worker`
attaches `WorkerObserver` via `execute_plan(plan, observers=WorkerObserver(...))` — append
semantics again, so a worker-run reconstruction still gets `SaveObserver`/`LoggingObserver`
locally in addition to remote reporting. See `phaser/web/worker.py` for the full polling
and job-management loop, which this page does not duplicate.

## Maintainer sources

- `phaser/observer.py`
- `phaser/execute.py`
- `phaser/state.py`
- `phaser/plan.py`
- `phaser/web/worker.py`
- `examples/optuna_study.py`
- `tests/test_doc_examples.py`
