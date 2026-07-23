# Extension testing

This page covers how Phaser's existing test suite is organized and invoked, patterns for
testing a custom [hook](../concepts/glossary.md#hook) and a custom
[observer](../concepts/glossary.md#observer) without running a full reconstruction, the
public API policy for importable internal modules, and — per the verified
[blocker outcome B12](../design/implementation-checklist.md#blocker-triage-phase-1-do-first) —
which parts of Phaser are demonstrated only by examples rather than asserted by a test.

## How the existing tests are organized and invoked

All tests live under `tests/` (`testpaths = ["tests"]`, `pyproject.toml`), and are run
with `pytest` from the repository root. As of this page, the directory holds:

- **Utility tests**: `test_num.py`, `test_image.py`, `test_object.py`, `test_optics.py`,
  `test_physics.py`, `test_misc.py` — array-backend helpers, image transforms, object/probe
  construction primitives, optics, physics constants, and grouping utilities.
- **Data-loading and initialization tests**: `test_empad.py`, `test_load.py`,
  `test_initialization.py` — raw-data loader parsing, `RawData` shape/dtype invariants, and
  the [initialization merge semantics](lifecycle.md#raw-data-loading-and-the-initialization-merge)
  (loader/`init.*` merge, `{}` restart behavior, previous-state reuse).
- **`test_doc_examples.py`** — the minimal custom-hook and custom-observer examples shown
  on this page and on [Observers](observers.md), so the documented examples are exercised
  by a test rather than only asserted to work in prose.

`tests/conftest.py` declares four backend markers — `numpy`, `jax`, `cupy`, `torch` — plus
`slow`, registered in `pyproject.toml`'s `[tool.pytest.ini_options]`. A test marked with a
backend name is skipped automatically if that backend is not importable (or, for `cupy`,
if no CUDA device is visible), via `pytest_runtest_setup`'s `AVAILABLE_BACKENDS` check —
so `pytest` alone is the right invocation everywhere; there is no separate command per
backend. `tests/utils.py`'s `with_backends(...)` decorator applies these markers as
`pytest.mark.parametrize`, so one test function can run once per available backend. CI
(`.github/workflows/ci.yaml`) installs `.[dev,jax,torch,web]` (no `cupy` extra, and no GPU
runner) and runs plain `pytest`, so `jax`- and `torch`-marked tests run in CI but
`cupy`-marked tests are always skipped there.

None of this requires an installed dataset or network access: the tests referenced on this
page run on the plain `numpy` backend with no optional dependencies, which is also the
constraint this page's own examples are held to.

## Testing a custom hook: property schema and callable contract

A [hook](../concepts/glossary.md#hook) is, once resolved, a plain function called as
`f(args, props=props)` (`Hook.__call__`, `phaser/hooks/hook.py:61-62` — see
[Hooks](hooks/index.md#a-resolved-hook-is-a-plain-callable)). Testing a custom hook does
not require constructing a `Hook` object, a full `ReconsPlan`, or going through
`phaser validate` at all: call the underlying function directly with a properties value
and the input shape that hook family expects (see each
[hook-family page](hooks/index.md#hook-families-and-when-they-run) for its exact argument
type). This tests the two things that matter for correctness — the property schema
(does the properties type accept and default the fields you expect?) and the callable
contract (does the function accept that properties value and the family's input type, and
return the family's output type?) — without needing plan parsing or reconstruction state
at all.

The example below is a minimal custom `post_load` hook (`Hook[RawData, RawData]`, see
[Post-load](hooks/post-load.md)) that subtracts a constant dark count and clips negative
intensities to zero, tested against a tiny synthetic `RawData` dictionary built directly
in the test, with no dataset:

```python
@dataclass
class SubtractDarkProps:
    """Properties for the 'subtract_dark' custom post_load hook."""
    dark: float = 0.0


def subtract_dark(raw_data: RawData, props: SubtractDarkProps) -> RawData:
    """Minimal custom post_load hook: subtract a constant dark count and
    clip negative intensities to zero. Matches the `Hook[RawData, RawData]`
    callable contract: called as `f(raw_data, props=props)`."""
    raw_data['patterns'] = numpy.clip(raw_data['patterns'] - props.dark, 0.0, None)
    return raw_data


def test_subtract_dark_hook():
    raw_data = _make_raw_data(
        numpy.array([[[[1.0, 2.0], [0.5, 10.0]]]], dtype=numpy.float32)
    )
    props = SubtractDarkProps(dark=1.0)

    result = subtract_dark(raw_data, props)

    assert result is raw_data
    numpy.testing.assert_allclose(
        result['patterns'], [[[[0.0, 1.0], [0.0, 9.0]]]]
    )
```

This is the same implementation and test as
`tests/test_doc_examples.py::test_subtract_dark_hook`; see that file for the
`_make_raw_data` helper (a few lines building a `RawData` dict around a `Sampling`).

If the custom hook is registered as an external reference in a plan
(`type: "package.module:function"`), it is **not** schema-validated the way a built-in
hook's registered properties dataclass is (see [Hooks](hooks/index.md#hook-anatomy)) —
`pane` never sees `SubtractDarkProps` unless you convert into it yourself. A
property-schema test like the one above (asserting the dataclass's own defaults and field
types) is therefore the only check that a malformed properties mapping would otherwise
only surface as a runtime error the first time the hook is actually called, not at
plan-parse time.

## Testing a custom observer: a synthetic event sequence

An [observer](../concepts/glossary.md#observer) reacts to a fixed sequence of method calls
(see [Observers: event lifecycle and call order](observers.md#event-lifecycle-and-call-order))
made by `phaser/execute.py` and the engine `run_engine` functions. Testing a custom
observer does not require running a real reconstruction: construct the observer and call
its methods directly, in the verified order, with placeholder arguments for whatever the
observer under test does not read. Because `Observer` methods are plain Python methods
with no runtime type checking, passing `None` for a `state`/`plan` argument the observer
does not use is safe and keeps the test free of any dependency on `ReconsState` or
`ReconsPlan` construction.

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


def test_loss_history_observer_event_sequence():
    observer = LossHistoryObserver()

    observer.init_recons(plan=None)
    observer.start_recons(init_state=None)

    observer.init_engine(None, recons_name='test', plan=None)
    observer.start_engine(None)

    for (i, loss) in enumerate([1.0, 0.5, 0.2], start=1):
        observer.update_group(None)
        observer.update_group(None)
        observer.update_iteration(None, i, 3, {'total_loss': loss})

    observer.finish_engine(None)
    observer.finish_recons(None)
    observer.close(None)

    assert observer.engines_started == 1
    assert observer.groups_seen == 6
    assert observer.losses == [1.0, 0.5, 0.2]
    assert observer.finished is True
```

This is the same implementation and test as
`tests/test_doc_examples.py::test_loss_history_observer_event_sequence`. The pattern
generalizes: drive whatever subsequence of the nine events (`init_recons`, `start_recons`,
`init_engine`, `start_engine`, any number of `update_group`/`update_iteration` pairs,
`finish_engine`, `finish_recons`, `close`) your observer actually overrides, in the order
given on the [Observers](observers.md#event-lifecycle-and-call-order) page, and assert on
whatever state the observer accumulated. This does not test that `phaser/execute.py` or an
engine calls your observer correctly — that part is Phaser's responsibility, verified once
for the built-in call sites and described on [Observers](observers.md) — only that your
observer does the right thing when those calls happen.

## Public API policy

The [documentation architecture](../design/documentation-architecture.md#non-goals)'s
non-goals state plainly: **"Importable internal modules are not automatically part of the
supported public API."** Nearly every internal Phaser module can be imported by a
developer working in the same environment (Python does not enforce module privacy), but
that alone does not mean a change to its function signatures is treated as a breaking
change the way a change to `phaser.execute.execute_plan` would be.

In practice, across this documentation:

- The names imported by this page's and [Observers](observers.md)'s examples —
  `phaser.execute.execute_plan`/`execute_engine`/`initialize_reconstruction`,
  `phaser.observer.Observer`/`ObserverSet`/`LoggingObserver`/`SaveObserver`/`PatienceObserver`,
  and the `Hook`-family classes and `RawData`/`PostLoadHook` types under `phaser.hooks` —
  are the surface a contributor is expected to build a custom hook or observer against, and
  are exercised directly by `tests/test_doc_examples.py`.
- Several internal modules (for example `phaser.execute`, `phaser.types`, `phaser.plan`,
  and most of `phaser.utils.*`) declare a module-level `__all__`, but this project does not
  currently generate or otherwise enforce a distinct, reviewed "public API" boundary from
  that convention alone — the [documentation architecture](../design/documentation-architecture.md#generated-references)'s
  planned generated signatures/docstrings for "the explicitly supported Python API" (its
  Phase 2/3 generated-references work) do not exist yet at the time of writing. Until that
  exists, treat "documented in the cookbook or this architecture guide" as the operative
  definition of supported, and everything else — including most of `phaser.engines.*` and
  undocumented helpers under `phaser.utils.*` — as an implementation detail that may change
  without notice, however convenient it is to import today.

## Example-demonstrated versus test-covered (B12)

Verified blocker [B12](../design/implementation-checklist.md#blocker-triage-phase-1-do-first)
found that the existing test suite (before `test_doc_examples.py`) covers utilities and
data-loading/initialization but **no test exercises**:

- `phaser/engines/conventional/` — the ePIE and LSQML solvers;
- `phaser/engines/gradient/` — `run_engine`, or the SGD/Adam/Polyak-SGD solvers;
- `phaser/engines/common/noise_models.py` — the amplitude, Anscombe, and Poisson noise
  models;
- `phaser/engines/common/position_correction.py` — the steepest-descent and momentum
  position solvers;
- `phaser/hooks/schedule.py` — constant, piecewise, and expression schedules, and flags;
- `phaser/hooks/regularization.py` — cost regularizers, group constraints, and iteration
  constraints;
- `phaser/observer.py` — none of the four tests that call `initialize_reconstruction`
  (`tests/test_initialization.py`) pass a non-empty `plan.engines`, and none call
  `execute_plan`/`execute_engine`. Because none of them pass `observers=` or
  `override_observers=` either, the default `LoggingObserver`/`SaveObserver` pair is
  constructed and its `init_recons`/`start_recons` no-ops do run incidentally — but no
  test asserts anything about observer behavior, and because no test ever runs an engine,
  `update_iteration`, `update_group`, `finish_engine`, `SaveObserver`'s file-writing, and
  `PatienceObserver` are exercised by no test at all.

`test_doc_examples.py` (this page and [Observers](observers.md)) closes a narrow, different
gap: it test-covers the two *minimal custom examples this documentation shows*
(`subtract_dark`, `LossHistoryObserver`) by construction, but it does **not** exercise any
built-in observer, engine, noise model, position solver, schedule, or regularizer — B12's
finding above is unchanged by its addition. Reconstruction correctness for every engine,
noise model, and position-correction path remains demonstrated only by `examples/*.yaml`
being run manually or as bounded smoke tests, never by an automated correctness assertion.

**Guidance for labeling documentation.** Per the
[architecture document](../design/documentation-architecture.md#sources-of-truth) and the
[authoring guide](../design/authoring-guide.md), every page that describes reconstruction
behavior (engine correctness, a noise model's numerical behavior, position correction,
schedules, or a regularizer's effect on state) must label that behavior **example-demonstrated**
— shown to run via a repository example or a documentation smoke test, but not asserted
correct by an automated test — rather than implying **test-covered**, which this
documentation reserves for behavior an automated test in `tests/` actually asserts. Do not
use "tested" or "verified" for example-only behavior; state which of the two applies, and
name the example or test.

## Maintainer sources

- `phaser/observer.py`
- `phaser/execute.py`
- `phaser/hooks/hook.py`
- `phaser/hooks/__init__.py`
- `tests/conftest.py`
- `tests/utils.py`
- `tests/test_initialization.py`
- `tests/test_doc_examples.py`
- `pyproject.toml`
- `.github/workflows/ci.yaml`
