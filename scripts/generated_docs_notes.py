"""
Curated runtime-default notes for the generated plan reference (WP-2 / blocker B14).

Some plan fields have a schema default that is a *sentinel* — the schema says
`None`, but the actual behavior at that default is decided by engine code at run
time, not by `phaser/plan.py`. `shuffle_groups` is the motivating example:
its schema default is `None`, but `phaser/engines/conventional/run.py:31` and
`phaser/engines/gradient/run.py:179` both resolve it as
`props.shuffle_groups or not props.compact` before use. A generated reference
that only prints "default: None" for a field like this would be technically
correct and practically misleading.

`scripts/generate_docs.py` cannot infer these resolutions — the formula lives in
engine code, not in the schema — so it never tries to. Instead, this module is a
plain, hand-maintained dict:

    RUNTIME_DEFAULT_NOTES: dict[str, str]

keyed by `"ClassName.field_name"`, where `ClassName` is the plan class exactly as
it is rendered in `docs/generated/plan/index.md` (e.g. `"ConventionalEnginePlan"`,
not the base class `EnginePlan` it may be inherited from — the same field can
resolve differently per engine, as `jit_unroll_slices` does below, so the key is
the concrete, rendered class). The generator looks up this dict while rendering
each field and appends the note (rendered as-is, in the field's "Notes" column)
when present. A missing entry renders nothing extra — never a guess.

Every entry here must cite the exact file and line(s) it was verified against.
Add an entry only after reading the cited code yourself; do not add a note for
a sentinel you have not personally traced to its resolution.
"""

RUNTIME_DEFAULT_NOTES: dict[str, str] = {
    "ConventionalEnginePlan.grouping": (
        "Schema default `None`. At runtime, `None` (or `0`) resolves to `64` "
        "(`self.grouping = grouping or 64`, `phaser/engines/common/simulation.py:31`, "
        "`GroupManager.__init__`, shared by both engines)."
    ),
    "GradientEnginePlan.grouping": (
        "Schema default `None`. At runtime, `None` (or `0`) resolves to `64` "
        "(`self.grouping = grouping or 64`, `phaser/engines/common/simulation.py:31`, "
        "`GroupManager.__init__`, shared by both engines)."
    ),
    "ConventionalEnginePlan.shuffle_groups": (
        "Schema default `None`. At runtime (`phaser/engines/conventional/run.py:31`), "
        "the resolved value is `props.shuffle_groups or not props.compact` — any "
        "falsy `shuffle_groups` (including an explicit `False`, not just `None`) "
        "resolves to `not compact`, so the effective default is `True` when "
        "`compact` is `False` and `False` when `compact` is `True`."
    ),
    "GradientEnginePlan.shuffle_groups": (
        "Schema default `None`. At runtime (`phaser/engines/gradient/run.py:179`), "
        "the resolved value is `props.shuffle_groups or not props.compact` — any "
        "falsy `shuffle_groups` (including an explicit `False`, not just `None`) "
        "resolves to `not compact`, so the effective default is `True` when "
        "`compact` is `False` and `False` when `compact` is `True`."
    ),
    "ConventionalEnginePlan.jit_unroll_slices": (
        "Schema default `None`. In the conventional engine "
        "(`phaser/engines/conventional/solvers.py:37` and `:324`, both "
        "`LSQMLSolver.__init__` and `EPIESolver.__init__`), `None` resolves to "
        "`False` (no slice unrolling)."
    ),
    "GradientEnginePlan.jit_unroll_slices": (
        "Schema default `None`. In the gradient engine "
        "(`phaser/engines/gradient/run.py:162`), `None` resolves to `10` "
        "(unroll 10 slices during JIT compilation)."
    ),
    "ConventionalEnginePlan.slices": (
        "Schema default `None`. At an engine boundary (`phaser/execute.py:439-444`), "
        "`None` means keep the object's current slice count and thicknesses (no "
        "reslicing); a non-null value resamples the object when its thicknesses "
        "differ from the current state's."
    ),
    "GradientEnginePlan.slices": (
        "Schema default `None`. At an engine boundary (`phaser/execute.py:439-444`), "
        "`None` means keep the object's current slice count and thicknesses (no "
        "reslicing); a non-null value resamples the object when its thicknesses "
        "differ from the current state's."
    ),
    "ReconsPlan.backend": (
        "Schema default `None`. At runtime (`phaser/utils/num.py:132-180`, "
        "`get_backend_module`/`get_default_backend`), `None` triggers "
        "auto-detection: a device-active JAX or Torch GPU/TPU is preferred, "
        "then the first importable backend among JAX, Torch, and CuPy (in that "
        "order), falling back to `numpy` if none are importable."
    ),
    "ReconsPlan.wavelength": (
        "Schema default `None`. At runtime (`phaser/execute.py:170-172`), `None` "
        "falls back to the wavelength reported by the `raw_data` loader hook; if "
        "neither the plan nor the loader supplies one, `execute_plan` raises "
        "`ValueError` (\"wavelength must be specified by raw_data or manually\")."
    ),
    "ReconsPlan.slices": (
        "Schema default `None`. This value is passed through as-is to the "
        "object-initialization hook. The built-in `random` object hook "
        "(`phaser/hooks/object.py:11-16`) treats `None` as a single slice (no "
        "slice dimension on the object array); a non-null value adds a slice "
        "dimension sized to `len(slices.thicknesses)`. A custom object hook "
        "could interpret `None` differently."
    ),
    "InitPlan.object": (
        "Schema default `None`. At runtime (`phaser/execute.py:339`), `None` "
        "resolves to `ObjectHook('random')` — the built-in `random` object hook "
        "with its own schema defaults — unless a prior state already supplies an "
        "object and `init.object` was left unset (see initialization merge "
        "semantics)."
    ),
}
