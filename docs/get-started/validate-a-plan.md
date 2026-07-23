# Validating a plan

`phaser validate` parses a [plan](../concepts/glossary.md#plan) file and checks it against
the schema, without loading data or running a reconstruction. Run it before every
`phaser run` — it is the fastest way to catch a typo or a wrong field, in seconds instead
of after a long run.

```console
$ phaser validate path/to/plan.yaml
Validation of plan successful!
```

## What it checks, and what it doesn't

Validation confirms the plan is structurally valid: every field name is recognized, types
and required fields are satisfied, and each [hook](../concepts/glossary.md#hook) short name
(loader, noise model, solver, regularizer, …) resolves to a registered implementation with
valid properties (`phaser/plan.py`, `phaser/cli/validate.py`).

It does **not** check anything that needs your data or the runtime:

- that `raw_data.path` exists or the file loads;
- that a schema-valid combination is actually runnable — the schema accepts some
  combinations that fail at run time (for example `noise_model: poisson` on a conventional
  engine). See [Choosing a reconstruction engine](../cookbook/engine-selection.md) for the
  real compatibility rules;
- that external `"package.module:function"` hooks exist or have valid properties —
  external hook properties are passed through unvalidated.

A plan that validates can still fail at run time; validation only rules out schema errors.

## Reading a validation error

On failure, `phaser validate` prints `Validation failed:` followed by the error and exits
with a nonzero status:

```console
$ phaser validate broken_plan.yaml
Validation failed:
<message identifying the offending field and why it was rejected>
```

The message names the field path and the reason (unknown field, wrong type, missing
required value, or an unresolvable hook name). Fix that field and re-validate. Compare
field names, types, and defaults against the [generated plan
reference](../generated/plan/index.md) and the [parameter
reference](../cookbook/parameters/index.md), which is organized by the decision you're
making rather than by YAML nesting.

## Useful options

- **Read from stdin.** With no path (or `-`), the plan is read from standard input, so you
  can validate a generated or piped plan: `cat plan.yaml | phaser validate`.
- **Machine-readable output.** `--json` emits `{"result": "success", ...}` or
  `{"result": "error", "error": "..."}` — useful in scripts and sweeps.
- **Multiple plans per file.** A YAML file may hold several documents (separated by `---`);
  validation reports how many validated (`Validation of N plans successful!`).

## Maintainer sources

- `phaser/cli/validate.py`
- `phaser/plan.py`
