# Parameter reference

Phaser's parameter reference is a hybrid of generated facts and curated guidance:

- **Generated inventories** (planned under `reference/`, built from `phaser/plan.py`,
  `phaser/types.py`, and the hook registries) are the factual source for YAML paths,
  accepted types, required status, defaults, built-in hook names, aliases, property
  schemas, and declared optional dependencies. They are complete and neutral, and are not
  yet built as of this writing (see Phase 2 of the
  [implementation plan](../../design/implementation-plan.md)).
- **These curated pages** organize the same options by the decision you're making —
  calibration, initialization, geometry, grouping and memory, noise model, solvers,
  schedules, regularization, termination, output — and supply what generation cannot:
  physical units, valid ranges in practice, lifecycle stage, supported engines/backends,
  interactions with other options, and a minimal example. Curated pages will embed or
  link generated facts rather than restating types and defaults by hand.

Every documented option states: type, default, units, valid range, lifecycle stage,
supported engines/backends, interactions, and a minimal example. Missing units or
descriptions are source-documentation debt — never guessed.

## What's here

- **[Data and calibration](data-and-calibration.md)**
- **[Initialization](initialization.md)**
- **[Simulation geometry](simulation-geometry.md)**
- **[Grouping and memory](grouping-and-memory.md)**
- **[Noise models](noise-models.md)**
- **[Solvers and learning rates](solvers-and-learning-rates.md)**
- **[Schedules and flags](schedules-and-flags.md)**
- **[Regularization](regularization.md)**
- **[Termination and diagnostics](termination-and-diagnostics.md)**
- **[Output and restart](output-and-restart.md)**

## Suggested reading order

Read [Initialization](initialization.md) and
[Simulation geometry](simulation-geometry.md) first — they determine what the rest of a
plan operates on. [Grouping and memory](grouping-and-memory.md),
[Noise models](noise-models.md), and [Solvers and learning rates](solvers-and-learning-rates.md)
follow from your choice of engine in
[Choosing a reconstruction engine](../engine-selection.md). The remaining pages
([Schedules and flags](schedules-and-flags.md), [Regularization](regularization.md),
[Termination and diagnostics](termination-and-diagnostics.md),
[Output and restart](output-and-restart.md)) are independent of each other and can be
read as needed.

All ten decision-domain pages are written. [Schedules and flags](schedules-and-flags.md)
carries the mandatory trust warning for expression schedules (see the
[authoring guide](../../design/authoring-guide.md#trust-model-and-warning-conventions)).

!!! note "Verification status"
    These pages were written from the plan schema, the generated reference, and the engine
    source in a documentation-only pass. Practical value ranges are given only where an
    example or benchmark evidences them; otherwise the page says selection guidance is
    pending. A later pass will validate the runnable snippets and fill evidenced ranges.
