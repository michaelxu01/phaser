# Documentation architecture

| Field | Value |
| --- | --- |
| Status | Accepted |
| Audience | Phaser users, contributors, and documentation maintainers |
| Owner | Phaser maintainers |
| Adopted | 2026-07-22 |
| Revised | 2026-07-22 — two-track structure, hybrid parameter reference, blocker tracking |
| Review trigger | A plan-format change, a new extension surface, or a documentation-builder change |

## Why this design exists

Phaser combines electron-ptychography concepts, reconstruction workflows, configuration plans, and an extensible Python implementation. Readers range from microscopists running their first reconstruction to developers adding reconstruction engines. A single reference arranged around Python modules or a flat list of configuration options will not serve all of them.

Phaser is best understood as a typed reconstruction-plan executor: a YAML/JSON plan is validated into a `ReconsPlan`, a raw-data loader hook runs, loader metadata is merged with explicit initialization hooks, a `PreparedRecons` is constructed, one or more engines run sequentially against shared state, and observers react throughout (logging, checkpoints, images, termination, remote updates). The documentation challenge is that configuration is not merely nested — it is **conditional** on lifecycle stage, engine family, backend, and reconstruction variable. A flat catalog of all options would obscure invalid combinations (for example, the Poisson noise model currently supports the gradient engine but not the conventional engines, even though the schema accepts it for both).

The previous documentation is also substantially out of date. It can identify topics worth revisiting, but it is not evidence for current behavior. This design establishes which sources are authoritative and how new material will remain understandable and verifiable.

## Goals

- Help a scientist reach a validated reconstruction without requiring software-engineering knowledge.
- Explain the reconstruction lifecycle and code structure without hiding scientifically important details.
- Provide separate, connected tracks for reconstruction users and extension developers.
- Make engine-, backend-, and lifecycle-dependent options visible before a reader copies them, with the engine compatibility matrix as a central artifact.
- Derive factual reference material from code so names, types, defaults, and built-in hooks do not drift, while curated pages supply selection guidance the code cannot.
- Build and publish the site with Zensical using portable Markdown.
- Make examples testable and organize them by user goal rather than specimen name.

## Non-goals

- This proposal does not make existing prose authoritative.
- The first migration does not rewrite every guide or promote every repository example into a tutorial.
- Generated API pages do not replace conceptual or task-oriented explanations.
- Importable internal modules are not automatically part of the supported public API.
- Fixing suspected code bugs uncovered during documentation work is out of scope for this effort; they are tracked as blockers (see [Verification blockers](#verification-blockers)) and resolved by code owners.

## Sources of truth

Documentation claims must be checked against the following sources, in order.

1. **Plan syntax, accepted types, and defaults:** `phaser/plan.py` and `phaser/types.py`.
2. **Built-in hook names and property schemas:** each hook category's `known` registry under `phaser/hooks/`, including registrations performed in `phaser/plan.py`. Hook resolution and custom `module:function` behavior come from `phaser/hooks/hook.py`.
3. **Lifecycle and runtime behavior:** `phaser/execute.py`, `phaser/observer.py`, and the relevant call sites under `phaser/engines/`.
4. **State and persistence contracts:** `phaser/state.py` and its serialization utilities.
5. **Demonstrated behavior:** tests, then examples. A test takes precedence when an example disagrees with it. Documentation must distinguish "demonstrated in an example" from "covered by an engine correctness test."
6. **Existing prose:** topic inventory only. A factual statement must be reverified against one of the sources above before reuse.

Every new substantive page should end with a short **Maintainer sources** section. This is a maintenance trail, not a request to expose implementation details throughout beginner material.

## Two-track structure

The documentation is organized as two major tracks plus shared front matter and generated reference material.

### Track A — Reconstruction cookbook and parameters (scientists)

For scientists preparing data, configuring reconstruction, inspecting results, and troubleshooting. It contains:

- an engine selection guide built around the compatibility matrix;
- minimal complete reconstructions — self-contained, portable, validated plans;
- goal-oriented recipes (coarse-to-fine, engine handoffs, adding probe modes, position/tilt refinement, restart, regularization, Optuna sweeps);
- a parameter reference organized by decision domain (see [Parameter reference model](#parameter-reference-model));
- performance recipes and troubleshooting organized by symptom.

A reader on this track should not need to understand Python inheritance, registries, protocols, or package imports.

### Track B — Architecture and extension guide (developers)

For contributors and researchers extending Phaser. It contains:

- an overview: what Phaser reconstructs, single-slice versus multislice, mixed-state probes, supported engines and backends, and the overall execution diagram;
- the reconstruction lifecycle: parsing and validation, raw-data loading, metadata merge, state initialization, preprocessing, engine transitions, iterations and groups, saving and restart;
- state and scientific conventions: shapes, units, axis ordering, FFT/diffraction origin, intensity scaling, phase conventions;
- the hook system, with one page per hook family (see [Hook-family page template](#hook-family-page-template));
- the observer API: event lifecycle, append (`observers=`) versus override (`override_observers=`) behavior, built-in observers, custom observers, Optuna integration, and the web-worker observer;
- a JAX implementation guide: backend abstraction, pytrees, JIT boundaries, static versus dynamic configuration, slice-loop unrolling, avoiding recompilation, device transfer and buffering, and writing JAX-compatible custom hooks;
- interfaces and deployment: CLI, Python API, web manager, local/manual worker, Slurm worker, and the trust model (see [Trust model](#trust-model));
- extension testing patterns and public API policy.

Navigation presents Track A before Track B: the scientist path is the primary audience per the goals above, even though this design describes the architecture track first in places. Cross-links connect the two — every cookbook page links to the architecture pages that explain its machinery, and every hook-family page links to recipes that use it.

### Concepts that must be documented prominently

These were repeatedly found subtle or undocumented and get explicit treatment rather than passing mentions:

1. **Sequential engines can change the representation at engine boundaries** — resizing simulation arrays, expanding or resampling the object, changing probe-mode count, reslicing the object, and creating a tilt map when tilt optimization first becomes active (`phaser/execute.py`). This is what makes staged workflows (coarse conventional → gradient refinement → multislice → position/tilt) possible and belongs in both tracks.
2. **Initialization merge semantics** — loader metadata may supply partial initialization hooks; user `init.*` configurations merge recursively into matching metadata-derived hooks, replace them when the hook type differs, and an empty mapping `{}` requests metadata-derived initialization instead of reusing that component from a prior state (`phaser/execute.py`, `tests/test_initialization.py`).
3. **Regularization has three lifecycles** — cost regularizers (differentiable terms in the gradient objective), group constraints (mutate state after each group), and iteration constraints (mutate state after each iteration). Each gets its own hook-family page; conventional engines support constraints but not cost regularizers.
4. **Extension mechanisms are distinct even when they look similar in YAML.** Documentation must distinguish: configuration hooks (construct behavior from a plan), stateful solver/protocol objects (maintain algorithm state inside an engine), constraints/regularizers (alter objective or state on group/iteration cadence), observers (react to execution events; not plan hooks), and worker transport (send state externally).
5. **Compatibility is conditional despite a shared schema** — e.g. noise models share an interface, but Poisson supplies only a differentiable loss and raises `NotImplementedError` for conventional wave updates; the gradient engine requires JAX or Torch; gradient solvers separate per-group variables (probe, object) from per-iteration variables (positions, tilt) and reject solvers that mix them.
6. **`buffer_n_groups` is tri-state** — `0` transfers each group synchronously, a positive integer prefetches that many groups, and `null` loads all patterns onto the device. Parameters with sentinel semantics like this must document each state, not just a type.

## Accessibility and interpretability rules

Task-oriented pages use progressive disclosure in this order:

1. State the outcome in one sentence.
2. List prerequisites, input data, units, and expected output.
3. Give the smallest complete, copyable example.
4. Explain the operation in scientific language.
5. Explain common parameter choices and trade-offs.
6. Put advanced implementation details after the workflow.
7. End with validation steps and likely failure modes.

Additional requirements:

- **Be concise: lead with the answer, and never let prose restate an adjacent table.** Pages must be scannable — a reader reaches the fact they came for without wading through connective prose. State cross-cutting framing once, link tersely, and keep verification notes to one line. The [authoring guide](authoring-guide.md#writing-style-tighten-and-lead-with-the-answer) gives the enforced rules and length targets; concision is reviewed with the same weight as template completeness. Cut words, never facts, units, shapes, restrictions, or citations.
- Define a technical term at first use and link to the glossary.
- Explain what a hook changes before describing its Python callable.
- State physical units beside each physical parameter.
- State coordinate order and array shape explicitly, such as `(y, x)`, `(modes, y, x)`, or `(slices, y, x)`.
- Separate required, commonly adjusted, and advanced parameters.
- Mark engine-, backend-, or experimental restrictions visibly.
- Avoid unexplained terms such as *backend*, *JIT*, *registry*, *protocol*, and *serialization*.
- Use descriptive link text rather than “click here.”
- Use correct heading order and keyboard-accessible navigation.
- Never communicate meaning using color alone. Diagrams require adjacent text descriptions, and images require meaningful alternative text.
- Show the expected files, plots, metrics, or terminal messages produced by a workflow.

## Target information architecture

```text
docs/
  index.md
  design/
    documentation-architecture.md
    implementation-checklist.md
  get-started/                      # shared front matter
    index.md
    install.md
    first-reconstruction.md
    validate-a-plan.md
  concepts/                         # shared scientific background
    ptychography.md
    coordinates-units-and-arrays.md
    glossary.md
  cookbook/                         # Track A
    index.md
    engine-selection.md             # compatibility matrix + selection guidance
    reconstructions/                # minimal complete reconstructions
      index.md
      simulated-single-slice-gradient.md
      simulated-multislice-gradient.md
      empad-experimental.md
      epie.md
      lsqml.md
    recipes/                        # goal-oriented recipes
      index.md
      coarse-to-fine.md
      conventional-to-gradient.md
      increasing-resolution.md
      changing-slices-between-engines.md
      mixed-state-probe-modes.md
      position-refinement.md
      tilt-refinement.md
      restart-from-hdf5.md
      restart-overriding-a-component.md
      cost-regularizers.md
      constraints.md
      optuna-sweeps.md
    parameters/                     # curated decision-domain reference
      index.md
      data-and-calibration.md
      initialization.md
      simulation-geometry.md
      grouping-and-memory.md
      noise-models.md
      solvers-and-learning-rates.md
      schedules-and-flags.md
      regularization.md
      termination-and-diagnostics.md
      output-and-restart.md
    performance.md
    troubleshooting.md
  architecture/                     # Track B
    index.md
    overview.md
    lifecycle.md
    state-and-conventions.md
    hooks/
      index.md                      # hook anatomy: resolution, schemas, custom hooks
      raw-data-loaders.md
      initialization.md
      post-load.md
      post-init.md
      schedules-and-flags.md
      noise-models.md
      solvers.md
      cost-regularizers.md
      group-constraints.md
      iteration-constraints.md
      engines.md
    observers.md
    jax.md
    interfaces.md                   # CLI, Python API, web manager, workers, trust model
    testing.md
  reference/                        # generated (see below); not committed
    plan/
    hooks/
    api/
```

The migration creates this structure incrementally. Legacy pages stay reachable with a warning until verified replacements exist. Work packages, task distribution, and sequencing live in [the implementation plan](implementation-plan.md); per-page tracking lives in [the implementation checklist](implementation-checklist.md).

## Parameter reference model

The parameter reference is a hybrid of generated facts and curated guidance:

- **Generated inventories** (under `docs/generated/`, built from `phaser/plan.py`, `phaser/types.py`, and the hook registries) are the factual source for YAML paths, accepted types, required status, defaults, built-in hook names, aliases, property schemas, and declared optional dependencies. They are complete and neutral.
- **Curated decision-domain pages** (`cookbook/parameters/`) organize the same options by the decision a user is making — calibration, initialization, geometry, grouping and memory, noise model, solvers, schedules, regularization, termination, output — and supply what generation cannot: physical units, valid ranges in practice, lifecycle stage, supported engines/backends, interactions with other options, and a minimal example. Curated pages embed or link generated facts rather than restating types and defaults by hand.

Every documented option must state: type, default, units, valid range, lifecycle stage, supported engines/backends, interactions, and a minimal example. Missing units or descriptions are source-documentation debt; the generator must not guess them.

The **engine compatibility matrix** (engines × backends × model features × noise models × refinable variables × regularizer categories × restart support × known limitations) starts as a curated table verified against code and tests. Once Phase 2 introduces explicit compatibility metadata in the registries, the matrix is generated from that metadata instead of maintained by hand. It must never be inferred by the generator from incidental code structure.

## Hook-family page template

Each hook-family page in `architecture/hooks/` documents, in order:

1. Lifecycle point — when the hook runs and what has already happened.
2. Callable signature and property schema.
3. Accepted state/input and returned value.
4. Built-in implementations (linked to the generated inventory).
5. A minimal custom implementation.
6. YAML invocation, for both built-in short names and external `package.module:function` hooks (noting external hook properties are not schema-validated).
7. Engine/backend restrictions.
8. Optional dependencies.
9. A testing pattern for custom implementations.

## Example flow template

Every documented reconstruction flow (minimal reconstructions and recipes) uses the same section structure:

1. **Goal** — what this reconstruction demonstrates.
2. **When to use it** — supported data and scientific assumptions.
3. **Compatibility** — engine, backend, noise model, and optimization variables.
4. **Input contract** — shapes, units, coordinate conventions, and normalization.
5. **Complete plan** — runnable YAML or Python.
6. **Execution flow** — loader → initialization → preprocessing → engine stages → output.
7. **Parameter walkthrough** — only parameters that matter for this flow, explaining every non-default option.
8. **Expected result** — metrics, images, state files, convergence behavior, and a basic success check.
9. **Variations** — small changes for related use cases.
10. **Failure modes** — symptoms, likely causes, and fixes.

In addition, every example records the following metadata, which also drives the goal-oriented index:

| Metadata | Purpose |
| --- | --- |
| Reader level | Beginner, intermediate, or advanced |
| Data origin | Simulated or experimental |
| Data loader | EMPAD, Gatan, Nion, manual, or custom |
| Model | Single-slice or multislice |
| Engine | Gradient, ePIE, or LSQML |
| Compute requirements | Backend, device assumptions, and optional dependencies |
| Updated variables | Object, probe, positions, and/or tilt |
| Features | Probe modes, schedules, regularization, restart, or custom extensions |
| Runtime class | Small smoke test, workstation run, or large reconstruction |
| Verification | Syntax validation, executable smoke test, or documented external run |
| Expected output | Files, metrics, images, and a basic success check |

Repository examples under `examples/` are evidence, not automatically tutorials: they are organized by specimen, and some contain machine-specific paths. Machine-specific paths and unavailable datasets must be replaced before publication as complete workflows. Complete plans are validated with the packaged `phaser validate` command; bounded examples also receive execution smoke tests.

## Trust model

Expression schedules currently use unrestricted Python `eval` (`phaser/hooks/schedule.py`), and external hooks execute arbitrary importable code. This is acceptable for a scientist running their own plans, but it means **a plan file is code**: running an untrusted plan is equivalent to running an untrusted script. Documentation must:

- carry an explicit warning on the schedules pages in both tracks;
- document the trust implications for the web manager and local/Slurm workers on `architecture/interfaces.md` — anyone who can submit a plan to a worker can execute code as that worker;
- never present expression schedules in beginner material without the warning.

If the implementation later sandboxes or restricts schedule expressions, these warnings are updated as part of that change.

## Diagrams and glossary

The initial maintained diagrams will describe:

1. The reconstruction lifecycle from plan validation to saved output, including engine transitions and what they may reshape.
2. Plan composition and where hook-valued fields appear.
3. Relationships among patterns, probe, object, scan, tilt, progress, and prepared reconstruction state.
4. Extension surfaces: hooks, stateful solvers, constraints, engines, observers, and worker transport.

Use Mermaid only after representative diagrams render accessibly in the pinned Zensical build. Otherwise, check in SVG output with adjacent editable source. Exact option lists do not belong in diagrams because they are generated from code.

The glossary is canonical and distinguishes scientific and implementation meanings where they overlap, including *object*, *probe*, *state*, *engine*, *mode*, *hook*, *backend*, and *iteration*.

## Generated references

Generated references are builder-neutral Markdown produced before Zensical runs. Generated files live under `docs/generated/`, are excluded from git, and are recreated from an empty directory for every build. A generation error fails the build.

### Generated facts

- YAML paths, accepted types, required status, and defaults from Pane-backed plan classes.
- Built-in hook names, aliases, property schemas, and declared optional dependencies from hook registries.
- Signatures and docstrings for the explicitly supported Python API.
- Links from a reference item to curated examples when metadata provides one.
- Compatibility facts, once explicit metadata exists (Phase 2); never inferred.

### Curated facts

- Physical units and scientific interpretation.
- Selection guidance and trade-offs.
- Compatibility and safety warnings.
- Expected behavior and troubleshooting.
- Public API support status.

Missing descriptions or units are source-documentation debt; the generator must not guess them. Generated content is not committed because this project has chosen build-time generation. Determinism is tested by producing the same tree twice and comparing it.

The first generated API surface is `phaser.state`, replacing the previous runtime mkdocstrings dependency. Later phases will generate plan and built-in hook references from `phaser.plan`, `phaser.types`, and the hook registries.

## Stub expansion from code

Most remaining stub pages can be drafted primarily from artifacts already in the
repository, then finished with curated guidance. This makes expansion a mechanical
extract-then-annotate task rather than open composition, and keeps the drafts verifiable.
Each stub names its **primary code source** — the artifact a draft is extracted from — and
its **curation** — the judgment a human or agent still adds. Anything requiring a
reconstruction to *run* is deferred to a later execution-enabled pass and marked
`Verification pending`, never fabricated.

| Stub | Primary code source (extract) | Curation (add) |
| --- | --- | --- |
| `reconstructions/{epie,lsqml,multislice-gradient,empad-experimental}.md` | the matching `examples/*.yaml` plan, plus the WP-5a smoke-data harness for a portable dataset | per-flow parameter walkthrough; expected-result numbers filled in the execution pass |
| `recipes/*.md` | diff the relevant `examples/*.yaml` against a baseline (staged plans, `update_*` flags, `solvers`/`init.state` blocks already demonstrate every recipe) | the "when to use it" framing and the one changed knob per recipe |
| `troubleshooting.md` | the actual `raise`/`warn` strings and finite-value checks in `phaser/execute.py`, engines, and loaders (grep for `ValueError`, `warn`, `RuntimeWarning`) | symptom → cause → fix mapping per message |
| `performance.md` | `buffer_n_groups`/`grouping`/`jit_unroll_slices` handling in `simulation.py`; `benchmarks/` | scaling intuition and device-specific advice |
| `get-started/install.md` | `pyproject.toml` dependency groups; CLI subcommands in `phaser/cli/` | the recommended path and backend/extra decision |
| `get-started/{first-reconstruction,validate-a-plan}.md` | the WP-5a smoke reconstruction and `phaser validate` output | narrative ordering for a first-time reader |
| `concepts/{ptychography,coordinates-units-and-arrays}.md` | `algorithms.md` math (reverified against engine code), `state.py` docstrings, `state-and-conventions.md` | scientific explanation pitched below the developer track |

The extract step reuses the existing infrastructure wherever possible: the generated
reference for schema facts, the smoke-data harness (`examples/smoke/`) for a portable
dataset instead of the absent `sample_data/`, and the verified blocker outcomes for known
restrictions. Where a class of fact recurs across many stubs (every recipe's changed knob,
every troubleshooting message), prefer teaching the generator to emit it over hand-copying
it — the same principle that governs the parameter reference.

## Zensical and publication

Zensical `0.0.51` is the pinned production builder. The project uses compatible `mkdocs.yml` input while avoiding unsupported runtime plugins. Production publication uses GitHub Pages Actions and one canonical site.

The migration decisions (implemented in Phase 0) are:

- Replace `markdown-include` with a self-contained home page.
- Generate API Markdown before building instead of requiring mkdocstrings.
- Replace Mike deployment and its runtime version selector with GitHub Pages artifact deployment.
- Preserve existing historical version directories only where compatible with the new Pages publication model; the new site does not claim a version selector until an equivalent is implemented.
- Retain representative MathJax coverage and verify it in every builder upgrade.
- Pin Zensical explicitly and review upgrades rather than floating to new pre-1.0 releases.

## Verification blockers

These findings must be resolved — verified against code, and where they are code defects, dispositioned by a code owner — **before** the affected documentation is written. They are not documentation errata; several are suspected implementation bugs. Documentation describes current behavior honestly in the meantime.

| # | Finding | Affected documentation | Resolution path |
| --- | --- | --- | --- |
| 1 | README describes gradient descent as JAX-only; code allows JAX or Torch (`phaser/execute.py`) | Engine selection, compatibility matrix, README | Verify and correct prose |
| 2 | Legacy-documented defaults disagree with the schema (backend, probe modes, iteration count) | All parameter pages | Superseded by generated reference; never copy legacy defaults |
| 3 | Terminology: prose says *regularizations*; schema uses *regularizers* | All regularization pages | Standardize on schema terminology |
| 4 | Legacy noise-model example implies a sequence; each engine accepts one noise-model hook | Noise-model pages | Correct in new pages; do not port legacy example |
| 5 | Diffraction-origin wording conflicts: state/loaders indicate corner-origin normalization; conventions prose is ambiguous | Conventions, input contracts | Verify against `phaser/state.py` and loaders; rewrite conventions |
| 6 | Poisson noise model raises `NotImplementedError` for conventional wave updates; schema does not reveal this | Compatibility matrix, noise-model pages | Document the restriction explicitly; candidate for compatibility metadata |
| 7 | ePIE position correction appears incomplete (TODO in `phaser/engines/conventional/solvers.py`) | Position-refinement recipe, compatibility matrix | Code owner verifies before the feature is documented as supported |
| 8 | Position correction may ignore `max_step_size` — `step_size` appears assigned to the cap field (`phaser/engines/common/position_correction.py`) | Position-refinement recipe, solver pages | Suspected code bug; code owner disposition required before documenting |
| 9 | Expression schedules use unrestricted `eval` | Schedules pages, interfaces page | Document per the trust model; warning is mandatory content |
| 10 | Some repository examples contain machine-specific paths (e.g. `examples/mos2_epie.yaml`) | All examples | Replace paths before promotion to documentation |
| 11 | Advertised features (segmented ptychography, adaptive propagator correction) were not clearly exposed in the plan schema | Overview, compatibility matrix | Verify existence before advertising; do not document unverified features |
| 12 | Tests emphasize utilities and initialization over engine correctness | Example verification labels | Label examples as example-demonstrated vs. test-covered |

A blocker is closed by recording the verified outcome in the implementation checklist. Blockers 7, 8, and 11 gate their pages entirely; the rest constrain content.

## Page ownership matrix

| Documentation area | Authoritative implementation sources | Reviewers |
| --- | --- | --- |
| Plans and parameters | `phaser/plan.py`, `phaser/types.py` | Plan-schema maintainers and a reconstruction user |
| Hooks and extensions | `phaser/hooks/`, `phaser/plan.py` registrations | Hook author and developer reviewer |
| Reconstruction lifecycle | `phaser/execute.py`, engine call sites | Engine maintainer |
| State and output | `phaser/state.py`, `phaser/observer.py`, output utilities | State/output maintainer |
| Engine guidance | Engine implementation and focused tests | Engine author and domain reviewer |
| Compatibility matrix | Engine and noise-model implementations, tests | Engine maintainer and domain reviewer |
| Examples | Validated plans, tests, and documented datasets | Domain user unfamiliar with the implementation |
| Interfaces and deployment | `phaser/cli/`, `phaser/web/`, worker implementations | Deployment maintainer |
| Deployment/tooling | Site config, generator, and documentation workflow | Documentation maintainer |

## Delivery phases

### Phase 0 — design and production-builder migration *(complete)*

- Publish this design.
- Make Zensical the production builder.
- Generate the existing state API page at build time.
- Add warnings to legacy prose.
- Keep existing navigation otherwise stable.

### Phase 1 — portable foundation and blocker triage

- Add both track landing pages, shared get-started and concepts stubs, the glossary, authoring guidance, and initial diagrams.
- Add link and accessibility checks.
- Triage every verification blocker: verify each against code, record outcomes in the checklist, and route suspected bugs (7, 8, 11) to code owners.
- Establish curated metadata conventions for generated reference facts.

### Phase 2 — generated plan/hook reference and compatibility metadata

- Introspect plan classes and all hook registries.
- Generate complete plan and hook inventories under `reference/`.
- Add deterministic-generation and freshness tests.
- Introduce explicit compatibility metadata in the registries and derive the compatibility matrix from it, replacing the interim curated table.

### Phase 3 — cookbook core (Track A)

- Write the engine selection guide with the (initially curated) compatibility matrix.
- Write installation, first reconstruction, and plan-validation pages.
- Publish the five minimal complete reconstructions using the example flow template, starting with a small simulated walkthrough and one redistributable experimental workflow.
- Write the decision-domain parameter pages, embedding generated facts.
- Write output/restart, performance, and troubleshooting pages.

### Phase 4 — architecture and extension guide (Track B)

- Document overview, lifecycle (including engine-boundary reshaping and initialization merge semantics), state and conventions, hook anatomy, and each hook-family page.
- Document observers, the JAX implementation guide, interfaces and the trust model, and extension testing.
- Include minimal importable custom-hook and custom-observer examples.

### Phase 5 — recipes, examples, and legacy retirement

- Publish the goal-oriented recipes and index all examples by metadata.
- Technically review any retained algorithm theory.
- Redirect or remove each legacy page (including `docs/api/state.md` and the empty `docs/using/metrics.md`) after its verified replacement is available.

### Phase 6 — concision pass and code-driven stub expansion

Runs after the first draft of each track exists (Phases 3–5 written the substantive pages;
some remain stubs). Two parallel efforts:

- **Concision pass** over every written page: apply the [writing-style rules](authoring-guide.md#writing-style-tighten-and-lead-with-the-answer) — lead with the answer, one lead sentence per table, cross-cutting context stated once, terse links, one-line verification notes, no process meta-commentary. Cut words only; facts, units, shapes, restrictions, and citations stay. This is a mechanical editorial pass with per-page word-count targets, not a rewrite, and must not change any cited claim.
- **Stub expansion from code**: draft each remaining stub from its named primary code source in the [stub-expansion table](#stub-expansion-from-code), then add curation. Facts needing a reconstruction to run are deferred and marked `Verification pending`. Where a fact class recurs across stubs, extend the generator rather than hand-copying.

Both efforts obey the existing verification discipline (strict build, `phaser validate` on
shown plans) and file-ownership rules. Neither is gated on the outstanding code-owner
blockers except where a specific page already was.

## Verification and release discipline

The documentation pipeline must:

- generate references from a clean directory;
- build with `zensical build --clean --strict`;
- detect broken links, missing navigation targets, and generation failures;
- validate every full YAML example with `phaser validate`;
- execute bounded reconstruction examples when CI dependencies permit;
- render representative mathematics, diagrams, code, admonitions, and light/dark presentation;
- check units, coordinates, shapes, expected output, heading order, contrast, and keyboard navigation during editorial review.

A user-facing plan field or built-in hook change requires its source metadata, generated reference, relevant example, tests, and release note to be considered together.

## Decision log

| Date | Decision |
| --- | --- |
| 2026-07-22 | Existing prose is explicitly non-authoritative until reverified. |
| 2026-07-22 | Documentation is split into scientist-first reconstruction and developer-extension paths. |
| 2026-07-22 | Zensical becomes the production builder immediately and is pinned while pre-1.0. |
| 2026-07-22 | Generated references are created during builds and are not committed. |
| 2026-07-22 | Legacy pages remain reachable with a prominent warning until replaced. |
| 2026-07-22 | The initial cutover removes Mike version selection; replacement versioning is future work. |
| 2026-07-22 | The IA is restructured into two tracks — a reconstruction cookbook/parameter track and an architecture/extension track — replacing the earlier six-section layout. The cookbook track leads in navigation. |
| 2026-07-22 | The parameter reference is hybrid: generated inventories are the factual source; curated decision-domain pages supply guidance and embed generated facts. |
| 2026-07-22 | The engine compatibility matrix starts curated and is later derived from explicit registry metadata, never inferred. |
| 2026-07-22 | Code findings from the documentation audit are tracked as verification blockers with code-owner disposition, not fixed within documentation work. Blockers 7, 8, and 11 gate their pages. |
| 2026-07-22 | Example pages use the ten-section flow template plus the example metadata table; the metadata drives a goal-oriented index. |
| 2026-07-22 | Plans are treated as code: trust warnings for expression schedules and external hooks are mandatory content wherever schedules or remote execution are documented. |
| 2026-07-23 | Concision is a first-class, review-enforced requirement with per-page length targets (authoring guide); a Phase 6 editorial pass tightens already-written pages without changing cited claims. |
| 2026-07-23 | Remaining stubs are expanded primarily by extracting from named code sources (example plans, error/warn strings, schema, docstrings) then adding curation; execution-dependent facts are deferred, not fabricated. |

## Maintainer sources

- `phaser/plan.py`
- `phaser/types.py`
- `phaser/hooks/`
- `phaser/execute.py`
- `phaser/observer.py`
- `phaser/state.py`
- `phaser/engines/`
- `phaser/cli/`
- `phaser/web/`
- `tests/`
- `examples/`
