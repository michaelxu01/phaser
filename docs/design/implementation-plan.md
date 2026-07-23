# Documentation implementation plan

| Field | Value |
| --- | --- |
| Status | Accepted |
| Audience | Documentation maintainers and implementing agents |
| Owner | Phaser maintainers |
| Adopted | 2026-07-22 |
| Depends on | [Documentation architecture](documentation-architecture.md), [implementation checklist](implementation-checklist.md) |

## Purpose

The [architecture document](documentation-architecture.md) defines *what* the documentation will be and the [checklist](implementation-checklist.md) tracks *whether* each piece is done. This plan defines *how* the work is executed: how it is cut into independent work packages, in what order, by whom (including autonomous agents working without shared context), and what every contribution must satisfy to merge.

Two priorities shape every decision in this plan:

1. **The documentation must serve code contributors.** The architecture/extension track is not an appendix — a contributor should be able to add a hook, solver, or observer using only the documentation, and the highest-leverage pages (hook anatomy, lifecycle, testing patterns) are scheduled early rather than last.
2. **The documentation system itself must be extensible and readable.** Adding a new built-in hook or plan field should propagate into the documentation through registry metadata and the generator, with the only manual step being curated guidance. The generator, templates, and metadata conventions are designed so a future contributor can add a page or a reference entry without reverse-engineering the build.

## Guiding principles

### Docs-as-code, code-as-docs-source

- Factual reference content is generated from `phaser/plan.py`, `phaser/types.py`, and the hook registries. When a contributor adds a hook, the generated inventory picks it up on the next build; missing descriptions or units surface as visible documentation debt, not silent omissions.
- Compatibility facts (engine × backend × noise model × variable) live as **explicit metadata on the registries**, introduced in WP-3. This is deliberately a code contribution: it makes compatibility machine-checkable and keeps the matrix from rotting. The generator must never infer compatibility from code structure.
- `scripts/generate_docs.py` is treated as reviewable library code, not a build script: small pure functions, no guessed defaults, deterministic output (tested by double-generation), and a failure — never a fallback — when introspection breaks.

### Predictability over cleverness

- Every page of a given kind uses its template exactly: the [example flow template](documentation-architecture.md#example-flow-template) for reconstructions and recipes, the [hook-family page template](documentation-architecture.md#hook-family-page-template) for hook pages, the [progressive-disclosure rules](documentation-architecture.md#accessibility-and-interpretability-rules) everywhere. Uniform structure is what lets many authors (human or agent) produce a coherent site — and lets reviewers check completeness mechanically.
- Terminology follows the schema (*regularizers*, not *regularizations*); the glossary is canonical and disputes are settled there, once.
- Every substantive page ends with a **Maintainer sources** section pointing at the code it was verified against.

### Small, independently verifiable units

- Work is cut so that one work package touches one directory (plus at most one designated shared file), can be reviewed in one sitting, and passes verification on its own: `phaser validate` for every complete YAML, `zensical build --clean --strict` for the site, smoke tests for bounded examples.
- Nothing merges on the strength of legacy prose. A claim is either verified against the [sources of truth](documentation-architecture.md#sources-of-truth) or not written.

## Work packages

Each work package (WP) is sized for a single focused contributor or agent session, owns a distinct set of files, and declares its dependencies. Checklist items map onto WPs; the checklist remains the tracking ledger.

### Dependency graph

```text
WP-0 (blocker triage)
  │
  ├──────────────► WP-1 (foundation: templates, glossary, landing pages, diagrams)
  │                  │
  │    ┌─────────────┼──────────────┬───────────────────────┐
  │    ▼             ▼              ▼                       ▼
  │  WP-2         WP-5a..e        WP-6                   WP-8a..d
  │  (generator)  (reconstr.)     (Track B core)         (observers, JAX,
  │    │             │              │                     interfaces, testing)
  │    ▼             │              ▼
  │  WP-3            │            WP-7a..k (hook-family pages)
  │  (compat         │
  │   metadata)      │
  │    │             │
  │    ▼             ▼
  │  WP-4a..j (parameter pages)   WP-9a..l (recipes)
  │    │             │              │
  └────┴─────────────┴──────────────┴────► WP-10 (perf + troubleshooting)
                                             │
                                             ▼
                                           WP-11 (legacy retirement)
```

Packages sharing a suffix letter range (e.g. WP-5a through WP-5e) are **mutually independent** and can run in parallel once their common prerequisite lands.

### WP-0 — Blocker triage *(serial; gates everything)*

- **Scope:** Verify blockers B1–B12 from the [checklist](implementation-checklist.md#blocker-triage-phase-1-do-first) against code; record outcomes; route B7, B8, B11 to code owners with a minimal reproduction or code citation each.
- **Owns:** checklist blocker section (outcome fields only).
- **Why first:** every later package consults blocker outcomes; unverified claims are the primary failure mode this whole design guards against.
- **Deliverable:** every blocker has a recorded outcome or an owner and a link to its disposition request.

### WP-1 — Foundation *(serial; unblocks all writing)*

- **Scope:** Home page routing; `get-started/` and `concepts/` stubs; the canonical glossary; both track landing pages; the authoring guide (templates, metadata table, admonition and warning conventions, Maintainer-sources footer format); initial diagrams 1–4 with text descriptions; link/accessibility checks in the build.
- **Owns:** `docs/index.md`, `docs/get-started/`, `docs/concepts/`, `docs/cookbook/index.md`, `docs/architecture/index.md`, the authoring guide, `mkdocs.yml` nav skeleton for the new tree.
- **Note:** WP-1 registers **all** planned pages in nav as stubs with scope statements. Later packages then fill files that already exist in nav — after WP-1, no other package edits `mkdocs.yml`, which eliminates the main merge-conflict surface.

### WP-2 — Reference generator *(parallel with WP-5/6/8 after WP-1)*

- **Scope:** Extend `scripts/generate_docs.py` to introspect plan classes and every hook registry into `docs/generated/plan/` and `docs/generated/hooks/`; determinism and freshness tests; a documented pattern for how curated pages embed generated facts (snippet include or anchor link — decide once, in this WP).
- **Owns:** `scripts/generate_docs.py`, generator tests, generator README section.
- **Contributor-extensibility requirement:** the generator's module docstring must explain, in under a page, how a new registry or plan class gets picked up — this is the document a future contributor reads when their new hook doesn't appear.

### WP-3 — Compatibility metadata *(after WP-2; code change)*

- **Scope:** Add explicit compatibility metadata to hook/engine registries (engines, backends, noise models, refinable variables, regularizer categories, restart); generate the compatibility matrix from it; retire the interim curated table in `cookbook/engine-selection.md` when it lands.
- **Owns:** registry metadata in `phaser/`, matrix generation, matrix section of `engine-selection.md`.
- **Depends on:** B6, B7, B11 outcomes; engine-maintainer review (this is a code contribution and follows normal code review).

### WP-4a–j — Decision-domain parameter pages *(parallel after WP-2)*

- **Scope:** One package per page under `cookbook/parameters/` (data-and-calibration, initialization, simulation-geometry, grouping-and-memory, noise-models, solvers-and-learning-rates, schedules-and-flags, regularization, termination-and-diagnostics, output-and-restart).
- **Each owns:** exactly one file.
- **Constraints:** every option states type, default, units, valid range, lifecycle stage, engines/backends, interactions, and a minimal example, embedding generated facts rather than restating them. Special obligations: initialization covers merge semantics and `{}` (B-outcome-verified); grouping-and-memory documents all three states of `buffer_n_groups`; schedules-and-flags carries the mandatory trust warning; solvers waits on B7/B8 for position-solver content.

### WP-5a–e — Minimal complete reconstructions *(parallel after WP-1)*

- **Scope:** One package per reconstruction (simulated single-slice gradient, simulated multislice gradient, EMPAD experimental, ePIE, LSQML) under `cookbook/reconstructions/`, each using the full example flow template with recorded metadata, `phaser validate` output, and a smoke test where bounded.
- **Each owns:** one page plus its example plan file.
- **Note:** these do not depend on WP-2 — a complete runnable plan is verified by `phaser validate`, not by the generated reference. EMPAD depends on B10 (portable dataset).

### WP-6 — Track B core *(serial within itself; parallel with WP-4/5)*

- **Scope:** `architecture/overview.md` (including the extension-mechanism comparison table), `architecture/lifecycle.md` (including engine-boundary reshaping and initialization merge), `architecture/state-and-conventions.md`, and `architecture/hooks/index.md` (hook anatomy: registries, lazy resolution, built-in vs. external hooks, validation scope).
- **Owns:** those four files.
- **Why one package:** these pages share one mental model of the runtime and must be written by one author against `phaser/execute.py`, `phaser/state.py`, and `phaser/hooks/hook.py` to stay consistent. They are the pages every other Track B author reads first.

### WP-7a–k — Hook-family pages *(parallel after WP-6)*

- **Scope:** One package per family: raw-data-loaders, initialization, post-load, post-init, schedules-and-flags, noise-models, solvers, cost-regularizers, group-constraints, iteration-constraints, engines.
- **Each owns:** one file under `architecture/hooks/`, following the hook-family template exactly, including a minimal custom implementation and its testing pattern.
- **Contributor emphasis:** the "minimal custom implementation" in each page is the primary extension documentation for that surface. It must be complete enough to copy, import, and test — reviewed by actually running it.

### WP-8a–d — Observers, JAX, interfaces, testing *(parallel after WP-6)*

- **Scope:** Four packages: `architecture/observers.md` (with runnable custom-observer example), `architecture/jax.md`, `architecture/interfaces.md` (CLI, Python API, web manager, workers, trust model), `architecture/testing.md` (extension testing patterns, public API policy, example-demonstrated vs. test-covered labeling per B12).
- **Each owns:** one file (plus example code imported by tests for observers/testing).

### WP-9a–l — Goal-oriented recipes *(parallel after WP-5 and WP-6)*

- **Scope:** One package per recipe listed in the checklist (coarse-to-fine through Optuna sweeps). Recipes reference the minimal reconstructions as starting points and link to the architecture pages that explain their machinery. Position-refinement is gated on B7/B8.
- **Each owns:** one file under `cookbook/recipes/` plus any example plan.

### WP-10 — Performance and troubleshooting *(after WP-4, WP-5)*

- **Scope:** `cookbook/performance.md` and `cookbook/troubleshooting.md`. Written late deliberately: they harvest the failure modes and tuning notes recorded by every earlier example and recipe package.
- **Owns:** two files.

### WP-11 — Legacy retirement *(last)*

- **Scope:** Per-page retirement of `docs/using/` and `docs/api/state.md` once each replacement is verified; redirects or removal notes; final nav cleanup; final full-site editorial pass against the accessibility rules.
- **Owns:** legacy files, final `mkdocs.yml` state.

## Task distribution for implementing agents

Any WP can be assigned to an autonomous agent. Agents start with no context, so every assignment uses the briefing template below; a briefing that cannot be filled in completely means the WP's prerequisites are not actually met.

### Agent briefing template

```text
TASK: <WP id and one-sentence deliverable>

READ FIRST (in order):
  1. docs/design/documentation-architecture.md   — authoritative design; obey its
     templates, accessibility rules, and terminology decisions
  2. docs/design/implementation-checklist.md     — your items and blocker outcomes
  3. <the specific template section for this page kind>
  4. <WP-specific sources of truth, e.g. phaser/execute.py, tests/test_initialization.py>

BLOCKER OUTCOMES THAT BIND YOU: <e.g. "B6: Poisson is gradient-only — state this
  in your compatibility section"; "B8 unresolved — omit position solvers">

YOU OWN (create/edit ONLY these):
  <explicit file list; never edit mkdocs.yml, the glossary, other packages' pages,
   or any file under phaser/ unless the WP says so>

MUST HOLD ON COMPLETION:
  - every factual claim verified against the listed sources, never against
    docs/using/* (legacy, non-authoritative)
  - page follows its template with no sections omitted or reordered
  - every YAML plan passes `phaser validate`; note the command and output
  - `zensical build --clean --strict` passes (run scripts/generate_docs.py first)
  - page ends with a Maintainer sources section
  - checklist item updated with evidence in the Done log

REPORT BACK: files changed, verification output, any claim you could NOT verify
  (left out of the page and flagged), any new blocker discovered.
```

### Rules that keep parallel work safe

1. **Exclusive file ownership.** A WP edits only the files it owns. After WP-1, `mkdocs.yml` and the glossary are frozen for everyone except designated updates: an agent needing a nav change or a new glossary term **requests it in its report** instead of editing, and the integrator applies it. This is the single most important conflict-avoidance rule.
2. **Isolation.** Agents work in a worktree or branch per WP; the integrator merges in dependency order. Parallel WPs never share files, so merges are trivial by construction.
3. **Unverifiable means absent.** An agent that cannot verify a claim omits it and flags it — it never writes the claim with a hedge, and never fills gaps from legacy prose or from its own general knowledge of ptychography software.
4. **Discoveries route to the ledger.** An agent that finds a new code inconsistency reports it as a candidate blocker; it does not fix code (outside WP-3), and does not expand its own scope.
5. **Templates are contracts.** Reviewers check template sections mechanically before reading prose. A missing "Failure modes" section is a rejection, not a note.

### Suggested waves

| Wave | Packages | Parallelism |
| --- | --- | --- |
| 1 | WP-0 | 1 agent |
| 2 | WP-1 | 1 agent |
| 3 | WP-2, WP-5a–e, WP-6 | up to 7 agents |
| 4 | WP-3, WP-4a–j, WP-7a–k, WP-8a–d | up to 26 agents; WP-4 starts after WP-2 |
| 5 | WP-9a–l | up to 12 agents |
| 6 | WP-10 | 1–2 agents |
| 7 | WP-11 | 1 agent |

The integrator (a maintainer, or one long-lived coordinating agent) merges each wave, applies requested nav/glossary changes, runs the full verification pipeline, and updates the checklist Done log before launching the next wave. Waves are a ceiling, not a quota — packages within a wave can also land one at a time.

## Definition of done

**Per page:**

- Template-complete for its page kind; progressive-disclosure order respected.
- All claims verified against sources of truth; Maintainer sources section present.
- YAML validated; bounded examples smoke-tested; metadata recorded for examples.
- Units beside every physical parameter; shapes and axis order stated explicitly; restrictions (engine/backend/experimental) visibly marked.
- Checklist item checked with evidence in the Done log.

**Per wave:**

- `scripts/generate_docs.py` from clean, then `zensical build --clean --strict`, link check, and accessibility check all pass on the merged result.
- No package edited files outside its ownership; nav and glossary changes went through the integrator.

**For the project:**

- All checklist items checked; all blockers dispositioned; legacy pages retired or redirected; a contributor can add a built-in hook and see it appear in the generated reference with only curated guidance left to write.

## Maintainer sources

- `docs/design/documentation-architecture.md`
- `docs/design/implementation-checklist.md`
- `scripts/generate_docs.py`
- `mkdocs.yml`
- `.github/workflows/docs.yaml`
