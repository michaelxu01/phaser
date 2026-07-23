# Authoring guide

| Field | Value |
| --- | --- |
| Status | Accepted |
| Audience | Anyone writing or reviewing a documentation page, human or agent |
| Owner | Phaser maintainers |
| Depends on | [Documentation architecture](documentation-architecture.md), [implementation plan](implementation-plan.md) |

This page is the practical companion to the [documentation architecture](documentation-architecture.md):
it reproduces the page templates, states the admonition and footer conventions every page
must use, and lists the commands that verify a page before it is considered done. If this
page and the architecture document ever disagree, the architecture document wins; report
the discrepancy so this page can be corrected.

## Before you write a page

1. Identify the page kind (landing page, example flow, hook-family page, decision-domain
   parameter page, or free-form guide) and use the matching template below.
2. Read the page's listed sources in the [implementation checklist](implementation-checklist.md)
   and, for terminology, the [glossary](../concepts/glossary.md). Verify every factual claim
   against the [sources of truth](documentation-architecture.md#sources-of-truth) — never
   against `docs/using/*`, which is legacy and non-authoritative.
3. If a claim cannot be verified, omit it and flag it in your report. Never hedge it into
   the page and never fill the gap from general ptychography knowledge.
4. Check whether the claim is constrained by an open item in the
   [verification blockers table](documentation-architecture.md#verification-blockers) or
   the checklist's blocker-triage section. If a blocker is unresolved and gates the page
   (currently B7, B8, B11), do not write that section at all.

## Writing style: tighten and lead with the answer

Pages must be scannable. A reader should reach the fact they came for without wading
through prose that restates the table beside it. These rules are enforced in review with
the same weight as the templates.

1. **Lead with the answer.** The first sentence of a page or section states the outcome,
   recommendation, or definition. Background, motivation, and caveats come after — never
   before.
2. **One lead sentence per table, not a paragraph.** Introduce a table with a single line,
   then let the cells carry the detail. Do not narrate in prose what a cell already states.
   If a per-item note is long, it belongs in the cell (or a footnote), not in the
   surrounding text.
3. **Say cross-cutting context once.** Page-wide framing — for example "types and defaults
   come from the [generated reference]; this page adds units and meaning" — goes in the
   page intro a single time. Do not repeat it under each heading.
4. **Link tersely.** Write `[descriptive text](url)` and stop. Drop trailers like "see X
   for the full explanation and why it matters" — the link *is* the pointer, and "why it
   matters" is padding.
5. **Verification and pending notes are one line.** For example: *Not re-run for this page;
   validate before relying on exact syntax.* No multi-sentence justification.
6. **Cut process meta-commentary.** Do not narrate the documentation effort inside a page
   ("this is a candidate code issue reported alongside this page, not something to work
   around"). State the fact; the `!!! warning "Restriction"` admonition already flags it.
7. **Prefer the tightest structure that still carries every required field.** A compact
   table beats prose that enumerates the same fields sentence by sentence. Required page
   fields (units, shapes, lifecycle stage, restrictions) are non-negotiable — cut words,
   never facts or citations.

Soft length targets (a page much over these is usually restating itself, not covering
more): decision-domain parameter and hook-family pages **≈ 800–1200 words**; landing pages
**< 400**; a single option or built-in entry **≤ 4 lines** including its example.

**Before / after** — a table intro rewritten to rule 2 and rule 4:

> *Before (61 words):* "Every property's type and default is generated in the Raw Data hook
> reference; this section adds units and practical meaning. Lifecycle stage for every
> property below is the same: read once, at raw-data loading, before any reconstruction
> state exists. The table below lists the properties shared across the built-in loaders,
> with their units and what each one is for."
>
> *After (24 words):* "Shared loader properties, all read once at raw-data loading. Types
> and defaults are in the [Raw Data reference](../../generated/hooks/raw-data.md); units
> and meaning below."

## Page templates

### Example flow template

Used by every minimal complete reconstruction (`cookbook/reconstructions/`) and every
goal-oriented recipe (`cookbook/recipes/`). Sections, in this order, with none omitted or
reordered:

1. **Goal** — what this reconstruction demonstrates.
2. **When to use it** — supported data and scientific assumptions.
3. **Compatibility** — engine, backend, noise model, and optimization variables.
4. **Input contract** — shapes, units, coordinate conventions, and normalization.
5. **Complete plan** — runnable YAML or Python.
6. **Execution flow** — loader → initialization → preprocessing → engine stages → output.
7. **Parameter walkthrough** — only parameters that matter for this flow, explaining
   every non-default option.
8. **Expected result** — metrics, images, state files, convergence behavior, and a basic
   success check.
9. **Variations** — small changes for related use cases.
10. **Failure modes** — symptoms, likely causes, and fixes.

Every example page also carries this metadata (usually as a table directly under the
title, before "Goal"); the metadata additionally drives the goal-oriented example index:

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

Repository examples under `examples/` are evidence, not automatically tutorials. Replace
machine-specific paths and unavailable datasets before promoting one into a complete
workflow. Validate every full YAML example with `phaser validate`; add an execution smoke
test for bounded examples. Label examples honestly as example-demonstrated versus
test-covered (see blocker B12) — do not imply engine-correctness test coverage that does
not exist.

### Hook-family page template

Used by every page under `architecture/hooks/` except `index.md` (which documents hook
anatomy in general, not one family). Sections, in this order:

1. **Lifecycle point** — when the hook runs and what has already happened.
2. **Callable signature and property schema.**
3. **Accepted state/input and returned value.**
4. **Built-in implementations** — linked to the generated inventory once it exists.
5. **A minimal custom implementation** — complete enough to copy, import, and test; this
   is the primary extension documentation for that surface, so it should be reviewed by
   actually running it.
6. **YAML invocation** — for both built-in short names and external `package.module:function`
   hooks, noting that external hook properties are not schema-validated.
7. **Engine/backend restrictions.**
8. **Optional dependencies.**
9. **A testing pattern** for custom implementations.

### Decision-domain parameter page template

Used by every page under `cookbook/parameters/` except `index.md`. There is no fixed
section list (each domain groups a different set of options), but every documented
option — however the page organizes them — must state all of the following, and a page
is incomplete if any option is missing one:

- **Type** and **default** (embed from the generated reference; never restate by hand).
- **Units**, where the option is physical.
- **Valid range** in practice, beyond what the type alone expresses.
- **Lifecycle stage** — when in the reconstruction this option takes effect.
- **Supported engines/backends** — visibly marked if restricted.
- **Interactions** with other options in this or another domain.
- **A minimal example.**

A stub for one of these pages carries this skeleton instead of the option table:

```markdown
## Options in this domain

## Type, default, and units

## Valid ranges and lifecycle stage

## Supported engines and backends

## Interactions

## Minimal example
```

### Landing pages

Every section and subsection index (`docs/index.md`, `get-started/index.md`,
`cookbook/index.md`, `cookbook/reconstructions/index.md`, `cookbook/recipes/index.md`,
`cookbook/parameters/index.md`, `architecture/index.md`, `architecture/hooks/index.md`)
gets real content, not a stub: what the section contains, a suggested reading order, and
links to every page in it. Landing pages are exempt from the templates above but must
still follow the accessibility rules and terminology rules on this page.

### Stub pages

Every other planned page in the target information architecture that has not yet been
written is created as a stub, so that navigation and cross-links are stable from the
start of the migration. A stub has:

- a title;
- a one-paragraph scope statement describing what the page will cover;
- the `!!! note "Planned page"` admonition (see below);
- if the page is an example flow, hook-family, or decision-domain parameter page, the
  empty section-heading skeleton from the matching template above, with no body text
  under any heading.

A stub does not carry a Maintainer sources section — it makes no factual claims yet.

## Admonition conventions

Use exactly these admonition types; do not invent new ones for the same purpose.

| Purpose | Admonition | Example |
| --- | --- | --- |
| A page is planned but not yet written | `!!! note "Planned page"` | see stub template above |
| A page is legacy content that may be out of date | `!!! warning "Legacy documentation"` | used on every page under `docs/using/` and on `docs/index.md`'s migration notice |
| An option, feature, or example is restricted to specific engines, backends, or is experimental | `!!! warning "Restriction"` | "Restriction: the gradient engine requires the `jax` or `torch` backend." |
| Expression schedules or external hooks that execute arbitrary code (the trust model) | `!!! danger "Trust warning"` | see [Trust model and warning conventions](#trust-model-and-warning-conventions) |

Restriction and trust admonitions must never be the only place a restriction is stated —
also say it in prose per the [accessibility rules](documentation-architecture.md#accessibility-and-interpretability-rules),
since admonitions alone are not sufficiently discoverable for every reader and meaning
must never rely on color or icon alone.

## Trust model and warning conventions

Expression schedules use unrestricted Python `eval` (`phaser/hooks/schedule.py`), and
external hooks (`"package.module:function"` references) execute arbitrary importable
code. A plan file is therefore equivalent to a script: running an untrusted plan is
running untrusted code. Every page that documents expression schedules or external hooks
carries a `!!! danger "Trust warning"` admonition and adjacent prose, for example:

```markdown
!!! danger "Trust warning"
    Expression schedules evaluate arbitrary Python via `eval` (`phaser/hooks/schedule.py`).
    A plan file that uses one is equivalent to a script. Only run plans you trust.
```

`architecture/interfaces.md` additionally documents the trust implications for the web
manager and local/Slurm workers: anyone who can submit a plan to a worker can execute
code as that worker. Never present expression schedules in beginner material without this
warning.

## Terminology rules

- Use *regularizer*, never *regularizations* or *regularization* as a noun for the hook
  family — the schema uses *regularizer* (`phaser/hooks/regularization.py`). "Regularization"
  is acceptable only as the general scientific concept, not as the name of a hook or page.
- Define a technical term at first use on a page and link to the
  [glossary](../concepts/glossary.md) rather than redefining it locally. If a term you
  need is missing from the glossary, request it in your report — do not add it yourself;
  the glossary is frozen to the same single-owner rule as `mkdocs.yml` after WP-1.
- State physical units beside every physical parameter, and coordinate order and array
  shape explicitly (e.g. `(y, x)`, `(modes, y, x)`, `(slices, y, x)`).
- Separate required, commonly adjusted, and advanced parameters rather than listing them
  in declaration order.
- Avoid unexplained use of *backend*, *JIT*, *registry*, *protocol*, and *serialization* —
  link to the glossary the first time any of these appears on a page.
- Use descriptive link text, never "click here."

## Maintainer sources footer format

Every substantive page (i.e. every page that makes a factual claim — this excludes
stubs) ends with a level-2 heading and a flat bullet list of the files it was verified
against, most-specific first:

```markdown
## Maintainer sources

- `phaser/plan.py`
- `phaser/execute.py`
```

List only files actually consulted for this page's claims, not the whole sources-of-truth
list from the architecture document. Do not cite `docs/using/*` here — it is not a source,
even if it originally suggested the topic.

## Verification commands

Run both before considering any page (or wave of pages) done. Both must exit
successfully; treat any failure as blocking, not advisory.

```console
python scripts/generate_docs.py
zensical build --clean --strict
```

`scripts/generate_docs.py` must be run first, from a clean `docs/generated/` (it removes
and recreates that directory itself) — the build depends on its output and is not
responsible for regenerating it. `zensical build --clean --strict` fails the build on
broken internal links, pages missing from `mkdocs.yml` navigation, and other structural
problems; treat a strict-mode failure as a specification defect in the page, not in the
build.

For pages with runnable examples, additionally run:

```console
phaser validate path/to/plan.yaml
```

for every complete YAML plan shown, and record the command and its output (or a short
transcript) near the example. Bounded examples (small enough to run in CI) additionally
get an execution smoke test; document what "bounded" means for that example (shape,
iteration count, expected wall-clock time) so a future maintainer can judge whether it is
still bounded.

## Diagrams

Diagrams are authored as Mermaid fences (`` ```mermaid ``), each with adjacent prose that
states the same information in words — diagrams are never the only place a fact appears,
per the accessibility rules. Do not put exact option lists in a diagram; those come from
the generated reference and drift if hand-copied into a picture.

```markdown
​```mermaid
graph TD
    A[Step] --> B[Next step]
​```

One sentence per diagram explaining, in prose, what the diagram shows.
```

Mermaid rendering in the pinned Zensical build was verified during WP-1 by building a
throwaway page and inspecting the generated HTML; see the WP-1 report for the evidence
and the one required `mkdocs.yml` configuration change
(`markdown_extensions.pymdownx.superfences.custom_fences`, needed for a ` ```mermaid ` fence
to become a `<pre class="mermaid">` block instead of a plain highlighted code block).

## Maintainer sources

- `docs/design/documentation-architecture.md`
- `docs/design/implementation-plan.md`
- `mkdocs.yml`
- `scripts/generate_docs.py`
