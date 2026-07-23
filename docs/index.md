# Phaser

Phaser is a Python package for reconstructing electron ptychography data, including
multislice and mixed-state models. A reconstruction is described by a validated **plan**:
how to load raw data, how to initialize the object, probe, scan, and (optionally) tilt,
and which reconstruction engines to run, in what order.

!!! warning "Documentation migration in progress"

    This site is being rebuilt around two tracks — a reconstruction cookbook for
    scientists and an architecture guide for contributors — verified page by page
    against the current plan schema, hook registries, runtime implementation, and
    tests. Pages under **Legacy documentation** remain reachable for topic discovery
    but may describe behavior that has changed; do not treat them as authoritative.
    See the [documentation architecture](design/documentation-architecture.md) for the
    full plan and the [implementation checklist](design/implementation-checklist.md)
    for what has been verified so far.

## Two tracks

**[Cookbook](cookbook/index.md)** — for scientists preparing data, configuring a
reconstruction, inspecting results, and troubleshooting. Start here if your goal is a
validated reconstruction; you should not need to understand Python inheritance,
registries, or package imports to use it.

**[Architecture](architecture/index.md)** — for contributors and researchers extending
Phaser: the reconstruction lifecycle, state and conventions, the hook system, observers,
the JAX implementation, interfaces, and testing patterns.

Both tracks share the [Get started](get-started/index.md) pages (installation, a first
reconstruction, and plan validation) and the [Concepts](concepts/glossary.md) pages
(scientific background and the canonical glossary). Every cookbook page links to the
architecture pages that explain its machinery, and every hook-family page links to
recipes that use it.

## Current project interfaces

The packaged command-line entry point is `phaser`:

```console
phaser validate path/to/plan.yaml
phaser run path/to/plan.yaml
```

## Where things live right now

- **New to Phaser:** [Get started](get-started/index.md).
- **Running or troubleshooting a reconstruction:** [Cookbook](cookbook/index.md).
- **Extending Phaser or understanding its internals:** [Architecture](architecture/index.md).
- **Contributing to this documentation:** the accepted
  [documentation architecture](design/documentation-architecture.md), the
  [implementation plan](design/implementation-plan.md), the
  [implementation checklist](design/implementation-checklist.md), and the
  [authoring guide](design/authoring-guide.md).
- **Not yet migrated:** the pages under **Legacy documentation** in the navigation, and
  the build-generated [`phaser.state` reference](generated/api/state.md) under
  **Reference**.

## Maintainer sources

- `phaser/cli/`
- `phaser/plan.py`
- `phaser/execute.py`
