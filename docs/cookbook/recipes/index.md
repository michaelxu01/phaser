# Recipes

Recipes are goal-oriented guides for making one specific change to a reconstruction.
Each recipe starts from one of the [minimal complete reconstructions](../reconstructions/index.md)
and follows the same [example flow template](../../design/authoring-guide.md#example-flow-template)
used there, so recipes and reconstructions can be compared directly. Every recipe links
to the [architecture](../../architecture/index.md) pages that explain the machinery it
relies on.

## What's here

- **[Coarse-to-fine reconstruction](coarse-to-fine.md)**
- **[Conventional engine into gradient descent](conventional-to-gradient.md)**
- **[Increasing resolution between engines](increasing-resolution.md)**
- **[Changing slice count between engines](changing-slices-between-engines.md)**
- **[Adding mixed-state probe modes](mixed-state-probe-modes.md)**
- **[Position refinement](position-refinement.md)**
- **[Tilt refinement](tilt-refinement.md)**
- **[Restarting from a saved state](restart-from-hdf5.md)**
- **[Restarting with an overridden component](restart-overriding-a-component.md)**
- **[Using cost regularizers](cost-regularizers.md)**
- **[Using group and iteration constraints](constraints.md)**
- **[Hyperparameter sweeps with Optuna](optuna-sweeps.md)**

## Suggested reading order

There is no single order — pick the recipe matching the change you want to make. Readers
new to multi-engine plans should read
[Conventional engine into gradient descent](conventional-to-gradient.md) or
[Coarse-to-fine reconstruction](coarse-to-fine.md) first, since later recipes
(increasing resolution, changing slice count, adding probe modes) build on the same
engine-transition mechanics.

!!! note "Planned pages"
    All twelve recipes are stubs as of this writing; see each page for its planned scope.
    [Position refinement](position-refinement.md) additionally depends on blockers B7 and
    B8 in the implementation checklist.
