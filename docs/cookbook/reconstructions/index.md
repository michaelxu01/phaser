# Minimal complete reconstructions

Each page here is a self-contained, portable, validated plan: the smallest complete
example that demonstrates one engine and data source combination. Every reconstruction
follows the [example flow template](../../design/authoring-guide.md#example-flow-template)
and records the same metadata table, so you can compare reconstructions by goal rather
than by specimen name.

## What's here

| Page | Engine | Data origin | Status |
| --- | --- | --- | --- |
| [Simulated single-slice reconstruction (gradient descent)](simulated-single-slice-gradient.md) | Gradient | Simulated | Written (smoke-tested) |
| [LSQML reconstruction (experimental PrScO₃)](lsqml.md) | LSQML | Experimental | Written |
| [Experimental reconstruction (Si, gradient descent)](empad-experimental.md) | Gradient | Experimental | Written |
| [Simulated multislice reconstruction (gradient descent)](simulated-multislice-gradient.md) | Gradient | Simulated | Stub |
| [ePIE reconstruction](epie.md) | ePIE | Simulated | Stub |

The two experimental pages use the downloadable `sample_data/` archive and the plans that
ship in `examples/` — download the data once (see [Your first
reconstruction](../../get-started/first-reconstruction.md#get-the-sample-data)).

## Suggested reading order

Start with the reconstruction closest to your own case:

- **New to Phaser:** the [simulated single-slice gradient](simulated-single-slice-gradient.md)
  example is the smallest, and runs in seconds on synthesized data with no download.
- **You have your own experimental detector data:** start with the [LSQML PrScO₃
  page](lsqml.md) — the recommended general-purpose engine — then the [Si gradient
  page](empad-experimental.md) if you need Poisson noise, cost regularizers, or the gradient
  engine's per-variable optimizers.

Use [Choosing a reconstruction engine](../engine-selection.md) first if you are not sure
which engine your data and goals call for. Every plan shown is validated with `phaser
validate`; the single-slice page is additionally executed end to end.

!!! note "Planned pages"
    The multislice-gradient and ePIE pages are still stubs; see each for its planned scope.
