phaser: The weapon of choice for ptychographic reconstructions
---

`phaser` is a fully-featured package for multislice electron ptychography.

## Features

- ePIE, LSQ-MLs, and gradient descent algorithms
- [`numpy`][numpy], [`cupy`][cupy], and [`jax`][jax] backends.
- Single and multislice ptychography
- Multiple incoherent probe modes
- Probe position correction
- Upsampled (sPIE) and segmented ptychography (work in progress)
- Adaptive propagator correction (work in progress)

## Installation

To install, first clone the repository from github:

```sh
$ git clone https://github.com/hexane360/phaser
# enter phaser directory
$ cd phaser
```

If you're using [`cupy`][cupy] or [`jax`][jax] with a GPU, follow the installation instructions for those packages.

Then, install with `pip`:

```sh
$ python -m pip install -e .
```

For the jax or cupy backend, or for the optional webserver, install with the corresponding options:

```sh
$ python -m pip install -e '.[cupy,web]' # or '.[jax,web]'
```

## Running

After installation, the `phaser` command should be available. Phaser can be run from the command line, or through a job server.

To run a single reconstruction on the command line, call `phaser run <file>`, where `file` is the path to a reconstruction plan file.

To run the webserver, call `phaser serve`. By default, the server serves on https://localhost:5050/, so navigate there in a web browser. The server interface can be used to start workers and schedule reconstruction jobs.

To run a worker, call `phaser worker <url>`, where `url` is the URL of a running job server.

## Alternatives

Other notable ptychography packages:

 - [`fold_slice`](https://github.com/yijiang1/fold_slice) branch of [PtychoShelves](https://www.psi.ch/en/sls/csaxs/software#coming-soon-ptychoshelves-a-versatile-high-level-framework-for-high-performance-analysis-of)
 - [`py4DSTEM`](https://github.com/py4dstem/py4DSTEM)
 - [`PtyLab.m/py/jl`](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026)
 - [PtyPy](https://ptycho.github.io/ptypy/)

[numpy]: https://numpy.org/
[cupy]: https://cupy.dev/
[jax]: https://docs.jax.dev/en/latest/