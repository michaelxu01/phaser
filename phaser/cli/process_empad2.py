
from pathlib import Path
from dataclasses import dataclass
import time
import typing as t

import click
from click.exceptions import Exit
import pane
from rich.prompt import Prompt
from rich.console import Console

from phaser.io.empad import EmpadMetadata, UnscannedDataset
from phaser.utils.config import Config
from phaser.utils.misc import natural_key
from .process_empad import (
    Parameter, OptStringParam, FloatParam, CameraParam, AduParam,
    default_console
)


class ProcessEmpad2Config(pane.PaneBase):
    diff_step: t.Dict[float, t.Tuple[float, float]] = {}
    """
    Diffraction pixel size calibration, given as a mapping kV: [camera_length, diff_step].
    'camera_length' is given in meters, and diff_step in mrad/pixel. At a given kV, the
    pixel size is interpolated linearly to get different camera lengths.
    E.g.:
    diff_step:
      300: [0.5766, 0.5]  # 300 kV: 576.6 mm camera length, 0.5 mrad/px
    """

    adu: t.Optional[t.Tuple[float, float]] = None
    """
    ADU (single electron intensity) calibration.
    Give a tuple of `[kV, adu]`. Other kVs assume linear scaling.
    """

    det_flips: t.Tuple[bool, bool, bool] = (False, False, False)
    """Detector flips (flip_y, flip_x, transpose). Set and forget"""

    det_rotation: float = 0.0
    """Detector rotation relative to scan axes (degrees)."""

    probe_current: float = 30.0
    """Default probe current (pA)"""

    conv_angle: float = 25.0
    """Default convergence angle (mrad). Used only if missing from metadata."""


@dataclass
class ConvAngleParam(FloatParam):
    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> float:
        if metadata.conv_angle is not None:
            # prefer value read from the microscope
            self.default_val = metadata.conv_angle
            self.default_str = f"{metadata.conv_angle:.2f}"
        return super().ask(metadata, console)


@dataclass
class AuthorParam(OptStringParam):
    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> t.Optional[str]:
        if metadata.author is not None:
            self.default_val = metadata.author
        return super().ask(metadata, console)


def make_params(config: ProcessEmpad2Config) -> t.Sequence[t.Tuple[str, Parameter]]:
    return [
        ('conv_angle', ConvAngleParam("Convergence angle", config.conv_angle, "mrad", f"{config.conv_angle:.1f}")),
        ('defocus', FloatParam("Defocus (CW is +)", 0.0, "nm", "0.0", 1e-9)), # nm to m
        ('beam_current', FloatParam("Approx. probe current", config.probe_current, "pA", f"{config.probe_current:.1f}", 1e-12)),  # pA to A
        ('diff_step', CameraParam(config.diff_step)),
        ('adu', AduParam(config.adu[1], config.adu[0] * 1e3) if config.adu is not None else AduParam()),
        ('author', AuthorParam("Author", None)),
    ]


def process_dir(path: t.Union[str, Path], config: ProcessEmpad2Config,
                params: t.Sequence[t.Tuple[str, Parameter]],
                console: Console, prompt: bool = True):
    path = Path(path)

    for metadata_path in sorted(path.rglob('metadata.toml'), key=natural_key):
        output_path = metadata_path.with_suffix('.json')
        if output_path.exists():
            # skip already created
            continue

        try:
            metadata = EmpadMetadata.from_toml(metadata_path)
            console.print(f"Loaded '{metadata_path}'.")
        except UnscannedDataset:
            console.print(f"[info]Skipping metadata '{metadata_path}' (not a 4D STEM dataset).")
            continue
        except Exception:
            console.print(f"[error]Couldn't parse TOML metadata '{metadata_path}'. Skipping.")
            console.print_exception()
            continue

        if not (metadata_path.parent / metadata.raw_filename).exists():
            console.print(f"[warning]Couldn't find raw data '{metadata.raw_filename}' in '{metadata_path.parent}'.")

        metadata.det_rotation = config.det_rotation
        metadata.det_flips = config.det_flips

        if prompt:
            if metadata.notes is not None:
                console.print(f"Notes: {metadata.notes}", markup=False)

            for (k, param) in params:
                val = param.ask(metadata, console)
                setattr(metadata, k, val)

            metadata.notes = Prompt.ask("Notes", default=metadata.notes, console=console)

        metadata.write_json(output_path, indent=4)
        console.print(f"Wrote metadata to '{output_path}'.")

    console.print("Processed all files.")


@click.command()
@click.option('--watch/--no-watch', default=False, help="Watch folder for changes")
@click.option('--prompt/--no-prompt', default=True, help="Prompt for additional metadata.")
@click.option('--make-config', is_flag=True, default=False, help="Make config file and exit")
@click.argument('folder', type=click.Path(exists=True, file_okay=False), required=False)
def process_empad2(
    folder: t.Union[str, Path, None],
    watch: bool = False,
    prompt: bool = True,
    make_config: bool = False,
):
    """
    Process EMPAD2 TOML metadata for all the datasets contained in FOLDER.
    """
    console = default_console()
    config = Config('process_empad2', ProcessEmpad2Config)

    if make_config:
        if not config.write_default():
            # failed to write
            console.print(f"[error]Config file already exists at '{config.path()}'.")
            raise Exit(1)
        console.print("Wrote config file!")
        return

    config = config.get()
    params = make_params(config)

    if folder is None:
        folder = Path('.')
    path = Path(folder).resolve()

    console.print(f"Processing dir '{path}'")
    process_dir(path, config, params, console, prompt)

    if not watch:
        return

    while True:
        time.sleep(2.)
        process_dir(path, config, params, console, prompt)
