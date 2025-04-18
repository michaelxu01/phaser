
from pathlib import Path
import re
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import Sampling
from phaser.utils.physics import Electron
from phaser.io.empad import load_4d, EmpadMetadata
from .. import LoadEmpadProps, RawData


def load_empad(args: None, props: LoadEmpadProps) -> RawData:
    path = Path(props.path).expanduser()

    if path.suffix.lower() == '.json':  # load as metadata
        meta = EmpadMetadata.from_json(path)
        assert meta.path is not None

        path = meta.path / meta.raw_filename

        voltage = props.kv * 1e3 if props.kv is not None else meta.voltage
        diff_step = props.diff_step or meta.diff_step
        scan_shape = meta.scan_shape

        probe_hook = {
            'type': 'focused',
            'conv_angle': meta.conv_angle,
            'defocus': meta.defocus * 1e10 if meta.defocus is not None else None,
        }
        # TODO: handle explicit scan_positions here
        scan_hook = {
            'type': 'raster',
            # [x, y] -> [y, x]
            'shape': tuple(reversed(meta.scan_shape)),
            'step_size': tuple(s*1e10 for s in reversed(meta.scan_step)),  # m to A
            'affine': meta.scan_correction[::-1, ::-1] if meta.scan_correction is not None else None,
        }

    else:
        voltage = props.kv * 1e3 if props.kv is not None else None
        diff_step = props.diff_step
        scan_shape = None
        probe_hook = scan_hook = None

    if voltage is None:
        raise ValueError("'kv'/'voltage' must be specified by metadata or passed to 'raw_data'")
    if diff_step is None:
        raise ValueError("'diff_step' must be specified by metadata or passed to 'raw_data'")

    wavelength = Electron(voltage).wavelength

    a = wavelength / (diff_step * 1e-3)  # recip. pixel size -> 1 / real space extent
    sampling = Sampling((128, 128), extent=(a, a))

    if not path.exists():
        raise ValueError(f"Couldn't find raw data at path {path}")

    patterns = numpy.fft.ifftshift(load_4d(path, scan_shape, memmap=True), axes=(-1, -2))

    mask = numpy.zeros_like(patterns, shape=patterns.shape[-2:])
    mask[2:-2, 2:-2] = 1.

    return {
        'patterns': patterns,
        'mask': numpy.fft.ifftshift(mask, axes=(-1, -2)),
        'sampling': sampling,
        'wavelength': wavelength,
        'probe_hook': probe_hook,
        'scan_hook': scan_hook,
        'seed': None,
    }