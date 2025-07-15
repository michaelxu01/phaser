
from pathlib import Path
import warnings
import logging
import typing as t

import numpy

from phaser.utils.num import Sampling
from phaser.utils.physics import Electron
from phaser.io.gatan import load_4d, GatanMetadata
from phaser.types import cast_length
from rsciio import digitalmicrograph as dm
from .. import LoadGatanProps, RawData


def load_gatan(args: None, props: LoadGatanProps) -> RawData:
    logger = logging.getLogger(__name__)

    path = Path(props.path).expanduser()

    # file = dm.file_reader(path, lazy=True)

    # if path.suffix.lower() == '.json':  # load as metadata
    #     pass # stub for now, not implemented

    # else: # grab from dm4 file
    metadata = GatanMetadata.from_dm4(path)
    
    assert metadata.path is not None

    # path = metadata.path / metadata.gatan_filename

    voltage = props.kv * 1e3 if props.kv is not None else metadata.voltage
    diff_step = props.diff_step or metadata.diff_step
    scan_shape = metadata.scan_shape
    adu = 1 #props.adu or meta.adu
    needs_scale = not metadata.is_simulated()

    probe_hook = {
        'type': 'focused',
        'conv_angle': metadata.conv_angle,
        'defocus': metadata.defocus * 1e10 if metadata.defocus is not None else None,
    }
    # TODO: handle explicit scan_positions here
    scan_hook = {
        'type': 'raster',
        # [x, y] -> [y, x]
        'shape': tuple(reversed(metadata.scan_shape)),
        'step_size': tuple(s*1e10 for s in reversed(metadata.scan_step)),  # m to A
        'affine': metadata.scan_correction[::-1, ::-1] if metadata.scan_correction is not None else None,
    }

# else:
#     voltage = props.kv * 1e3 if props.kv is not None else None
#     diff_step = props.diff_step
#     scan_shape = None
#     probe_hook = scan_hook = None
#     adu = None
#     needs_scale = False

    if metadata.voltage is None:
        raise ValueError("'kv'/'voltage' must be specified by metadata or passed to 'raw_data'")
    if metadata.diff_step is None:
        raise ValueError("'diff_step' must be specified by metadata or passed to 'raw_data'")

    wavelength = Electron(voltage).wavelength

    if not path.exists():
        raise ValueError(f"Couldn't find gatan data at path {path}")

    patterns = numpy.fft.ifftshift(load_4d(path, scan_shape, memmap=True), axes=(-1, -2))

    if needs_scale:
        if adu is None:
            warnings.warn("ADU not supplied for experimental dataset. This is not recommended.")
        else:
            logger.info(f"Scaling patterns by ADU ({adu:.1f})")
            patterns /= adu

    print(diff_step)

    a = wavelength / (diff_step * 1e-3)  # recip. pixel size -> 1 / real space extent

    sampling = Sampling(cast_length(patterns.shape[-2:], 2), extent=(a, a))



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