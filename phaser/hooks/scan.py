
import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module
from phaser.utils.scan import make_raster_scan
from . import ScanHookArgs, RasterScanProps


def raster_scan(args: ScanHookArgs, props: RasterScanProps) -> (NDArray[numpy.floating], tuple):
   

   return (make_raster_scan(
        props.shape, props.step_size, props.rotation,
        dtype=args['dtype'], xp=cast_array_module(args['xp']), crop=props.crop
    ), props.crop)