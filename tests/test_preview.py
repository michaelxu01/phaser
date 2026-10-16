import json
import shutil

import numpy
import pytest
from click.testing import CliRunner

from .utils import INPUT_FILES_PATH


def disc_patterns(scan_shape=(4, 6), det_shape=(32, 32), radius=5., center=(0., 0.)):
    """Uniform disc patterns, fftshifted (as returned by the raw data hooks). Intensity scales with scan index."""
    ky = numpy.arange(det_shape[0]) - det_shape[0] // 2
    kx = numpy.arange(det_shape[1]) - det_shape[1] // 2
    kyy, kxx = numpy.meshgrid(ky, kx, indexing='ij')
    disc = ((kyy - center[0])**2 + (kxx - center[1])**2 <= radius**2).astype(numpy.float32)
    scale = 1. + numpy.arange(numpy.prod(scan_shape), dtype=numpy.float32).reshape(scan_shape)
    patterns = scale[..., None, None] * disc
    return numpy.fft.ifftshift(patterns, axes=(-2, -1)), disc


@pytest.mark.parametrize(('center', 'edge'), [
    # 32 px detector at 0.5 mrad/px spans [-8.25, 7.75] mrad
    ((0., 0.), 7.75),
    ((2., -3.), 6.75),
])
def test_compute_previews(center, edge):
    from phaser.cli.preview import Aperture, compute_previews

    (patterns, disc) = disc_patterns(center=center)
    mask = numpy.ones(patterns.shape[-2:], dtype=numpy.float32)
    previews = compute_previews(
        patterns, mask, 0.5, [Aperture('BF', 0., 1.), Aperture('ADF', 4., numpy.inf)], grid=(0., 1.)
    )

    # mean pattern is centered, and the aperture follows the disc
    numpy.testing.assert_allclose(previews.mean_pattern, disc * 12.5)
    assert previews.center == pytest.approx((center[0] * 0.5, center[1] * 0.5))

    # grid is (y, x) indices, at the corners of the scan
    numpy.testing.assert_array_equal(previews.coords, [(0, 0), (0, 5), (3, 0), (3, 5)])
    numpy.testing.assert_allclose(previews.patterns[1], disc * 6.)

    [(bf_ap, bf), (adf_ap, adf)] = previews.virtual
    assert (bf_ap.name, adf_ap.name) == ('BF', 'ADF')
    # infinite outer radius is limited to the detector edge
    assert adf_ap.outer == pytest.approx(edge)
    assert bf.shape == adf.shape == (4, 6)
    # 1 mrad radius at 0.5 mrad/px: 13 pixels, all inside the disc
    numpy.testing.assert_allclose(bf, 13. * (1. + numpy.arange(24).reshape(4, 6)))
    # ADF is outside the disc
    numpy.testing.assert_allclose(adf, 0.)


@pytest.mark.parametrize(('conv_angle', 'bf_radius'), [
    (2., 2.),
    # estimated from the area of the disc (81 px)
    (None, 0.5 * numpy.sqrt(81 / numpy.pi)),
])
def test_compute_previews_default_apertures(conv_angle, bf_radius):
    from phaser.cli.preview import compute_previews

    (patterns, _) = disc_patterns(center=(2., -3.))
    mask = numpy.ones(patterns.shape[-2:], dtype=numpy.float32)
    previews = compute_previews(patterns, mask, 0.5, conv_angle=conv_angle)

    [(point_ap, point), (bf_ap, _), (adf_ap, _)] = previews.virtual
    assert [ap.name for ap in (point_ap, bf_ap, adf_ap)] == ['Center pixel', 'BF', 'ADF']
    assert bf_ap.outer == pytest.approx(bf_radius)
    assert adf_ap.inner == pytest.approx(1.2 * bf_radius)
    assert adf_ap.outer == pytest.approx(6.75)
    # single pixel at the disc center
    numpy.testing.assert_allclose(point, 1. + numpy.arange(24).reshape(4, 6))


def test_compute_previews_mask():
    from phaser.cli.preview import Aperture, compute_previews

    (patterns, _) = disc_patterns()
    mask = numpy.zeros(patterns.shape[-2:], dtype=numpy.float32)
    previews = compute_previews(patterns, mask, 1., [Aperture('BF', 0., 2.)])
    numpy.testing.assert_allclose(previews.virtual[0][1], 0.)
    numpy.testing.assert_allclose(previews.mean_pattern, 0.)


def test_compute_previews_invalid_grid():
    from phaser.cli.preview import compute_previews

    (patterns, _) = disc_patterns()
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_previews(patterns, numpy.ones(patterns.shape[-2:]), 1., [], grid=(1.5,))


def make_dataset(path, **overrides):
    """EMPAD1 dataset with 2x2 scan, from 'empad_v1.raw'"""
    path.mkdir()
    shutil.copy(INPUT_FILES_PATH / 'empad_v1.raw', path / 'scan_x2_y2.raw')
    meta = json.loads((INPUT_FILES_PATH / 'metadata_v2.json').read_text())
    meta.update(raw_filename='scan_x2_y2.raw', scan_shape=[2, 2], empad_version=1, **overrides)
    meta_path = path / 'acquisition.json'
    meta_path.write_text(json.dumps(meta))
    return meta_path


@pytest.mark.parametrize('diff_step', [0.78, None])
def test_preview_cli(tmp_path, diff_step):
    from phaser.cli.preview import preview

    make_dataset(tmp_path / 'acq_1', diff_step=diff_step)
    make_dataset(tmp_path / 'acq_2', diff_step=diff_step)
    # calibrated metadata is skipped
    shutil.copy(tmp_path / 'acq_1' / 'acquisition.json', tmp_path / 'acq_1' / 'acquisition_calib.json')
    # as are unrelated JSON files
    (tmp_path / 'other').mkdir()
    (tmp_path / 'other' / 'unknown_type.json').write_text(json.dumps({'file_type': 'other'}))
    (tmp_path / 'other' / 'no_type.json').write_text(json.dumps({'name': 'acq'}))
    (tmp_path / 'other' / 'list.json').write_text('[1, 2]')
    (tmp_path / 'other' / 'invalid.json').write_text('{')

    runner = CliRunner()
    result = runner.invoke(preview, [str(tmp_path)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "2 file(s) to process." in result.output
    assert "Failed" not in result.output
    assert ("using pixel units" in result.output) == (diff_step is None)

    for acq in ('acq_1', 'acq_2'):
        out_dir = tmp_path / acq / 'previews'
        assert sorted(f.name for f in out_dir.iterdir()) == ['mean_pattern.png', 'pattern_grid.png', 'virtual_images.png']

    # existing previews are skipped
    result = runner.invoke(preview, [str(tmp_path)], catch_exceptions=False)
    assert "0 file(s) to process." in result.output

    result = runner.invoke(preview, [str(tmp_path), '--no-skip-existing', '--include', 'acq*.json'], catch_exceptions=False)
    assert "2 file(s) to process." in result.output


def test_preview_cli_unknown_file(tmp_path):
    from phaser.cli.preview import preview

    meta_path = tmp_path / 'unknown.json'
    meta_path.write_text(json.dumps({'file_type': 'other'}))
    result = CliRunner().invoke(preview, [str(meta_path)])
    assert "1 file(s) to process." in result.output
    assert "unknown 'file_type'" in result.output
    assert not (tmp_path / 'previews').exists()


@pytest.mark.parametrize(('module', 'cls', 'loader'), [
    ('phaser.io.empad', 'EmpadMetadata', 'empad'),
    ('phaser.io.gatan', 'GatanMetadata', 'gatan'),
    ('phaser.io.nion', 'NionMetadata', 'nion'),
])
def test_loaders_match_metadata(module, cls, loader):
    import importlib
    import typing as t
    from phaser.cli.preview import LOADERS
    from phaser.hooks import RawDataHook

    if module == 'phaser.io.gatan':
        pytest.importorskip('rsciio')
    file_types = t.get_args(t.get_type_hints(getattr(importlib.import_module(module), cls))['file_type'])
    for file_type in file_types:
        assert LOADERS[file_type] == loader
    assert loader in RawDataHook.known


def test_preview_cli_bad_grid(tmp_path):
    from phaser.cli.preview import preview

    result = CliRunner().invoke(preview, [str(tmp_path), '--grid', 'a,b'])
    assert result.exit_code != 0
    assert "comma-separated" in result.output


def test_preview_cli_grid_out_of_range(tmp_path):
    from phaser.cli.preview import preview

    result = CliRunner().invoke(preview, [str(tmp_path), '--grid', '0.5,1.5'])
    assert result.exit_code != 0
    assert "between 0 and 1" in result.output
