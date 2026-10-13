import json

import pytest
from pane.errors import ConvertError

from phaser.io.empad import EmpadMetadata, UnscannedDataset

from .utils import INPUT_FILES_PATH


def test_read_empad_metadata():
    metadata = EmpadMetadata.from_json(INPUT_FILES_PATH / 'metadata_v2.json')

    print(repr(metadata))
    assert metadata.file_type == 'empad_metadata'
    assert metadata.name == 'acquisition_5_7.2Mx_17.9mrad_460mmcl_-20nmdefocus'
    assert metadata.raw_filename == 'scan_x128_y128.raw'
    assert metadata.author == "Abinash"
    assert metadata.path == INPUT_FILES_PATH
    assert metadata.time == '2021-05-02T06:58:26.775565'
    assert metadata.voltage == 200000.0
    assert metadata.conv_angle == 17.9
    assert metadata.camera_length == 0.4575
    assert metadata.scan_fov == (1.2435077703182981e-08,) * 2
    assert metadata.scan_step == (0.9714904455611704e-10,) * 2
    assert metadata.beam_current == 30.0e-12
    assert metadata.notes is None
    assert metadata.adu == 375.0
    assert metadata.defocus is None


def test_read_empad2_toml_metadata(tmp_path):
    import shutil

    toml_path = tmp_path / 'metadata.toml'
    shutil.copy(INPUT_FILES_PATH / 'empad2_metadata.toml', toml_path)
    metadata = EmpadMetadata.from_toml(toml_path)

    print(repr(metadata))
    assert metadata.file_type == 'empad_metadata'
    assert metadata.name == 'acquisition_53'
    assert metadata.empad_version == 2
    # raw file not present, conventional name is used
    assert metadata.raw_filename == 'scan_x200_y200.raw'
    assert metadata.orig_path == tmp_path
    assert metadata.author is None
    assert metadata.time == '2026-10-10T19:45:30.408447+00:00'
    assert metadata.time_unix == pytest.approx(1791661530.408447)
    assert metadata.bg_unix == pytest.approx(1791661179.214721)
    assert metadata.has_bg is True
    assert metadata.voltage == 300000.0
    assert metadata.conv_angle == pytest.approx(25.011686787433686)
    assert metadata.camera_length == 0.5766
    assert metadata.scan_rotation == 0.0
    assert metadata.scan_shape == (200, 200)
    assert metadata.scan_fov == (9.585889570552145e-09,) * 2
    assert metadata.scan_step == (4.769099351542572e-11,) * 2
    assert metadata.exposure_time == 0.0005
    assert metadata.notes == "15nm"
    assert metadata.defocus is None

    # single raw file on disk is used
    (tmp_path / 'other.raw').touch()
    metadata = EmpadMetadata.from_toml(toml_path)
    assert metadata.raw_filename == 'other.raw'

    # only raster scans are 4D STEM datasets
    toml_path.write_text(toml_path.read_text().replace('"stem_raster"', '"other"'))
    with pytest.raises(UnscannedDataset):
        EmpadMetadata.from_toml(toml_path)


def test_read_pymultislicer_metadata():
    metadata = EmpadMetadata.from_json(INPUT_FILES_PATH / 'sim_metadata.json')

    print(repr(metadata))
    assert metadata.file_type == 'pyMultislicer_metadata'
    assert metadata.name == 'SiC-ErV_kh_-20focus_320mm'
    assert metadata.voltage == 200000.0
    assert metadata.conv_angle == 18.9
    assert metadata.defocus == 20.0e-9
    assert metadata.diff_step == 1.1083125
    assert metadata.scan_rotation == 0.0
    assert metadata.scan_shape == (44, 46)
    assert metadata.scan_fov == pytest.approx((1.4607789440000002e-09, 1.5272094792000002e-09))
    assert metadata.scan_step == pytest.approx((3.32e-11,) * 2)


def test_metadata_bad_version():
    with open(INPUT_FILES_PATH / 'metadata_v2.json') as f:
        obj = json.load(f)

    obj['version'] = '2.5'
    with pytest.raises(ConvertError, match="Version 2.5 is not supported, version 2.0 required"):
        EmpadMetadata.from_data(obj)

    obj['version'] = '1_5'
    with pytest.raises(ConvertError, match="Invalid version string"):
        EmpadMetadata.from_data(obj)