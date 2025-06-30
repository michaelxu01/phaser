
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from .utils import with_backends, get_backend_module, get_backend_scipy, mock_importerror

from phaser.utils.num import (
    get_array_module, get_scipy_module,
    to_real_dtype, to_complex_dtype,
    fft2, ifft2, abs2,
    to_numpy, as_array,
    ufunc_outer
)


@with_backends('cpu', 'jax', 'cuda')
def test_get_array_module(backend: str):
    expected = get_backend_module(backend)

    mocked_imports = {
        # on cpu, pretend cupy and jax don't exist
        'cpu': {'cupy', 'jax'},
        'jax': {},
        'cuda': {},
    }[backend]

    assert get_array_module() is numpy

    with mock_importerror(mocked_imports):
        assert get_array_module(
            numpy.array([1., 2., 3.]),
            expected.array([1, 2, 3]),
            None,
            numpy.array([1., 2., 3.]),
        ) is expected


@with_backends('cpu', 'jax', 'cuda')
def test_get_scipy_module(backend: str):
    import scipy

    xp = get_backend_module(backend)
    expected = get_backend_scipy(backend)

    mocked_imports = {
        # on cpu, pretend cupyx doesn't exist
        'cpu': {'cupyx'},
        'jax': {},
        'cuda': {},
    }[backend]

    assert get_scipy_module() is scipy

    with mock_importerror(mocked_imports):
        assert get_scipy_module(
            numpy.array([1., 2., 3.]),
            xp.array([1, 2, 3]),
            None,
            numpy.array([1., 2., 3.]),
        ) is expected


@pytest.mark.parametrize(('input', 'expected'), [
    (numpy.complex64, numpy.float32),
    (numpy.complex128, numpy.float64),
    (numpy.complexfloating, numpy.floating),
    ('complex128', numpy.float64),
    (float, numpy.float64),
    (complex, numpy.float64),
])
def test_to_real_dtype(input, expected):
    assert to_real_dtype(input) is expected
    # test idempotence
    assert to_real_dtype(expected) is expected


@pytest.mark.parametrize(('input', 'expected'), [
    (numpy.float32, numpy.complex64),
    (numpy.float64, numpy.complex128),
    (numpy.floating, numpy.complexfloating),
    ('float32', numpy.complex64),
    (float, numpy.complex128),
    (complex, numpy.complex128),
])
def test_to_complex_dtype(input, expected):
    assert to_complex_dtype(input) is expected
    # test idempotence
    assert to_complex_dtype(expected) is expected


def test_to_real_dtype_invalid():
    with pytest.raises(TypeError, match="Non-floating point datatype"):
        to_real_dtype(numpy.int_)


def test_to_complex_dtype_invalid():
    with pytest.raises(TypeError, match="Non-floating point datatype"):
        to_complex_dtype(numpy.int_)


@with_backends('cpu', 'jax', 'cuda')
def test_fft2(backend: str):
    xp = get_backend_module(backend)

    # point input, f = 5 delta(x) delta(y)
    a = xp.pad(xp.array([[5.]], dtype=numpy.float32), ((2, 2), (2, 2)))

    # even input, so output is real
    # delta function input, so output is constant
    # normalized so intensity in = intensity out
    assert_array_almost_equal(
        to_numpy(fft2(a)),
        numpy.full((5, 5), 1., dtype=numpy.complex64)
    )

    # plane input, f = 1 + 1i
    a = xp.full((5, 5), 1+1.j, dtype=numpy.complex64)

    # constant input, so output is delta function at k=0
    # normalized so intensity in = intensity out
    # zero frequency is cornered
    assert_array_almost_equal(
        to_numpy(fft2(a)),
        numpy.pad([[5.+5.j]], ((0, 4), (0, 4))).astype(numpy.complex64)
    )


@with_backends('cpu', 'jax', 'cuda')
def test_ifft2(backend: str):
    xp = get_backend_module(backend)

    # point input, F = delta(k_x) delta(k_y)
    a = xp.pad(xp.array([[5.]], dtype=numpy.float32), ((0, 4), (0, 4)))

    # even input, so output is real
    # delta function input, so output is constant
    # normalized so intensity in = intensity out
    assert_array_almost_equal(
        to_numpy(ifft2(a)),
        numpy.full((5, 5), 1., dtype=numpy.complex64),
        decimal=5
    )

    # plane input, F = 1 + 1i
    a = xp.full((5, 5), 1+1.j, dtype=numpy.complex64)

    # constant input, so output is delta function at k=0
    # normalized so intensity in = intensity out
    # zero position is centered
    assert_array_almost_equal(
        to_numpy(ifft2(a)),
        numpy.pad([[5.+5.j]], ((2, 2), (2, 2))).astype(numpy.complex64),
        decimal=5
    )


@with_backends('cpu', 'jax', 'cuda')
def test_abs2(backend: str):
    xp = get_backend_module(backend)

    if backend == 'cpu':
        assert_array_almost_equal(abs2([1.+1.j, 1.-1.j]), numpy.array([2., 2.]))

    assert_array_almost_equal(
        to_numpy(abs2(xp.array([1.+1.j, 1.-1.j]))),
        numpy.array([2., 2.]),
    )

    assert_array_almost_equal(
        to_numpy(abs2(xp.array([1., -2., 5.], dtype=numpy.float32))),
        numpy.array([1, 4., 25.], dtype=numpy.float32),
        decimal=5  # this is pretty poor performance
    )


@with_backends('cpu', 'jax', 'cuda')
def test_to_numpy(backend: str):
    xp = get_backend_module(backend)

    arr = xp.array([1., 2., 3., 4.])

    assert_array_almost_equal(
        to_numpy(arr),
        numpy.array([1., 2., 3., 4.])
    )


@with_backends('cpu', 'jax', 'cuda')
def test_to_array(backend: str):
    xp = get_backend_module(backend)

    arr = xp.array([1., 2., 3., 4.])
    assert as_array(arr) is arr

    arr = as_array([1., 2., 3., 4.])
    assert isinstance(arr, numpy.ndarray)
    assert_array_almost_equal(
        arr,
        numpy.array([1., 2., 3., 4.])
    )


@with_backends('cpu', 'jax', 'cuda')
def test_ufunc_outer(backend: str):
    xp = get_backend_module(backend)

    xs = numpy.arange(12).reshape(4, 3)
    ys = numpy.arange(30).reshape(5, 6)

    expected = numpy.multiply.outer(xs, ys)
    actual = to_numpy(ufunc_outer(xp.multiply, xp.array(xs), xp.array(ys)))
    assert_array_equal(expected, actual)
