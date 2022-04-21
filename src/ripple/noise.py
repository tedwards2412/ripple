"""
Detector noise power spectral densities.
"""

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources  # type: ignore

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import noise_resources
from .typing import Array


f_range_LIGOI = (40.0, 1e4)
r"""
LIGO-I frequency range [Hz].

References:
    `<https://arxiv.org/abs/gr-qc/0010009>`_
"""


def Sn_LIGOI(f: Array) -> Array:
    r"""
    LIGO-I noise PSD.

    References:
        `<https://arxiv.org/abs/gr-qc/0010009>`_

    Args:
        f: frequency

    Returns:
        The noise PSD.
    """
    fs = 40  # Hz
    f_theta = 150  # Hz
    x = f / f_theta
    normalization = 1e-46
    return jnp.where(
        f > fs,
        normalization
        * 9
        * ((4.49 * x) ** (-56) + 0.16 * x ** (-4.52) + 0.52 + 0.32 * x**2),
        jnp.inf,
    )


f_range_LISA = (2e-5, 1.0)  # Hz, from LISA Science Requirements fig. 1
r"""
LISA frequency band [Hz].

References:
    LISA Science Requirements, fig. 1 (`<https://www.cosmos.esa.int/documents/678316/1700384/SciRD.pdf>`_)
"""


def Sn_LISA(f):
    r"""
    LISA noise PSD, averaged over sky position and polarization angle.

    Reference:
        Travis Robson et al 2019 Class. Quantum Grav. 36 105011
        `<https://arxiv.org/abs/1803.01944>`_`
    """
    return jnp.where(
        jnp.logical_and(f >= f_range_LISA[0], f <= f_range_LISA[1]),
        1
        / f**14
        * 1.80654e-17
        * (0.000606151 + f**2)
        * (3.6864e-76 + 3.6e-37 * f**8 + 2.25e-30 * f**10 + 8.66941e-21 * f**14),
        jnp.inf,
    )


def _load_noise(name: str, asd: bool = False) -> Tuple[Callable[[Array], Array], Tuple[float, float]]:
    r"""
    Loads noise PSD from text data file into an interpolator. The file's
    columns must contain the frequencies and corresponding noise power spectral
    density or amplitude spectral density values.

    Args:
        name: name of data file in ``noise_resources`` without the `.dat`
            extension
        asd: ``True`` if the file contains ASD (ie, sqrt(PSD)) rather than PSD data

    Returns
        Interpolator for noise PSD returning ``inf`` above and below the
        frequency range in the data file
    """
    path_context = pkg_resources.path(noise_resources, f"{name}.dat")
    with path_context as path:
        fs, Sns = np.loadtxt(path, unpack=True)

    if asd:
        Sns = Sns**2

    Sns[Sns == 0.0] = np.inf

    fs = jnp.array(fs)
    Sns = jnp.array(Sns)
    f_range = (fs[0], fs[-1])

    return jax.jit(lambda f: jnp.interp(f, fs, Sns, left=jnp.inf, right=jnp.inf)), f_range


Sn_aLIGO, f_range_aLIGO = _load_noise("aLIGO", asd=True)
r"""The advanced LIGO noise PSD and frequency range.

References:
    `<https://dcc.ligo.org/LIGO-T1800044/public>`_
"""

Sn_ce, f_range_ce = _load_noise("ce", asd=True)
r"""The Cosmic Explorer noise PSD and frequency range.

References:
    `<https://dcc.ligo.org/LIGO-P1600143/public>`_
"""

Sn_et, f_range_et = _load_noise("et", asd=True)
r"""The Einstein Telescope noise PSD and frequency range.

References:
    `<http://www.et-gw.eu/index.php/etsensitivities>`_
"""


Sn_aLIGOZeroDetHighPower, f_range_aLIGOZeroDetHighPower = _load_noise("aLIGOZeroDetHighPower")
r"""The aLIGOZeroDetHighPower noise PSD from pycbc and frequency range.

References:
    ???

Args:
    f: frequency

Returns:
    The noise power spectral density
"""

Sn_O3a, f_range_O3a = _load_noise("O3a_Livingston_ASD", asd=True)
r"""The LIGO O3a Livingston noise PSD and frequency range.

References:
    ???

Args:
    f: frequency

Returns:
    The noise power spectral density
"""

Sn_O2, f_range_O2 = _load_noise("O2_ASD", asd=True)
r"""The LIGO O2 noise PSD and frequency range.

References:
    `<https://github.com/jroulet/template_bank>`_

Args:
    f: frequency

Returns:
    The noise power spectral density
"""
