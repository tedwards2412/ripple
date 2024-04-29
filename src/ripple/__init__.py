"""
Core utilities for calculating properties of binaries, sampling their parameters
and comparing waveforms.
"""

# Note that we turned this off to test float32 capabilities
# from jax.config import config
# config.update("jax_enable_x64", True)

from math import pi
from typing import Callable, Optional, Tuple
import warnings

from jax import random
import jax.numpy as jnp

from .constants import C, G
from .typing import Array


def Mc_eta_to_ms(m):
    r"""
    Converts chirp mass and symmetric mass ratio to binary component masses.

    Args:
        m: the binary component masses ``(Mchirp, eta)``

    Returns:
        :math:`(m1, m2)`, with the chirp mass in the same units as
        the component masses
    """
    Mchirp, eta = m
    M = Mchirp / (eta ** (3 / 5))
    m2 = (M - jnp.sqrt(M**2 - 4 * M**2 * eta)) / 2
    m1 = M - m2
    return m1, m2


def ms_to_Mc_eta(m):
    r"""
    Converts binary component masses to chirp mass and symmetric mass ratio.

    Args:
        m: the binary component masses ``(m1, m2)``

    Returns:
        :math:`(\mathcal{M}, \eta)`, with the chirp mass in the same units as
        the component masses
    """
    m1, m2 = m
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5), m1 * m2 / (m1 + m2) ** 2


# TODO in code below, reduce copy-paste
def lambdas_to_lambda_tildes_from_q(params: Array):
    """
    Convert from individual tidal parameters to domainant tidal term. (Code taken from Bilby)

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Returns
    =======
    lambda_tilde: float
        Dominant tidal term.
    """
    lambda_1, lambda_2, q = params
    eta = q / (1 + q) ** 2
    lambda_plus = lambda_1 + lambda_2
    lambda_minus = lambda_1 - lambda_2
    lambda_tilde = (
        8
        / 13
        * (
            (1 + 7 * eta - 31 * eta**2) * lambda_plus
            + (1 - 4 * eta) ** 0.5 * (1 + 9 * eta - 11 * eta**2) * lambda_minus
        )
    )

    delta_lambda_tilde = (
        1
        / 2
        * (
            (1 - 4 * eta) ** 0.5
            * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2)
            * lambda_plus
            + (1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2 + 3380 / 1319 * eta**3)
            * lambda_minus
        )
    )

    return lambda_tilde, delta_lambda_tilde


def lambdas_to_lambda_tildes(params: Array):
    """
    Convert from individual tidal parameters to domainant tidal term. (Code taken from Bilby)

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Returns
    =======
    lambda_tilde: float
        Dominant tidal term.
    """
    lambda_1, lambda_2, mass_1, mass_2 = params
    _, eta = ms_to_Mc_eta(jnp.array([mass_1, mass_2]))
    lambda_plus = lambda_1 + lambda_2
    lambda_minus = lambda_1 - lambda_2
    lambda_tilde = (
        8
        / 13
        * (
            (1 + 7 * eta - 31 * eta**2) * lambda_plus
            + (1 - 4 * eta) ** 0.5 * (1 + 9 * eta - 11 * eta**2) * lambda_minus
        )
    )

    delta_lambda_tilde = (
        1
        / 2
        * (
            (1 - 4 * eta) ** 0.5
            * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2)
            * lambda_plus
            + (1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2 + 3380 / 1319 * eta**3)
            * lambda_minus
        )
    )

    return lambda_tilde, delta_lambda_tilde


def lambda_tildes_to_lambdas(params: Array):
    """
    Convert from dominant tidal terms to individual tidal parameters. Code taken from bilby.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    lambda_tilde: float
        Dominant tidal term.
    delta_lambda_tilde: float
        Secondary tidal term.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Returns
    =======
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.

    """

    lambda_tilde, delta_lambda_tilde, mass_1, mass_2 = params

    _, eta = ms_to_Mc_eta(jnp.array([mass_1, mass_2]))
    coefficient_1 = 1 + 7 * eta - 31 * eta**2
    coefficient_2 = (1 - 4 * eta) ** 0.5 * (1 + 9 * eta - 11 * eta**2)
    coefficient_3 = (1 - 4 * eta) ** 0.5 * (
        1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2
    )
    coefficient_4 = (
        1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2 + 3380 / 1319 * eta**3
    )
    lambda_1 = (
        13 * lambda_tilde / 8 * (coefficient_3 - coefficient_4)
        - 2 * delta_lambda_tilde * (coefficient_1 - coefficient_2)
    ) / (
        (coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4)
        - (coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4)
    )
    lambda_2 = (
        13 * lambda_tilde / 8 * (coefficient_3 + coefficient_4)
        - 2 * delta_lambda_tilde * (coefficient_1 + coefficient_2)
    ) / (
        (coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4)
        - (coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4)
    )

    return lambda_1, lambda_2


def lambda_tildes_to_lambdas_from_q(params: Array):
    """
    Convert from dominant tidal terms to individual tidal parameters. Code taken from bilby.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    lambda_tilde: float
        Dominant tidal term.
    delta_lambda_tilde: float
        Secondary tidal term.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Returns
    =======
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.

    """

    lambda_tilde, delta_lambda_tilde, q = params

    eta = q / (1 + q) ** 2

    coefficient_1 = 1 + 7 * eta - 31 * eta**2
    coefficient_2 = (1 - 4 * eta) ** 0.5 * (1 + 9 * eta - 11 * eta**2)
    coefficient_3 = (1 - 4 * eta) ** 0.5 * (
        1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2
    )
    coefficient_4 = (
        1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2 + 3380 / 1319 * eta**3
    )
    lambda_1 = (
        13 * lambda_tilde / 8 * (coefficient_3 - coefficient_4)
        - 2 * delta_lambda_tilde * (coefficient_1 - coefficient_2)
    ) / (
        (coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4)
        - (coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4)
    )
    lambda_2 = (
        13 * lambda_tilde / 8 * (coefficient_3 + coefficient_4)
        - 2 * delta_lambda_tilde * (coefficient_1 + coefficient_2)
    ) / (
        (coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4)
        - (coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4)
    )

    return lambda_1, lambda_2


def get_chi_eff(params: Array) -> float:
    """Compute effective spin.

    Args:
        params (Array): Parameters: mass1, mass2, spin1, spin2

    Returns:
        float: Effective spin.
    """
    m1, m2, chi1, chi2 = params

    chi_eff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    return chi_eff


def get_f_isco(m):
    r"""
    Computes the ISCO frequency for a black hole.

    Args:
        m: the black hole's mass in kg

    Returns:
        The ISCO frequency in Hz
    """
    return 1 / (6 ** (3 / 2) * pi * m / (C**3 / G))


def get_M_eta_sampler(M_range: Tuple[float, float], eta_range: Tuple[float, float]):
    """
    Uniformly values of the chirp mass and samples over the specified ranges.
    This function may be removed in the future since it is trivial.
    """

    def sampler(key, n):
        M_eta = random.uniform(
            key,
            minval=jnp.array([M_range[0], eta_range[0]]),
            maxval=jnp.array([M_range[1], eta_range[1]]),
            shape=(n, 2),
        )
        return M_eta

    return sampler


def get_m1_m2_sampler(m1_range: Tuple[float, float], m2_range: Tuple[float, float]):
    r"""
    Creates a function to uniformly sample two parameters, with the restriction
    that the first is larger than the second.

    Note:
        While this function is particularly useful for sampling masses in a
        binary, nothing in it is specific to that context.

    Args:
        m1_range: the minimum and maximum values of the first parameter
        m2_range: the minimum and maximum values of the second parameter

    Returns:
        The sampling function
    """

    def sampler(key, n):
        ms = random.uniform(
            key,
            minval=jnp.array([m1_range[0], m2_range[0]]),
            maxval=jnp.array([m1_range[1], m2_range[1]]),
            shape=(n, 2),
        )
        return jnp.stack([ms.max(axis=1), ms.min(axis=1)]).T  # type: ignore

    return sampler


def get_eff_pads(fs: Array) -> Tuple[Array, Array]:
    r"""
    Gets arrays of zeros to pad a function evaluated on a frequency grid so the
    function values can be passed to ``jax.numpy.fft.ifft``.

    Args:
        fs: uniformly-spaced grid of frequencies. It is assumed that the first
            element in the grid must be an integer multiple of the grid spacing
            (i.e., ``fs[0] % df == 0``, where ``df`` is the grid spacing).

    Returns:
        The padding arrays of zeros. The first is of length ``fs[0] / df`` and
        the second is of length ``fs[-1] / df - 2``.
    """
    df = (fs[-1] - fs[0]) / (len(fs) - 1)

    if not jnp.allclose(jnp.diff(fs), df).all():
        warnings.warn("frequency grid may not be evenly spaced")

    if fs[0] % df != 0 or fs[-1] % df != 0:
        warnings.warn(
            "The first and/or last elements of the frequency grid are not integer "
            "multiples of the grid spacing. The frequency grid and pads from this "
            "function will thus yield inaccurate results when used with fft/ifft."
        )

    N = 2 * jnp.array(fs[-1] / df - 1).astype(int)
    pad_low = jnp.zeros(jnp.array(fs[0] / df).astype(int))
    pad_high = jnp.zeros(N - jnp.array(fs[-1] / df).astype(int))
    return pad_low, pad_high


# pad_low, pad_high, Sns, h1s, h2s
def get_phase_maximized_inner_product_arr(
    del_t: Array, fs: Array, Sns: Array, h1s: Array, h2s: Array
) -> Array:
    r"""
    Calculates the inner product between two waveforms, maximized over the difference
    in phase at coalescence. This is just the absolute value of the noise-weighted
    inner product.

    Args:
        del_t: difference in the time at coalescence for the waveforms
        h1s: the first set of strains
        h2s: the second set of strains
        Sns: the noise power spectral densities
        fs: uniformly-spaced grid of frequencies used to perform the integration

    Returns:
        The noise-weighted inner product between the waveforms, maximized over
        the phase at coalescence
    """
    # Normalize both waveforms. Factors of 4 and df drop out.
    norm1 = jnp.sqrt(jnp.sum(jnp.abs(h1s) ** 2 / Sns))
    norm2 = jnp.sqrt(jnp.sum(jnp.abs(h2s) ** 2 / Sns))
    # Compute unnormalized match, maximizing over phi_0 by taking the absolute value
    integral = jnp.abs(
        jnp.sum(h1s.conj() * h2s * jnp.exp(1j * 2 * pi * fs * del_t) / Sns)
    )
    return integral / (norm1 * norm2)


def get_phase_maximized_inner_product(
    del_t: Array,
    fs: Array,
    Sn: Callable[[Array], Array],
    theta1: Array,
    theta2: Array,
    amp1: Callable[[Array, Array], Array],
    Psi1: Callable[[Array, Array], Array],
    amp2: Optional[Callable[[Array, Array], Array]],
    Psi2: Optional[Callable[[Array, Array], Array]],
) -> Array:
    r"""
    Calculates the inner product between two waveforms, maximized over the difference
    in phase at coalescence. This is just the absolute value of the noise-weighted
    inner product.

    Args:
        theta1: parameters for the first waveform
        theta2: parameters for the second waveform
        del_t: difference in the time at coalescence for the waveforms
        amp1: amplitude function for first waveform
        Psi1: phase function for first waveform
        amp2: amplitude function for second waveform
        Psi2: phase function for second waveform
        fs: uniformly-spaced grid of frequencies used to perform the integration
        Sn: power spectral density of the detector noise

    Returns:
        The noise-weighted inner product between the waveforms, maximized over
        the phase at coalescence
    """
    h1s = amp1(fs, theta1) * jnp.exp(1j * Psi1(fs, theta1))
    if amp2 is None:
        amp2 = amp1
    if Psi2 is None:
        Psi2 = Psi1
    h2s = amp2(fs, theta2) * jnp.exp(1j * Psi2(fs, theta2))
    Sns = Sn(fs)
    return get_phase_maximized_inner_product_arr(del_t, fs, Sns, h1s, h2s)


def get_match_arr(
    pad_low: Array, pad_high: Array, Sns: Array, h1s: Array, h2s: Array
) -> Array:
    """
    Calculates the match between two frequency-domain complex strains. The maximizations
    over the difference in time and phase at coalescence are performed by taking
    the absolute value of the inverse Fourier transform.

    Args:
        h1s: the first set of strains
        h2s: the second set of strains
        Sns: the noise power spectral densities
        pad_low: array of zeros to pad the left side of the integrand before it
            is passed to ``jax.numpy.fft.ifft``
        pad_right: array of zeros to pad the right side of the integrand before
            it is passed to ``jax.numpy.fft.ifft``

    Returns:
        The match.
    """
    # Factors of 4 and df drop out due to linearity
    norm1 = jnp.sqrt(jnp.sum(jnp.abs(h1s) ** 2 / Sns))
    norm2 = jnp.sqrt(jnp.sum(jnp.abs(h2s) ** 2 / Sns))

    # Use IFFT trick to maximize over t_c. Ref: Maggiore's book, eq. 7.171.
    integrand_padded = jnp.concatenate((pad_low, h1s.conj() * h2s / Sns, pad_high))
    return jnp.abs(len(integrand_padded) * jnp.fft.ifft(integrand_padded)).max() / (
        norm1 * norm2
    )


def get_match(
    fs: Array,
    pad_low: Array,
    pad_high: Array,
    Sn: Callable[[Array], Array],
    theta1: Array,
    theta2: Array,
    amp1: Callable[[Array, Array], Array],
    Psi1: Callable[[Array, Array], Array],
    amp2: Optional[Callable[[Array, Array], Array]],
    Psi2: Optional[Callable[[Array, Array], Array]],
) -> Array:
    r"""
    Calculates the match between two waveforms with different parameters and of
    distinct types. The match is defined as the noise-weighted inner product maximized
    over the difference in time and phase at coalescence. The maximizations are
    performed using the absolute value of the inverse Fourier transform trick.

    Args:
        theta1: parameters for the first waveform
        theta2: parameters for the second waveform
        amp1: amplitude function for first waveform
        Psi1: phase function for first waveform
        amp2: amplitude function for second waveform
        Psi2: phase function for second waveform
        fs: uniformly-spaced grid of frequencies used to perform the integration
        Sn: power spectral density of the detector noise
        pad_low: array of zeros to pad the left side of the integrand before it
            is passed to ``jax.numpy.fft.ifft``
        pad_right: array of zeros to pad the right side of the integrand before
            it is passed to ``jax.numpy.fft.ifft``

    Returns:
        The match :math:`m[\theta_1, \theta_2]`
    """
    h1s = amp1(fs, theta1) * jnp.exp(1j * Psi1(fs, theta1))
    if amp2 is None:
        amp2 = amp1
    if Psi2 is None:
        Psi2 = Psi1
    h2s = amp2(fs, theta2) * jnp.exp(1j * Psi2(fs, theta2))
    Sns = Sn(fs)
    return get_match_arr(pad_low, pad_high, Sns, h1s, h2s)
