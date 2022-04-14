import jax.numpy as jnp
from typing import Tuple


def get_match(
    h1: jnp.ndarray,
    h2: jnp.ndarray,
    Sns: jnp.ndarray,
    fs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the match between two frequency-domain complex strains. The maximizations
    over the difference in time and phase at coalescence are performed by taking
    the absolute value of the inverse Fourier transform.
    Args:
        h1: the first set of strains
        h2: the second set of strains
        Sns: the noise power spectral densities
        fs: frequencies at which the strains and noise PSDs were evaluated
        pad_low: array of zeros to pad the left side of the integrand before it
            is passed to ``jax.numpy.fft.ifft``
        pad_right: array of zeros to pad the right side of the integrand before
            it is passed to ``jax.numpy.fft.ifft``
    Returns:
        The match.
    """
    # Get the padding so that
    pad_low, pad_high = get_eff_pads(fs)

    # Factors of 4 and df drop out due to linearity
    norm1 = jnp.sqrt(jnp.sum(jnp.abs(h1) ** 2 / Sns))
    norm2 = jnp.sqrt(jnp.sum(jnp.abs(h2) ** 2 / Sns))

    # Use IFFT trick to maximize over t_c. Ref: Maggiore's book, eq. 7.171.
    integrand_padded = jnp.concatenate((pad_low, h1.conj() * h2 / Sns, pad_high))
    return jnp.abs(len(integrand_padded) * jnp.fft.ifft(integrand_padded)).max() / (
        norm1 * norm2
    )


def get_eff_pads(fs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    df = fs[1] - fs[0]
    N = 2 * jnp.array(fs[-1] / df - 1).astype(int)
    pad_low = jnp.zeros(jnp.array(fs[0] / df).astype(int))
    pad_high = jnp.zeros(N - jnp.array(fs[-1] / df).astype(int))
    return pad_low, pad_high