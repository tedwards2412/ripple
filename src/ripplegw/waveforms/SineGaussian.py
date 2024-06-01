import jax.numpy as jnp
from jax.lax import complex

from ..constants import PI
from ..typing import Array


def semi_major_minor_from_e(e: Array) -> tuple[Array, Array]:
    """
    Calculate the semi-major and semi-minor axes of an ellipse given the
    eccentricity of the ellipse.

    Args:
        e: Eccentricity of the ellipse
    Returns:
        Semi-major (a) and semi-minor (b) axes of the ellipse
    """
    a = 1.0 / jnp.sqrt(2.0 - (e * e))
    b = a * jnp.sqrt(1.0 - (e * e))
    return a, b


def gen_SineGaussian_hphc(
    quality: Array,
    frequency: Array,
    hrss: Array,
    phase: Array,
    eccentricity: Array,
    duration: float,
    sample_rate: float = 4096.0,
) -> tuple[Array, Array]:
    """
    Generate lalinference implementation of a sine-Gaussian waveform in Jax.
    See
    git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/lib/LALInferenceBurstRoutines.c#L381
    for details on parameter definitions.

    Args:
        quality:
            Quality factor of the sine-Gaussian waveform
        frequency:
            Central frequency of the sine-Gaussian waveform
        hrss:
            Hrss of the sine-Gaussian waveform
        phase:
            Phase of the sine-Gaussian waveform
        eccentricity:
            Eccentricity of the sine-Gaussian waveform.
            Controls the relative amplitudes of the
            hplus and hcross polarizations.
    Returns:
        Jax Arrays of plus and cross polarizations
    """
    # determine times based on requested duration and sample rate
    # and shift so that the waveform is centered at t=0
    num = int(duration * sample_rate)
    times = jnp.arange(num) / sample_rate
    times -= duration / 2.0

    # add dimension for calculating waveforms in batch
    frequency = frequency.reshape(-1, 1)
    quality = quality.reshape(-1, 1)
    hrss = hrss.reshape(-1, 1)
    phase = phase.reshape(-1, 1)
    eccentricity = eccentricity.reshape(-1, 1)

    pi = jnp.array([PI])

    # calculate relative hplus / hcross amplitudes based on eccentricity
    # as well as normalization factors
    a, b = semi_major_minor_from_e(eccentricity)
    norm_prefactor = quality / (4.0 * frequency * jnp.sqrt(pi))
    cosine_norm = norm_prefactor * (1.0 + jnp.exp(-quality * quality))
    sine_norm = norm_prefactor * (1.0 - jnp.exp(-quality * quality))

    cos_phase, sin_phase = jnp.cos(phase), jnp.sin(phase)

    h0_plus = (
        hrss * a / jnp.sqrt(cosine_norm * (cos_phase**2) + sine_norm * (sin_phase**2))
    )
    h0_cross = (
        hrss * b / jnp.sqrt(cosine_norm * (sin_phase**2) + sine_norm * (cos_phase**2))
    )

    # cast the phase to a complex number
    phi = 2 * pi * frequency * times
    complex_phase = complex(jnp.zeros_like(phi), (phi - phase))

    # calculate the waveform and apply a tukey
    # window to taper the waveform
    fac = jnp.exp(phi**2 / (-2.0 * quality**2) + complex_phase)

    cross = fac.imag * h0_cross
    plus = fac.real * h0_plus

    return plus, cross
