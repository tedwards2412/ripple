"""
This file implements the NRTidalv2 corrections that can be applied to any BBH baseline, see http://arxiv.org/abs/1905.06011 for equations used.
"""

import jax
import jax.numpy as jnp
from ..constants import gt, m_per_Mpc, PI, TWO_PI, MRSUN
from ..typing import Array
from ripple import Mc_eta_to_ms, lambda_tildes_to_lambdas
from .IMRPhenom_tidal_utils import get_quadparam_octparam, get_kappa
from ripple.waveforms.IMRPhenomD import Phase, Amp, get_IIb_raw_phase
from .IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
)
from .IMRPhenomD_QNMdata import fM_CUT

#################
### AMPLITUDE ###
#################


# The code below to compute the Planck taper is obtained from gwfast (https://github.com/CosmoStatGW/gwfast/blob/ccde00e644682639aa8c9cbae323e42718fd61ca/gwfast/waveforms.py#L1332)
@jax.custom_jvp
def get_planck_taper(x: Array, y: float) -> Array:
    """
    Compute the Planck taper function.

    Args:
        x (Array): Array of frequencies
        y (float): Point at which the Planck taper starts. The taper ends at 1.2 times y.

    Returns:
        Array: Planck taper function.
    """
    a = 1.2
    yp = a * y
    return jnp.where(
        x < y,
        1.0,
        jnp.where(
            x > yp,
            0.0,
            1.0 - 1.0 / (jnp.exp((yp - y) / (x - y) + (yp - y) / (x - yp)) + 1.0),
        ),
    )


def get_planck_taper_der(x: Array, y: float):
    """
    Derivative of the Planck taper function.

    Args:
        x (Array): Array of frequencies
        y (float): Starting point of the Planck taper.

    Returns:
        Array: Array of derivative of Planck taper.
    """
    a = 1.2
    yp = a * y
    tangent_out = jnp.where(
        x < y,
        0.0,
        jnp.where(
            x > yp,
            0.0,
            jnp.exp((yp - y) / (x - y) + (yp - y) / (x - yp))
            * (
                (-1.0 + a) / (x - y)
                + (-1.0 + a) / (x - yp)
                + (-y + yp) / ((x - y) ** 2)
                + 1.2 * (-y + yp) / ((x - yp) ** 2)
            )
            / ((jnp.exp((yp - y) / (x - y) + (yp - y) / (x - yp)) + 1.0) ** 2),
        ),
    )
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out


get_planck_taper.defjvps(
    None, lambda y_dot, primal_out, x, y: get_planck_taper_der(x, y) * y_dot
)


def get_amp0_lal(M: float, distance: float):
    """
    Get the amp0 prefactor as defined in LAL in LALSimIMRPhenomD, line 331.

    Args:
        M (float): Total mass in solar masses
        distance (float): Distance to the source in Mpc.

    Returns:
        float: amp0 from LAL.
    """
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M * MRSUN * M * gt / distance
    return amp0


def get_tidal_amplitude(x: Array, theta: Array, kappa: float, distance: float = 1):
    """
    Get the tidal amplitude corrections as given in equation (24) of the NRTidal paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)
        kappa (float): Tidal parameter kappa
        distance (float, optional): Distance to the source in Mpc.

    Returns:
        Array: Tidal amplitude corrections A_T from NRTidalv2 paper.
    """

    # Mass variables
    m1, m2, _, _, _, _ = theta
    M = m1 + m2

    # Convert distance to meters
    distance *= m_per_Mpc

    # Pade approximant
    n1 = 4.157407407407407
    n289 = 2519.111111111111
    d = 13477.8073677
    num = 1.0 + n1 * x + n289 * x**2.89
    den = 1.0 + d * x**4.0
    poly = num / den

    # Prefactors are taken from lal source code
    prefac = -9.0 * kappa
    ampT = prefac * x ** (13.0 / 4.0) * poly
    amp0 = get_amp0_lal(M, distance)
    ampT *= amp0 * 2 * jnp.sqrt(PI / 5)

    return ampT


#############
### PHASE ###
#############


def get_tidal_phase(x: Array, theta: Array, kappa: float) -> Array:
    """
    Computes the tidal phase psi_T from equation (17) of the NRTidalv2 paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters in the order (mass1, mass2, chi1, chi2, lambda1, lambda2)
        kappa (float): Tidal parameter kappa, precomputed in the main function.

    Returns:
        Array: Tidal phase correction.
    """

    # Compute auxiliary quantities
    m1, m2, _, _, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    # eta = m1_s * m2_s / (M_s**2.0)

    X1 = m1_s / M_s
    X2 = m2_s / M_s

    # Compute powers
    x_2 = x ** (2.0)
    x_3 = x ** (3.0)
    x_3over2 = x ** (3.0 / 2.0)
    x_5over2 = x ** (5.0 / 2.0)

    # Initialize the coefficients
    c_Newt = 2.4375
    n_1 = -12.615214237993088
    n_3over2 = 19.0537346970349
    n_2 = -21.166863146081035
    n_5over2 = 90.55082156324926
    n_3 = -60.25357801943598
    d_1 = -15.111207827736678
    d_3over2 = 22.195327350624694
    d_2 = 8.064109635305156

    # Pade approximant
    num = (
        1.0
        + (n_1 * x)
        + (n_3over2 * x_3over2)
        + (n_2 * x_2)
        + (n_5over2 * x_5over2)
        + (n_3 * x_3)
    )
    den = 1.0 + (d_1 * x) + (d_3over2 * x_3over2) + (d_2 * x_2)
    ratio = num / den

    # Assemble everything
    psi_T = -kappa * c_Newt / (X1 * X2) * x_5over2
    psi_T *= ratio

    return psi_T


def get_spin_phase_correction(x: Array, theta: Array) -> Array:
    """
    Get the higher order spin corrections, as detailed in Section III C of the NRTidalv2 paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)

    Returns:
        Array: Higher order spin corrections to the phase
    """

    # Compute auxiliary quantities
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # Compute the auxiliary variables
    X1 = m1_s / M_s
    X1sq = X1 * X1
    chi1_sq = chi1 * chi1

    X2 = m2_s / M_s
    X2sq = X2 * X2
    chi2_sq = chi2 * chi2

    # Compute quadrupole parameters
    quadparam1, octparam1 = get_quadparam_octparam(lambda1)
    quadparam2, octparam2 = get_quadparam_octparam(lambda2)

    # Remove 1 for the BBH baseline, from here on, quadparam is "quadparam hat" as referred to in the NRTidalv2 paper etc
    quadparam1 -= 1
    quadparam2 -= 1
    octparam1 -= 1
    octparam2 -= 1

    # Get phase contributions
    SS_2 = -50.0 * quadparam1 * chi1_sq * X1sq - 50.0 * quadparam2 * chi2_sq * X2sq

    SS_3 = (
        5.0
        / 84.0
        * (9407.0 + 8218.0 * X1 - 2016.0 * X1sq)
        * quadparam1
        * X1sq
        * chi1_sq
        + 5.0
        / 84.0
        * (9407.0 + 8218.0 * X2 - 2016.0 * X2sq)
        * quadparam2
        * X2sq
        * chi2_sq
    )

    SS_3p5 = (
        -400.0 * PI * quadparam1 * chi1_sq * X1sq
        - 400.0 * PI * quadparam2 * chi2_sq * X2sq
    )
    SSS_3p5 = (
        10.0
        * ((X1sq + 308.0 / 3.0 * X1) * chi1 + (X2sq - 89.0 / 3.0 * X2) * chi2)
        * quadparam1
        * X1sq
        * chi1_sq
        + 10.0
        * ((X2sq + 308.0 / 3.0 * X2) * chi2 + (X1sq - 89.0 / 3.0 * X1) * chi1)
        * quadparam2
        * X2sq
        * chi2_sq
        - 440.0 * octparam1 * X1 * X1sq * chi1_sq * chi1
        - 440.0 * octparam2 * X2 * X2sq * chi2_sq * chi2
    )

    prefac = 3.0 / (128.0 * eta)
    psi_SS = prefac * (
        SS_2 * x ** (-1.0 / 2.0) + SS_3 * x ** (1.0 / 2.0) + (SS_3p5 + SSS_3p5) * x
    )

    return psi_SS


def _get_merger_frequency(theta: Array, kappa: float = None):
    """
    Computes the merger frequency in Hz of the given system. This is defined in equation (11) in https://arxiv.org/abs/1804.02235 and the lal source code.

    Args:
        theta (Array): Intrinsic parameters with order (m1, m2, chi1, chi2, lambda1, lambda2)
        kappa (float, optional): Tidal parameter kappa. Defaults to None, so that it is computed from the given parameters theta.

    Returns:
        float: The merger frequency in Hz.
    """

    # Compute auxiliary quantities
    m1, m2, _, _, _, _ = theta
    M = m1 + m2
    m1_s = m1 * gt
    m2_s = m2 * gt
    q = m1_s / m2_s

    if kappa is None:
        kappa = get_kappa(theta)
    kappa_2 = kappa * kappa

    # Initialize coefficients
    a_0 = 0.3586
    n_1 = 3.35411203e-2
    n_2 = 4.31460284e-5
    d_1 = 7.54224145e-2
    d_2 = 2.23626859e-4

    # Get ratio and prefactor
    num = 1.0 + n_1 * kappa + n_2 * kappa_2
    den = 1.0 + d_1 * kappa + d_2 * kappa_2
    Q_0 = a_0 * (q) ** (-1.0 / 2.0)

    # Dimensionless angular frequency of merger
    Momega_merger = Q_0 * (num / den)

    # Convert from angular frequency to frequency (divide by 2*pi) and then convert from dimensionless frequency to Hz
    fHz_merger = Momega_merger / (M * gt) / (TWO_PI)

    return fHz_merger


def _gen_IMRPhenomD_NRTidalv2(
    f: Array,
    theta_intrinsic: Array,
    theta_extrinsic: Array,
    bbh_amp: Array,
    bbh_psi: Array,
    no_taper: bool = False,
):
    """
    Master internal function to get the GW strain for given parameters. The function takes
    a BBH strain, computed from an underlying BBH approximant, e.g. IMRPhenomD, and applies the
    tidal corrections to it afterwards, according to equation (25) of the NRTidalv2 paper.

    Args:
        f (Array): Frequencies in Hz
        theta_intrinsic (Array): Internal parameters of the system: m1, m2, chi1, chi2, lambda1, lambda2
        theta_extrinsic (Array): Extrinsic parameters of the system: d_L, tc and phi_c
        h0_bbh (Array): The BBH strain of the underlying model (i.e. before applying tidal corrections).

    Returns:
        Array: Final complex-valued strain of GW.
    """

    # Compute x: see NRTidalv2 paper for definition
    m1, m2, _, _, _, _ = theta_intrinsic
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    x = (PI * M_s * f) ** (2.0 / 3.0)

    # Compute kappa
    kappa = get_kappa(theta=theta_intrinsic)

    # Compute amplitudes
    A_T = get_tidal_amplitude(x, theta_intrinsic, kappa, distance=theta_extrinsic[0])
    f_merger = _get_merger_frequency(theta_intrinsic, kappa)

    # Decide whether to include the Planck taper or not
    if no_taper:
        A_P = jnp.ones_like(f)
    else:
        A_P = get_planck_taper(f, f_merger)

    # Get tidal phase and spin corrections for BNS
    psi_T = get_tidal_phase(x, theta_intrinsic, kappa)
    psi_SS = get_spin_phase_correction(x, theta_intrinsic)

    # Assemble everything
    h0 = A_P * (bbh_amp + A_T) * jnp.exp(1.0j * -(bbh_psi + psi_T + psi_SS))

    return h0


def gen_IMRPhenomD_NRTidalv2(
    f: Array,
    params: Array,
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
) -> Array:
    """
    Generate NRTidalv2 frequency domain waveform following NRTidalv2 paper.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    lambda1: Dimensionless tidal deformability of primary object
    lambda2: Dimensionless tidal deformability of secondary object
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    f_ref: Reference frequency for the waveform

    Returns:
    --------
        h0 (array): Strain
    """

    # Get component masses
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    if use_lambda_tildes:
        lambda1, lambda2 = lambda_tildes_to_lambdas(
            jnp.array([params[4], params[5], m1, m2])
        )
    else:
        lambda1, lambda2 = params[4], params[5]
    chi1, chi2 = params[2], params[3]

    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    theta_extrinsic = params[6:]

    # Generate the BBH part:
    bbh_theta_intrinsic = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(bbh_theta_intrinsic)
    M_s = (bbh_theta_intrinsic[0] + bbh_theta_intrinsic[1]) * gt

    # Shift phase so that peak amplitude matches t = 0
    transition_freqs = get_transition_frequencies(
        bbh_theta_intrinsic, coeffs[5], coeffs[6]
    )
    _, _, _, f4, f_RD, f_damp = transition_freqs
    t0 = jax.grad(get_IIb_raw_phase)(
        f4 * M_s, bbh_theta_intrinsic, coeffs, f_RD, f_damp
    )

    # Call the amplitude and phase now
    Psi = Phase(f, bbh_theta_intrinsic, coeffs, transition_freqs)
    Psi_ref = Phase(f_ref, bbh_theta_intrinsic, coeffs, transition_freqs)
    Mf_ref = f_ref * M_s
    Psi -= t0 * ((f * M_s) - Mf_ref) + Psi_ref
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] - 2 * theta_extrinsic[2]
    Psi += ext_phase_contrib
    fcut_above = lambda f: (fM_CUT / M_s)
    fcut_below = lambda f: f[jnp.abs(f - (fM_CUT / M_s)).argmin() - 1]
    fcut_true = jax.lax.cond((fM_CUT / M_s) > f[-1], fcut_above, fcut_below, f)
    Psi = Psi * jnp.heaviside(fcut_true - f, 0.0) + 2.0 * PI * jnp.heaviside(
        f - fcut_true, 1.0
    )

    A = Amp(f, bbh_theta_intrinsic, coeffs, transition_freqs, D=theta_extrinsic[0])

    bbh_amp = A
    bbh_psi = Psi

    # Use BBH waveform and add tidal corrections
    return _gen_IMRPhenomD_NRTidalv2(
        f, theta_intrinsic, theta_extrinsic, bbh_amp, bbh_psi, no_taper=no_taper
    )


def gen_IMRPhenomD_NRTidalv2_hphc(
    f: Array,
    params: Array,
    f_ref: float,
    use_lambda_tildes: bool = True,
    no_taper: bool = False,
):
    """
    vars array contains both intrinsic and extrinsic variables

    IMRphenom denotes the name of the underlying BBH approximant used, before applying tidal corrections.

    theta = [Mchirp, eta, chi1, chi2, lambda1, lambda2, D, tc, phic, inclination]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence
    inclination: Inclination angle of the binary [between 0 and PI]

    f_ref: Reference frequency for the waveform

    Returns:
    --------
        hp (array): Strain of the plus polarization
        hc (array): Strain of the cross polarization
    """
    iota = params[-1]
    h0 = gen_IMRPhenomD_NRTidalv2(
        f, params[:-1], f_ref, use_lambda_tildes=use_lambda_tildes, no_taper=no_taper
    )

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
