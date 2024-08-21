import jax
import jax.numpy as jnp
from ripplegw import Mc_eta_to_ms

from ..constants import gt, MSUN
import numpy as np
from .IMRPhenomD import Phase as PhDPhase
from .IMRPhenomD import Amp as PhDAmp
from .IMRPhenomD_utils import get_coeffs
from .IMRPhenomD_NRTidalv2 import get_planck_taper, _get_merger_frequency
from .IMRPhenom_tidal_utils import get_kappa
from .IMRPhenomPv2 import PhenomPCoreTwistUp

from ..typing import Array
from .IMRPhenomPv2_utils import *
from .IMRPhenomD_utils import *

def get_tidal_phase(fHz, Xa, Xb, total_mass, kappa):
    """ 
    Computes the tidal phase from the NRTidalv1 model. 

    Parameters
    ----------
    fHz : jnp.array[float64]
        Array of frequencies in Hz.
    Xa : jnp.array[float64]
        Mass/total_mass of first object
    Xb : jnp.array[float64]
        Mass/total_mass of second object
    total_mass : jnp.array[float64]
        Total mass of binary system
    kappa : jnp.array[float64]
        Tidal coupling constant
        
    Returns
    ------- 
    tidal_phase : jnp.array[float64]
        Tidal phase from NRTidalv1 model
    """
    # Constants
    c_Newt = 2.4375
    n_1 = -17.428
    n_3over2 = 31.867
    n_2 = -26.414
    n_5over2 = 62.362
    d_1 = n_1 - 2.496
    d_3over2 = 36.089

    # Dimensionless angular GW frequency
    M_omega = jnp.pi * fHz * (total_mass * gt)

    # Calculating powers of the frequency term
    PN_x = jnp.power(M_omega, 2.0 / 3.0)
    PN_x_3over2 = jnp.power(PN_x, 3.0 / 2.0)
    PN_x_5over2 = jnp.power(PN_x, 5.0 / 2.0)

    # Tidal phase calculation
    tidal_phase = -kappa * c_Newt / (Xa * Xb) * PN_x_5over2

    # Numerator and denominator for ratio
    num = (
        1.0
        + (n_1 * PN_x)
        + (n_3over2 * PN_x_3over2)
        + (n_2 * PN_x * PN_x)
        + (n_5over2 * PN_x_5over2)
    )
    den = 1.0 + (d_1 * PN_x) + (d_3over2 * PN_x_3over2)

    # Ratio
    ratio = num / den

    # Final tidal phase calculation
    tidal_phase *= ratio

    return tidal_phase


def get_nr_tuned_tidal_phase_taper(fHz, m1, m2, lambda1, lambda2):
    """
    Computes the NRTidalv1 model's tidal phase taper for a binary system with tuned parameters.

    Parameters
    ----------
    fHz : jnp.array[float64]
        Array of frequencies in Hertz at which to compute the tidal phase taper.
    m1 : jnp.array[float64]
        Mass of the first object in solar masses.
    m2 : jnp.array[float64]
        Mass of the second object in solar masses.
    lambda1 : jnp.array[float64]
        Tidal deformability parameter for the first object. Represents the tidal coupling strength.
    lambda2 : jnp.array[float64]
        Tidal deformability parameter for the second object. Represents the tidal coupling strength.

    Returns
    -------
    tidal_phase_taper : jnp.array[float64]
        Array of tidal phase taper values computed using the NRTidalv1 model. 
    """

    total_mass = m1 + m2
    q = m1 / m2

    # Xa and Xb are the masses normalized for total_mass = 1
    Xa = m1 / total_mass
    Xb = m2 / total_mass


    kappa = get_kappa([m1, m2, 0, 0, lambda1, lambda2])
    fHz_mrg = _get_merger_frequency([m1, m2, 0, 0, 0, 0], kappa)

    phi_tidal = get_tidal_phase(fHz, Xa, Xb, total_mass, kappa)
    planck_taper = get_planck_taper(fHz, fHz_mrg)

    return phi_tidal, planck_taper


def PhenomPOneFrequency(
    fs, m1, m2, chi1, chi2, chip, phic, M, dist_mpc, coeffs, transition_freqs
):
    """
    m1, m2: in solar masses
    phic: Orbital phase at the peak of the underlying non precessing model (rad)
    M: Total mass (Solar masses)
    """
    # These are the parametrs that go into the waveform generator
    # Note that JAX does not give index errors, so if you pass in the
    # the wrong array it will behave strangely
    norm = 2.0 * jnp.sqrt(5.0 / (64.0 * jnp.pi))
    theta_ripple = jnp.array([m1, m2, chi1, chi2])
    # coeffs = get_coeffs(theta_ripple)
    # transition_freqs = phP_get_transition_frequencies(
    #     theta_ripple, coeffs[5], coeffs[6], chip
    # )

    phase = PhDPhase(fs, theta_ripple, coeffs, transition_freqs)
    Dphi = lambda f: -PhDPhase(f, theta_ripple, coeffs, transition_freqs)

    phase -= phic
    Amp = PhDAmp(fs, theta_ripple, coeffs, transition_freqs, D=dist_mpc) / norm

    # phase -= 2. * phic; # line 1316 ???
    hPhenom = Amp * (jnp.exp(-1j * phase))
    return hPhenom, Dphi


def PhenomPOneFrequencyWithTides(
    fs,
    m1,
    m2,
    chi1,
    chi2,
    chip,
    lambda1,
    lambda2,
    phic,
    M,
    dist_mpc,
    coeffs,
    transition_freqs,
):
    """
    Computes the gravitational waveform frequency domain representation for the PhenomP model, incorporating tidal effects.

    Parameters
    ----------
    fs : jnp.array[float64]
        Array of frequencies in Hertz at which to evaluate the waveform.

    m1 : float
        Mass of the first object in solar masses.

    m2 : float
        Mass of the second object in solar masses.

    chi1 : float
        Dimensionless spin parameter of the first object. Defines the spin orientation and magnitude.

    chi2 : float
        Dimensionless spin parameter of the second object. Defines the spin orientation and magnitude.

    chip : float
        Effective spin parameter, which combines the individual spins and their orientations relative to the orbital angular momentum.

    lambda1 : float
        Tidal deformability parameter for the first object. Represents the strength of the tidal interactions.

    lambda2 : float
        Tidal deformability parameter for the second object. Represents the strength of the tidal interactions.

    phic : float
        Coalescence phase of the binary system, which shifts the phase of the waveform.

    M : float
        Total mass of the binary system in solar masses.

    dist_mpc : float
        Distance to the binary system in megaparsecs.

    coeffs : jnp.array[float64]
        Coefficients used in the model to account for various waveform parameters and tidal effects.

    transition_freqs : jnp.array[float64]
        Array of transition frequencies that characterize the changes in the waveform model due to different physical effects.

    Returns
    -------
    hPhenom : jnp.array[float64]
        Gravitational waveform strain in the frequency domain, including tidal effects.
    
    dPhi : function
        Way to evaluate phase (including tides) at a particular point. We 
        will take the derivative of this function later 
    """

    # These are the parametrs that go into the waveform generator
    # Note that JAX does not give index errors, so if you pass in the
    # the wrong array it will behave strangely
    norm = 2.0 * jnp.sqrt(5.0 / (64.0 * jnp.pi))
    theta_ripple = jnp.array([m1, m2, chi1, chi2])

    # getting amplitude and phase terms
    ampTidal = 0.0 # unused
    phaseTidal, planckTaper = get_nr_tuned_tidal_phase_taper(
        fs, m1, m2, lambda1, lambda2
    )

    phase = PhDPhase(fs, theta_ripple, coeffs, transition_freqs)
    Dphi = lambda f: -PhDPhase(f, theta_ripple, coeffs, transition_freqs) - get_nr_tuned_tidal_phase_taper(f, m1, m2, lambda1, lambda2)[0] 
    phase -= phic

    Amp = PhDAmp(fs, theta_ripple, coeffs, transition_freqs, D=dist_mpc) / norm
    hPhenom = Amp * jnp.exp(-1j * (phase + phaseTidal)) * planckTaper 

    return hPhenom, Dphi


def gen_IMRPhenomPv2_NRTidal(
    fs: Array,
    theta: Array,
    f_ref: float,
):
    """
    Thetas are waveform parameters.
    m1 must be larger than m2.
    """
    (
        m1,
        m2,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        dist_mpc,
        tc,
        phiRef,
        incl,
        lambda1,
        lambda2,
    ) = theta

    # flip m1 m2. For some reason LAL uses this convention for PhenomPv2
    m1, m2 = m2, m1
    lambda1, lambda2 = lambda2, lambda1
    s1x, s2x = s2x, s1x
    s1y, s2y = s2y, s1y
    s1z, s2z = s2z, s1z
    # from now on, m1 < m2

    (
        chi1_l,
        chi2_l,
        chip,
        thetaJN,
        alpha0,
        phi_aligned,
        zeta_polariz,
    ) = convert_spins(m1, m2, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z)
    phic = 2 * phi_aligned
    q = m2 / m1  # q>=1
    M = m1 + m2
    chi_eff = (m1 * chi1_l + m2 * chi2_l) / M
    chil = (1.0 + q) / q * chi_eff
    eta = m1 * m2 / (M * M)
    m_sec = M * gt
    piM = jnp.pi * m_sec

    omega_ref = piM * f_ref
    logomega_ref = jnp.log(omega_ref)
    omega_ref_cbrt = (piM * f_ref) ** (1 / 3)  # == v0
    omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

    angcoeffs = ComputeNNLOanglecoeffs(q, chil, chip)

    alphaNNLOoffset = (
        angcoeffs["alphacoeff1"] / omega_ref
        + angcoeffs["alphacoeff2"] / omega_ref_cbrt2
        + angcoeffs["alphacoeff3"] / omega_ref_cbrt
        + angcoeffs["alphacoeff4"] * logomega_ref
        + angcoeffs["alphacoeff5"] * omega_ref_cbrt
    )

    epsilonNNLOoffset = (
        angcoeffs["epsiloncoeff1"] / omega_ref
        + angcoeffs["epsiloncoeff2"] / omega_ref_cbrt2
        + angcoeffs["epsiloncoeff3"] / omega_ref_cbrt
        + angcoeffs["epsiloncoeff4"] * logomega_ref
        + angcoeffs["epsiloncoeff5"] * omega_ref_cbrt
    )

    Y2m2 = SpinWeightedY(thetaJN, 0, -2, 2, -2)
    Y2m1 = SpinWeightedY(thetaJN, 0, -2, 2, -1)
    Y20 = SpinWeightedY(thetaJN, 0, -2, 2, -0)
    Y21 = SpinWeightedY(thetaJN, 0, -2, 2, 1)
    Y22 = SpinWeightedY(thetaJN, 0, -2, 2, 2)
    Y2 = [Y2m2, Y2m1, Y20, Y21, Y22]

    # Shift phase so that peak amplitude matches t = 0

    # theta_intrinsic = jnp.array([m2, m1, chi2_l, chi1_l, lambda1, lambda2])
    theta_intrinsic = jnp.array([m2, m1, chi2_l, chi1_l])
    coeffs = get_coeffs(theta_intrinsic)
    transition_freqs = phP_get_transition_frequencies(
        theta_intrinsic, coeffs[5], coeffs[6], chip
    )


    hPhenomDs, phi_IIb = PhenomPOneFrequencyWithTides(
        fs,
        m2,
        m1,
        chi2_l,
        chi1_l,
        chip,
        lambda2,
        lambda1,
        phic,
        M,
        dist_mpc,
        coeffs,
        transition_freqs,
    )

    hp, hc = PhenomPCoreTwistUp(
        fs,
        hPhenomDs,
        # phase,
        # Amp,
        eta,
        chi1_l,
        chi2_l,
        chip,
        M,
        angcoeffs,
        Y2,
        alphaNNLOoffset - alpha0,
        epsilonNNLOoffset,
    )

    # for BNS the final frequency is not the same as BBHs
    kappa = get_kappa([m1, m2, 0., 0., lambda1, lambda2])
    f_merger = _get_merger_frequency([m2, m1, 0, 0, 0, 0], kappa)
    f_final = f_merger

    t0 = jax.grad(phi_IIb)(f_final) / (2 * jnp.pi)
    phase_corr = jnp.cos(2 * jnp.pi * fs * (t0)) - 1j * jnp.sin(2 * jnp.pi * fs * (t0))
    M_s = (m1 + m2) * gt
    phase_corr_tc = jnp.exp(-1j * fs * M_s * tc)
    hp *= phase_corr * phase_corr_tc
    hc *= phase_corr * phase_corr_tc

    # final touches to hp and hc, stolen from Scott
    c2z = jnp.cos(2 * zeta_polariz)
    s2z = jnp.sin(2 * zeta_polariz)
    final_hp = c2z * hp + s2z * hc
    final_hc = c2z * hc - s2z * hp
    return final_hp, final_hc


def gen_IMRPhenomPv2_NRTidal_hphc(f: Array, params: Array, f_ref: float):
    """
    wrapper around gen_Pph but the first two parameters are Mc and eta
    instead of m1 and m2
    """
    Mc = params[0]
    eta = params[1]
    m1, m2 = Mc_eta_to_ms(jnp.array([Mc, eta]))
    m1m2params = jnp.array(
        [
            m1,
            m2,
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            params[8],
            params[9],
            params[10],
            params[11],
            params[12],
            params[13]
        ]
    )
    hp, hc = gen_IMRPhenomPv2_NRTidal(f, m1m2params, f_ref)
    return hp, hc
