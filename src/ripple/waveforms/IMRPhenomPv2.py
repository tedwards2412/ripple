import jax
import jax.numpy as jnp
from ripple import Mc_eta_to_ms

from ..constants import gt, MSUN
import numpy as np
from .IMRPhenomD import Phase as PhDPhase
from .IMRPhenomD import Amp as PhDAmp
from .IMRPhenomD_utils import get_coeffs

from ..typing import Array
from .IMRPhenomPv2_utils import *
from .IMRPhenomD_utils import *


def PhenomPCoreTwistUp(
    fHz,
    hPhenom,
    # phase,
    # Amp,
    eta,
    chi1_l,
    chi2_l,
    chip,
    M,
    angcoeffs,
    Y2m,
    alphaoffset,
    epsilonoffset,
):
    assert angcoeffs is not None
    assert Y2m is not None

    # here it is used to be LAL_MTSUN_SI
    f = fHz * gt * M  # Frequency in geometric units
    q = (1.0 + jnp.sqrt(1.0 - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
    m1 = 1.0 / (1.0 + q)  # Mass of the smaller BH for unit total mass M=1.
    m2 = q / (1.0 + q)  # Mass of the larger BH for unit total mass M=1.
    Sperp = chip * (
        m2 * m2
    )  # Dimensionfull spin component in the orbital plane. S_perp = S_2_perp
    # chi_eff = m1 * chi1_l + m2 * chi2_l  # effective spin for M=1

    SL = chi1_l * m1 * m1 + chi2_l * m2 * m2  # Dimensionfull aligned spin.

    omega = jnp.pi * f
    logomega = jnp.log(omega)
    omega_cbrt = (omega) ** (1 / 3)
    omega_cbrt2 = omega_cbrt * omega_cbrt

    alpha = (
        angcoeffs["alphacoeff1"] / omega
        + angcoeffs["alphacoeff2"] / omega_cbrt2
        + angcoeffs["alphacoeff3"] / omega_cbrt
        + angcoeffs["alphacoeff4"] * logomega
        + angcoeffs["alphacoeff5"] * omega_cbrt
    ) - alphaoffset

    epsilon = (
        angcoeffs["epsiloncoeff1"] / omega
        + angcoeffs["epsiloncoeff2"] / omega_cbrt2
        + angcoeffs["epsiloncoeff3"] / omega_cbrt
        + angcoeffs["epsiloncoeff4"] * logomega
        + angcoeffs["epsiloncoeff5"] * omega_cbrt
    ) - epsilonoffset

    # print("alpha, epsilon: ", alpha, epsilon)
    cBetah, sBetah = WignerdCoefficients(omega_cbrt, SL, eta, Sperp)

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah2 * cBetah
    cBetah4 = cBetah3 * cBetah
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah2 * sBetah
    sBetah4 = sBetah3 * sBetah

    # d2 = jnp.array(
    #     [
    #         sBetah4,
    #         2 * cBetah * sBetah3,
    #         jnp.sqrt(6) * sBetah2 * cBetah2,
    #         2 * cBetah3 * sBetah,
    #         cBetah4,
    #     ]
    # )
    Y2mA = jnp.array(Y2m)  # need to pass Y2m in a 5-component list
    hp_sum = 0
    hc_sum = 0

    cexp_i_alpha = jnp.exp(1j * alpha)
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    T2m = (
        cexp_2i_alpha * cBetah4 * Y2mA[0]
        - cexp_i_alpha * 2 * cBetah3 * sBetah * Y2mA[1]
        + 1 * jnp.sqrt(6) * sBetah2 * cBetah2 * Y2mA[2]
        - cexp_mi_alpha * 2 * cBetah * sBetah3 * Y2mA[3]
        + cexp_m2i_alpha * sBetah4 * Y2mA[4]
    )
    Tm2m = (
        cexp_m2i_alpha * sBetah4 * jnp.conjugate(Y2mA[0])
        + cexp_mi_alpha * 2 * cBetah * sBetah3 * jnp.conjugate(Y2mA[1])
        + 1 * jnp.sqrt(6) * sBetah2 * cBetah2 * jnp.conjugate(Y2mA[2])
        + cexp_i_alpha * 2 * cBetah3 * sBetah * jnp.conjugate(Y2mA[3])
        + cexp_2i_alpha * cBetah4 * jnp.conjugate(Y2mA[4])
    )
    hp_sum = T2m + Tm2m
    hc_sum = 1j * (T2m - Tm2m)
    eps_phase_hP = jnp.exp(-2j * epsilon) * hPhenom / 2.0

    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return hp, hc


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


# def PhenomPOneFrequency_phase(
#     f: float,
#     m1: float,
#     m2: float,
#     chi1: float,
#     chi2: float,
#     chip: float,
#     phic: float,
#     M: float,
#     dist_mpc: float,
# ):
#     """ """
#     theta_ripple = jnp.array([m1, m2, chi1, chi2])
#     coeffs = get_coeffs(theta_ripple)
#     transition_freqs = phP_get_transition_frequencies(
#         theta_ripple, coeffs[5], coeffs[6], chip
#     )

#     phase = PhDPhase(f, theta_ripple, coeffs, transition_freqs)
#     return -phase


def gen_IMRPhenomPv2(
    fs: Array,
    theta: Array,
    f_ref: float,
):
    """
    Thetas are waveform parameters.
    m1 must be larger than m2.
    """
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phiRef, incl = theta

    # flip m1 m2. For some reason LAL uses this convention for PhenomPv2
    m1, m2 = m2, m1
    s1x, s2x = s2x, s1x
    s1y, s2y = s2y, s1y
    s1z, s2z = s2z, s1z
    # from now on, m1 < m2

    # m1_SI = m1 * MSUN
    # m2_SI = m2 * MSUN
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
    theta_intrinsic = jnp.array([m2, m1, chi2_l, chi1_l])
    coeffs = get_coeffs(theta_intrinsic)

    transition_freqs = phP_get_transition_frequencies(
        theta_intrinsic, coeffs[5], coeffs[6], chip
    )

    hPhenomDs, phi_IIb = PhenomPOneFrequency(
        fs, m2, m1, chi2_l, chi1_l, chip, phic, M, dist_mpc, coeffs, transition_freqs
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
    # unpack transition_freqs
    _, _, _, _, f_RD, _ = transition_freqs

    # phi_IIb = lambda f: PhenomPOneFrequency_phase(
    #     f, m2, m1, chi2_l, chi1_l, chip, phiRef, M, dist_mpc
    # )
    t0 = jax.grad(phi_IIb)(f_RD) / (2 * jnp.pi)
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


def gen_IMRPhenomPv2_hphc(f: Array, params: Array, f_ref: float):
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
        ]
    )
    hp, hc = gen_IMRPhenomPv2(f, m1m2params, f_ref)
    return hp, hc
