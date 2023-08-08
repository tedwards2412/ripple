import jax
import jax.numpy as jnp
from ripple import Mc_eta_to_ms

from typing import Tuple
from ..constants import gt, MSUN
import numpy as np
from .IMRPhenomD import Phase as PhDPhase
from .IMRPhenomD import Amp as PhDAmp
from .IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
    EradRational0815,
)
from ..typing import Array
from .IMRPhenomD_QNMdata import QNMData_a, QNMData_fRD, QNMData_fdamp
from .IMRPhenomPv2_utils import *
from .IMRPhenomD_utils import *


def PhenomPCoreTwistUp(
    fHz,
    hPhenom,
    phase,
    Amp,
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
    chi_eff = m1 * chi1_l + m2 * chi2_l  # effective spin for M=1

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

    d2 = jnp.array(
        [
            sBetah4,
            2 * cBetah * sBetah3,
            jnp.sqrt(6) * sBetah2 * cBetah2,
            2 * cBetah3 * sBetah,
            cBetah4,
        ]
    )
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


def L2PNR(v: float, eta: float) -> float:
    eta2 = eta**2
    x = v**2
    x2 = x**2
    return (
        eta
        * (
            1.0
            + (1.5 + eta / 6.0) * x
            + (3.375 - (19.0 * eta) / 8.0 - eta2 / 24.0) * x2
        )
    ) / x**0.5


def WignerdCoefficients(v: float, SL: float, eta: float, Sp: float):
    # We define the shorthand s := Sp / (L + SL)
    L = L2PNR(v, eta)
    s = Sp / (L + SL)
    s2 = s**2
    cos_beta = 1.0 / (1.0 + s2) ** 0.5
    cos_beta_half = ((1.0 + cos_beta) / 2.0) ** 0.5  # cos(beta/2)
    sin_beta_half = ((1.0 - cos_beta) / 2.0) ** 0.5  # sin(beta/2)

    return cos_beta_half, sin_beta_half


def ComputeNNLOanglecoeffs(q, chil, chip):
    m2 = q / (1.0 + q)
    m1 = 1.0 / (1.0 + q)
    dm = m1 - m2
    mtot = 1.0
    eta = m1 * m2  # mtot = 1
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    mtot2 = mtot * mtot
    mtot4 = mtot2 * mtot2
    mtot6 = mtot4 * mtot2
    mtot8 = mtot6 * mtot2
    chil2 = chil * chil
    chip2 = chip * chip
    chip4 = chip2 * chip2
    dm2 = dm * dm
    dm3 = dm2 * dm
    m2_2 = m2 * m2
    m2_3 = m2_2 * m2
    m2_4 = m2_3 * m2
    m2_5 = m2_4 * m2
    m2_6 = m2_5 * m2
    m2_7 = m2_6 * m2
    m2_8 = m2_7 * m2

    angcoeffs = {}
    angcoeffs["alphacoeff1"] = -0.18229166666666666 - (5 * dm) / (64.0 * m2)

    angcoeffs["alphacoeff2"] = (-15 * dm * m2 * chil) / (128.0 * mtot2 * eta) - (
        35 * m2_2 * chil
    ) / (128.0 * mtot2 * eta)

    angcoeffs["alphacoeff3"] = (
        -1.7952473958333333
        - (4555 * dm) / (7168.0 * m2)
        - (15 * chip2 * dm * m2_3) / (128.0 * mtot4 * eta2)
        - (35 * chip2 * m2_4) / (128.0 * mtot4 * eta2)
        - (515 * eta) / 384.0
        - (15 * dm2 * eta) / (256.0 * m2_2)
        - (175 * dm * eta) / (256.0 * m2)
    )

    angcoeffs["alphacoeff4"] = (
        -(35 * jnp.pi) / 48.0
        - (5 * dm * jnp.pi) / (16.0 * m2)
        + (5 * dm2 * chil) / (16.0 * mtot2)
        + (5 * dm * m2 * chil) / (3.0 * mtot2)
        + (2545 * m2_2 * chil) / (1152.0 * mtot2)
        - (5 * chip2 * dm * m2_5 * chil) / (128.0 * mtot6 * eta3)
        - (35 * chip2 * m2_6 * chil) / (384.0 * mtot6 * eta3)
        + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
        + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
    )

    angcoeffs["alphacoeff5"] = (
        4.318908476114694
        + (27895885 * dm) / (2.1676032e7 * m2)
        - (15 * chip4 * dm * m2_7) / (512.0 * mtot8 * eta4)
        - (35 * chip4 * m2_8) / (512.0 * mtot8 * eta4)
        - (485 * chip2 * dm * m2_3) / (14336.0 * mtot4 * eta2)
        + (475 * chip2 * m2_4) / (6144.0 * mtot4 * eta2)
        + (15 * chip2 * dm2 * m2_2) / (256.0 * mtot4 * eta)
        + (145 * chip2 * dm * m2_3) / (512.0 * mtot4 * eta)
        + (575 * chip2 * m2_4) / (1536.0 * mtot4 * eta)
        + (39695 * eta) / 86016.0
        + (1615 * dm2 * eta) / (28672.0 * m2_2)
        - (265 * dm * eta) / (14336.0 * m2)
        + (955 * eta2) / 576.0
        + (15 * dm3 * eta2) / (1024.0 * m2_3)
        + (35 * dm2 * eta2) / (256.0 * m2_2)
        + (2725 * dm * eta2) / (3072.0 * m2)
        - (15 * dm * m2 * jnp.pi * chil) / (16.0 * mtot2 * eta)
        - (35 * m2_2 * jnp.pi * chil) / (16.0 * mtot2 * eta)
        + (15 * chip2 * dm * m2_7 * chil2) / (128.0 * mtot8 * eta4)
        + (35 * chip2 * m2_8 * chil2) / (128.0 * mtot8 * eta4)
        + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
        + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
        + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
    )

    angcoeffs["epsiloncoeff1"] = -0.18229166666666666 - (5 * dm) / (64.0 * m2)
    angcoeffs["epsiloncoeff2"] = (-15 * dm * m2 * chil) / (128.0 * mtot2 * eta) - (
        35 * m2_2 * chil
    ) / (128.0 * mtot2 * eta)
    angcoeffs["epsiloncoeff3"] = (
        -1.7952473958333333
        - (4555 * dm) / (7168.0 * m2)
        - (515 * eta) / 384.0
        - (15 * dm2 * eta) / (256.0 * m2_2)
        - (175 * dm * eta) / (256.0 * m2)
    )
    angcoeffs["epsiloncoeff4"] = (
        -(35 * jnp.pi) / 48.0
        - (5 * dm * jnp.pi) / (16.0 * m2)
        + (5 * dm2 * chil) / (16.0 * mtot2)
        + (5 * dm * m2 * chil) / (3.0 * mtot2)
        + (2545 * m2_2 * chil) / (1152.0 * mtot2)
        + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
        + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
    )
    angcoeffs["epsiloncoeff5"] = (
        4.318908476114694
        + (27895885 * dm) / (2.1676032e7 * m2)
        + (39695 * eta) / 86016.0
        + (1615 * dm2 * eta) / (28672.0 * m2_2)
        - (265 * dm * eta) / (14336.0 * m2)
        + (955 * eta2) / 576.0
        + (15 * dm3 * eta2) / (1024.0 * m2_3)
        + (35 * dm2 * eta2) / (256.0 * m2_2)
        + (2725 * dm * eta2) / (3072.0 * m2)
        - (15 * dm * m2 * jnp.pi * chil) / (16.0 * mtot2 * eta)
        - (35 * m2_2 * jnp.pi * chil) / (16.0 * mtot2 * eta)
        + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
        + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
        + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
    )
    return angcoeffs


def FinalSpin0815_s(eta, s):
    eta2 = eta * eta
    eta3 = eta2 * eta
    s2 = s * s
    s3 = s2 * s
    return eta * (
        3.4641016151377544
        - 4.399247300629289 * eta
        + 9.397292189321194 * eta2
        - 13.180949901606242 * eta3
        + s
        * (
            (1.0 / eta - 0.0850917821418767 - 5.837029316602263 * eta)
            + (0.1014665242971878 - 2.0967746996832157 * eta) * s
            + (-1.3546806617824356 + 4.108962025369336 * eta) * s2
            + (-0.8676969352555539 + 2.064046835273906 * eta) * s3
        )
    )


def FinalSpin0815(eta, chi1, chi2):
    Seta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1 * m1
    m2s = m2 * m2
    s = m1s * chi1 + m2s * chi2
    return FinalSpin0815_s(eta, s)


def FinalSpin_inplane(m1, m2, chi1_l, chi2_l, chip):
    M = m1 + m2
    eta = m1 * m2 / (M * M)
    # Here I assume m1 > m2, the convention used in phenomD
    # (not the convention of internal phenomP)
    q_factor = m1 / M
    af_parallel = FinalSpin0815(eta, chi1_l, chi2_l)
    Sperp = chip * q_factor * q_factor
    af = jnp.copysign(1.0, af_parallel) * jnp.sqrt(
        Sperp * Sperp + af_parallel * af_parallel
    )
    return af


def phP_get_fRD_fdamp(m1, m2, chi1_l, chi2_l, chip):
    # m1 > m2 should hold here
    finspin = FinalSpin_inplane(m1, m2, chi1_l, chi2_l, chip)
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s**2.0)
    fRD = jnp.interp(finspin, QNMData_a, QNMData_fRD) / (
        1.0 - EradRational0815(eta_s, chi1_l, chi2_l)
    )
    fdamp = jnp.interp(finspin, QNMData_a, QNMData_fdamp) / (
        1.0 - EradRational0815(eta_s, chi1_l, chi2_l)
    )

    return fRD / M_s, fdamp / M_s


def phP_get_transition_frequencies(
    theta: Array,
    gamma2: float,
    gamma3: float,
    chip: float,
) -> Tuple[float, float, float, float, float, float]:
    # m1 > m2 should hold here

    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    f_RD, f_damp = phP_get_fRD_fdamp(m1, m2, chi1, chi2, chip)

    # Phase transition frequencies
    f1 = 0.018 / (M * gt)
    f2 = 0.5 * f_RD

    # Amplitude transition frequencies
    f3 = 0.014 / (M * gt)
    f4_gammaneg_gtr_1 = lambda f_RD_, f_damp_, gamma3_, gamma2_: jnp.abs(
        f_RD_ + (-f_damp_ * gamma3_) / gamma2_
    )
    f4_gammaneg_less_1 = lambda f_RD_, f_damp_, gamma3_, gamma2_: jnp.abs(
        f_RD_ + (f_damp_ * (-1 + jnp.sqrt(1 - (gamma2_) ** 2.0)) * gamma3_) / gamma2_
    )
    f4 = jax.lax.cond(
        gamma2 >= 1,
        f4_gammaneg_gtr_1,
        f4_gammaneg_less_1,
        f_RD,
        f_damp,
        gamma3,
        gamma2,
    )
    return f1, f2, f3, f4, f_RD, f_damp


def PhenomPOneFrequency(fsHz, m1, m2, chi1, chi2, chip, phic, M, dist_mpc):
    """
    m1, m2: in solar masses
    phic: Orbital phase at the peak of the underlying non precessing model (rad)
    M: Total mass (Solar masses)
    """
    # These are the parametrs that go into the waveform generator
    # Note that JAX does not give index errors, so if you pass in the
    # the wrong array it will behave strangely
    magicalnumber = 2.0 * jnp.sqrt(5.0 / (64.0 * jnp.pi))
    f = fsHz  # * MSUN * M
    theta_ripple = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(theta_ripple)
    transition_freqs = phP_get_transition_frequencies(
        theta_ripple, coeffs[5], coeffs[6], chip
    )

    phase = PhDPhase(f, theta_ripple, coeffs, transition_freqs)

    phase -= phic
    Amp = PhDAmp(f, theta_ripple, coeffs, transition_freqs, D=dist_mpc) / magicalnumber

    # phase -= 2. * phic; # line 1316 ???
    hPhenom = Amp * (jnp.exp(-1j * phase))
    return hPhenom, phase, Amp


def PhenomPOneFrequency_phase(
    f: float,
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    chip: float,
    phic: float,
    M: float,
    dist_mpc: float,
):
    """
    m1, m2: in solar masses
    phic: Orbital phase at the peak of the underlying non precessing model (rad)
    M: Total mass (Solar masses)
    """
    # print("inside:", dist_mpc)
    # These are the parametrs that go into the waveform generator
    # Note that JAX does not give index errors, so if you pass in the
    # the wrong array it will behave strangely
    theta_ripple = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(theta_ripple)
    transition_freqs = phP_get_transition_frequencies(
        theta_ripple, coeffs[5], coeffs[6], chip
    )

    phase = PhDPhase(f, theta_ripple, coeffs, transition_freqs)
    return -phase


def time_corr(m1, m2, chi1_l, chi2_l, chip, phiRef, M, dist_mpc):
    """
    here m1 > m2
    """
    theta_intrinsic = jnp.array([m1, m2, chi1_l, chi2_l])
    coeffs = get_coeffs(theta_intrinsic)

    theta_ripple = jnp.array([m1, m2, chi1_l, chi2_l])

    transition_freqs = phP_get_transition_frequencies(
        theta_ripple, coeffs[5], coeffs[6], chip
    )
    # unpack transition_freqs
    _, _, _, _, f_RD, _ = transition_freqs
    phi_IIb = lambda f: PhenomPOneFrequency_phase(
        f, m1, m2, chi1_l, chi2_l, chip, phiRef, M, dist_mpc
    )
    t0 = jax.grad(phi_IIb)(f_RD) / (2 * jnp.pi)
    return t0


def PhenomPcore(
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

    m1_SI = m1 * MSUN
    m2_SI = m2 * MSUN
    (
        chi1_l,
        chi2_l,
        chip,
        thetaJN,
        alpha0,
        phi_aligned,
        zeta_polariz,
    ) = Pv2utils.convert_spins(
        m1_SI, m2_SI, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z
    )
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

    Y2m2 = Pv2utils.SpinWeightedY(thetaJN, 0, -2, 2, -2)
    Y2m1 = Pv2utils.SpinWeightedY(thetaJN, 0, -2, 2, -1)
    Y20 = Pv2utils.SpinWeightedY(thetaJN, 0, -2, 2, -0)
    Y21 = Pv2utils.SpinWeightedY(thetaJN, 0, -2, 2, 1)
    Y22 = Pv2utils.SpinWeightedY(thetaJN, 0, -2, 2, 2)
    Y2 = [Y2m2, Y2m1, Y20, Y21, Y22]

    hPhenomDs, phase, Amp = PhenomPOneFrequency(
        fs, m2, m1, chi2_l, chi1_l, chip, phic, M, dist_mpc
    )

    hp, hc = PhenomPCoreTwistUp(
        fs,
        hPhenomDs,
        phase,
        Amp,
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

    # Shift phase so that peak amplitude matches t = 0
    theta_intrinsic = jnp.array([m2, m1, chi2_l, chi1_l])
    coeffs = get_coeffs(theta_intrinsic)

    theta_ripple = jnp.array([m2, m1, chi2_l, chi1_l])

    transition_freqs = phP_get_transition_frequencies(
        theta_ripple, coeffs[5], coeffs[6], chip
    )
    # unpack transition_freqs
    _, _, _, _, f_RD, _ = transition_freqs

    phi_IIb = lambda f: PhenomPOneFrequency_phase(
        f, m2, m1, chi2_l, chi1_l, chip, phiRef, M, dist_mpc
    )
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


def gen_IMRPhenomP_hphc(f: Array, params: Array, f_ref: float):
    """
    wrapper around phenomPcore but the first two parameters are Mc and eta
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
        ]
    )
    hp, hc = PhenomPcore(f, m1m2params, f_ref)
    return hp, hc
