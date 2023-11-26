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
    FinalSpin0815_s,
)
from ..typing import Array
from .IMRPhenomD_QNMdata import QNMData_a, QNMData_fRD, QNMData_fdamp


# helper functions for LALtoPhenomP:
def ROTATEZ(angle, x, y, z):
    tmp_x = x * jnp.cos(angle) - y * jnp.sin(angle)
    tmp_y = x * jnp.sin(angle) + y * jnp.cos(angle)
    return tmp_x, tmp_y, z


def ROTATEY(angle, x, y, z):
    tmp_x = x * jnp.cos(angle) + z * jnp.sin(angle)
    tmp_z = -x * jnp.sin(angle) + z * jnp.cos(angle)
    return tmp_x, y, tmp_z


def FinalSpin0815(eta, chi1, chi2):
    Seta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1 * m1
    m2s = m2 * m2
    s = m1s * chi1 + m2s * chi2
    return FinalSpin0815_s(eta, s)


def convert_spins(
    m1: float,
    m2: float,
    f_ref: float,
    phiRef: float,
    incl: float,
    s1x: float,
    s1y: float,
    s1z: float,
    s2x: float,
    s2y: float,
    s2z: float,
) -> Tuple[float, float, float, float, float, float, float]:
    # m1 = m1_SI / MSUN  # Masses in solar masses
    # m2 = m2_SI / MSUN
    M = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    eta = m1 * m2 / (M * M)  # Symmetric mass-ratio

    # From the components in the source frame, we can easily determine
    # chi1_l, chi2_l, chip and phi_aligned, which we need to return.
    # We also compute the spherical angles of J,
    # which we need to transform to the J frame

    # Aligned spins
    chi1_l = s1z  # Dimensionless aligned spin on BH 1
    chi2_l = s2z  # Dimensionless aligned spin on BH 2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * jnp.sqrt(s1x**2 + s1y**2)
    S2_perp = m2_2 * jnp.sqrt(s2x**2 + s2y**2)

    A1 = 2 + (3 * m2) / (2 * m1)
    A2 = 2 + (3 * m1) / (2 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp
    num = jnp.maximum(ASp1, ASp2)
    den = A2 * m2_2  # warning: this assumes m2 > m1
    chip = num / den

    m_sec = M * gt
    piM = jnp.pi * m_sec
    v_ref = (piM * f_ref) ** (1 / 3)
    L0 = M * M * L2PNR(v_ref, eta)
    J0x_sf = m1_2 * s1x + m2_2 * s2x
    J0y_sf = m1_2 * s1y + m2_2 * s2y
    J0z_sf = L0 + m1_2 * s1z + m2_2 * s2z
    J0 = jnp.sqrt(J0x_sf * J0x_sf + J0y_sf * J0y_sf + J0z_sf * J0z_sf)

    thetaJ_sf = jnp.arccos(J0z_sf / J0)

    phiJ_sf = jnp.arctan2(J0y_sf, J0x_sf)

    phi_aligned = -phiJ_sf

    # First we determine kappa
    # in the source frame, the components of N are given in Eq (35c) of T1500606-v6
    Nx_sf = jnp.sin(incl) * jnp.cos(jnp.pi / 2.0 - phiRef)
    Ny_sf = jnp.sin(incl) * jnp.sin(jnp.pi / 2.0 - phiRef)
    Nz_sf = jnp.cos(incl)

    tmp_x = Nx_sf
    tmp_y = Ny_sf
    tmp_z = Nz_sf

    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)

    kappa = -jnp.arctan2(tmp_y, tmp_x)

    # Then we determine alpha0, by rotating LN
    tmp_x, tmp_y, tmp_z = 0, 0, 1
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    alpha0 = jnp.arctan2(tmp_y, tmp_x)

    # Finally we determine thetaJ, by rotating N
    tmp_x, tmp_y, tmp_z = Nx_sf, Ny_sf, Nz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)
    Nx_Jf, Nz_Jf = tmp_x, tmp_z
    thetaJN = jnp.arccos(Nz_Jf)

    # Finally, we need to redefine the polarizations:
    # PhenomP's polarizations are defined following Arun et al (arXiv:0810.5336)
    # i.e. projecting the metric onto the P,Q,N triad defined with P=NxJ/|NxJ| (see (2.6) in there).
    # By contrast, the triad X,Y,N used in LAL
    # ("waveframe" in the nomenclature of T1500606-v6)
    # is defined in e.g. eq (35) of this document
    # (via its components in the source frame; note we use the defautl Omega=Pi/2).
    # Both triads differ from each other by a rotation around N by an angle \zeta
    # and we need to rotate the polarizations accordingly by 2\zeta

    Xx_sf = -jnp.cos(incl) * jnp.sin(phiRef)
    Xy_sf = -jnp.cos(incl) * jnp.cos(phiRef)
    Xz_sf = jnp.sin(incl)
    tmp_x, tmp_y, tmp_z = Xx_sf, Xy_sf, Xz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    # Now the tmp_a are the components of X in the J frame
    # We need the polar angle of that vector in the P,Q basis of Arun et al
    # P = NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J frame
    PArunx_Jf = 0.0
    PAruny_Jf = -1.0
    PArunz_Jf = 0.0

    # Q = NxP
    QArunx_Jf = Nz_Jf
    QAruny_Jf = 0.0
    QArunz_Jf = -Nx_Jf

    # Calculate the dot products XdotPArun and XdotQArun
    XdotPArun = tmp_x * PArunx_Jf + tmp_y * PAruny_Jf + tmp_z * PArunz_Jf
    XdotQArun = tmp_x * QArunx_Jf + tmp_y * QAruny_Jf + tmp_z * QArunz_Jf

    zeta_polariz = jnp.arctan2(XdotQArun, XdotPArun)
    return chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz


def SpinWeightedY(theta, phi, s, l, m):
    "copied from SphericalHarmonics.c in LAL"
    if s == -2:
        if l == 2:
            if m == -2:
                fac = (
                    jnp.sqrt(5.0 / (64.0 * jnp.pi))
                    * (1.0 - jnp.cos(theta))
                    * (1.0 - jnp.cos(theta))
                )
            elif m == -1:
                fac = (
                    jnp.sqrt(5.0 / (16.0 * jnp.pi))
                    * jnp.sin(theta)
                    * (1.0 - jnp.cos(theta))
                )
            elif m == 0:
                fac = jnp.sqrt(15.0 / (32.0 * jnp.pi)) * jnp.sin(theta) * jnp.sin(theta)
            elif m == 1:
                fac = (
                    jnp.sqrt(5.0 / (16.0 * jnp.pi))
                    * jnp.sin(theta)
                    * (1.0 + jnp.cos(theta))
                )
            elif m == 2:
                fac = (
                    jnp.sqrt(5.0 / (64.0 * jnp.pi))
                    * (1.0 + jnp.cos(theta))
                    * (1.0 + jnp.cos(theta))
                )
            else:
                raise ValueError(f"Invalid mode s={s}, l={l}, m={m} - require |m| <= l")
    return fac * np.exp(1j * m * phi)


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
    Erad = EradRational0815(eta_s, chi1_l, chi2_l)
    fRD = jnp.interp(finspin, QNMData_a, QNMData_fRD) / (1.0 - Erad)
    fdamp = jnp.interp(finspin, QNMData_a, QNMData_fdamp) / (1.0 - Erad)

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
