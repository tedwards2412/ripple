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


# helper functions for LALtoPhenomP:
def ROTATEZ(angle, x, y, z):
    tmp_x = x * jnp.cos(angle) - y * jnp.sin(angle)
    tmp_y = x * jnp.sin(angle) + y * jnp.cos(angle)
    return tmp_x, tmp_y, z


def ROTATEY(angle, x, y, z):
    tmp_x = x * jnp.cos(angle) + z * jnp.sin(angle)
    tmp_z = -x * jnp.sin(angle) + z * jnp.cos(angle)
    return tmp_x, y, tmp_z


def convert_spins(
    m1_SI: float,
    m2_SI: float,
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
    m1 = m1_SI / MSUN  # Masses in solar masses
    m2 = m2_SI / MSUN
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
