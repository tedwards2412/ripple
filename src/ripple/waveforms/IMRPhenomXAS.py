# from math import PI
import jax
import jax.numpy as jnp
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ripple.waveforms import IMRPhenomX_utils
from ..typing import Array

from ripple import Mc_eta_to_ms

eqspin_indx = 10
uneqspin_indx = 39


def get_inspiral_phase(fM_s: Array, theta: Array, coeffs: Array) -> Array:
    """
    Calculate the inspiral phase for the IMRPhenomD waveform.
    """
    # First lets calculate some of the vairables that will be used below
    # Mass variables
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    eta2 = eta * eta
    eta3 = eta2 * eta
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))

    # Spin variables
    chia = chi1 - chi2

    chi1L2L = chi1 * chi2
    chi1L2 = chi1 * chi1
    chi1L3 = chi1 * chi1 * chi1
    chi2L2 = chi2 * chi2
    chi2L3 = chi2 * chi2 * chi2

    # These are the TaylorF2 terms used in IMRPhenomXAS
    phi0 = 1.0
    phi1 = 0.0
    phi2 = (3715.0 / 756.0 + (55.0 * eta) / 9.0) * PI ** (2.0 / 3.0)
    phi3 = (
        -16.0 * PI**2
        + (
            (
                113.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                - 76.0 * (chi1 + chi2) * eta
            )
            / 6.0
        )
        * PI
    )
    phi4 = (
        15293365.0 / 508032.0 + (27145.0 * eta) / 504.0 + (3085.0 * eta2) / 72.0
    ) * PI ** (4.0 / 3.0) + (
        (
            -5.0
            * (
                81.0 * chi1L2 * (1 + delta - 2 * eta)
                + 316.0 * chi1L2L * eta
                - 81.0 * chi2L2 * (-1 + delta + 2 * eta)
            )
        )
        / 16.0
    ) * PI ** (
        4.0 / 3.0
    )
    phi5 = 0.0
    phi5L = ((5 * (46374 - 6552 * eta) * PI) / 4536.0) * PI ** (5.0 / 3.0) + (
        (
            -732985 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
            - 560 * (-1213 * (chi1 + chi2) + 63 * (chi1 - chi2) * delta) * eta
            + 85680 * (chi1 + chi2) * eta2
        )
        / 4536.0
    ) * PI ** (5.0 / 3.0)
    phi6L = (-6848 / 63.0) * PI**2.0
    phi6 = (
        (
            11583231236531 / 4.69421568e9
            - (5 * eta * (3147553127 + 588 * eta * (-45633 + 102260 * eta)))
            / 3.048192e6
            - (6848 * EulerGamma) / 21.0
            - (640 * PI**2.0) / 3.0
            + (2255 * eta * PI**2.0) / 12.0
            - (13696 * jnp.log(2)) / 21.0
            - (6848 * jnp.log(PI)) / 63.0
        )
        * PI**2.0
        + (
            (
                5
                * (
                    227 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                    - 156 * (chi1 + chi2) * eta
                )
                * PI
            )
            / 3.0
        )
        * PI**2.0
        + (
            (
                5
                * (
                    20 * chi1L2L * eta * (11763 + 12488 * eta)
                    + 7
                    * chi2L2
                    * (
                        -15103 * (-1 + delta)
                        + 2 * (-21683 + 6580 * delta) * eta
                        - 9808 * eta2
                    )
                    - 7
                    * chi1L2
                    * (
                        -15103 * (1 + delta)
                        + 2 * (21683 + 6580 * delta) * eta
                        + 9808 * eta2
                    )
                )
            )
            / 4032.0
        )
        * PI**2.0
    )
    phi7 = (
        ((5 * (15419335 + 168 * (75703 - 29618 * eta) * eta) * PI) / 254016.0)
        * PI ** (7.0 / 3.0)
        + (
            (
                5
                * (
                    -5030016755 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                    + 4
                    * (2113331119 * (chi1 + chi2) + 675484362 * (chi1 - chi2) * delta)
                    * eta
                    - 1008
                    * (208433 * (chi1 + chi2) + 25011 * (chi1 - chi2) * delta)
                    * eta2
                    + 90514368 * (chi1 + chi2) * eta3
                )
            )
            / 6.096384e6
        )
        * PI ** (7.0 / 3.0)
        + (
            -5
            * (
                57 * chi1L2 * (1 + delta - 2 * eta)
                + 220 * chi1L2L * eta
                - 57 * chi2L2 * (-1 + delta + 2 * eta)
            )
            * PI
        )
        * PI ** (7.0 / 3.0)
        + (
            (
                14585 * (-(chi2L3 * (-1 + delta)) + chi1L3 * (1 + delta))
                - 5
                * (
                    chi2L3 * (8819 - 2985 * delta)
                    + 8439 * chi1 * chi2L2 * (-1 + delta)
                    - 8439 * chi1L2 * chi2 * (1 + delta)
                    + chi1L3 * (8819 + 2985 * delta)
                )
                * eta
                + 40 * (chi1 + chi2) * (17 * chi1L2 - 14 * chi1L2L + 17 * chi2L2) * eta2
            )
            / 48.0
        )
        * PI ** (7.0 / 3.0)
    )
    phi8 = (
        (
            -5
            * (
                1263141 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                - 2 * (794075 * (chi1 + chi2) + 178533 * (chi1 - chi2) * delta) * eta
                + 94344 * (chi1 + chi2) * eta2
            )
            * PI
            * (-1 + jnp.log(PI))
        )
        / 9072.0
    ) * PI ** (8.0 / 3.0)
    phi8L = (
        (
            -5.0
            * (
                1263141.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                - 2.0
                * (794075.0 * (chi1 + chi2) + 178533.0 * (chi1 - chi2) * delta)
                * eta
                + 94344.0 * (chi1 + chi2) * eta2
            )
            * PI
        )
        / 9072.0
    ) * PI ** (8.0 / 3.0)

    gpoints4 = jnp.array([0.0, 1.0 / 4.0, 3.0 / 4.0, 1.0])
    # Note that they do not use 4.1 from 2001.11412, they actually use
    # (Cos(i PI / 3) + 1)/2

    _, _, fMECO, _ = IMRPhenomX_utils.get_cutoff_fs(m1, m2, chi1, chi2)

    fPhaseInsMin = 0.0026
    fPhaseInsMax = 1.020 * fMECO

    deltax = fPhaseInsMax - fPhaseInsMin
    xmin = fPhaseInsMin

    CollocationPointsPhaseIns0 = gpoints4[0] * deltax + xmin
    CollocationPointsPhaseIns1 = gpoints4[1] * deltax + xmin
    CollocationPointsPhaseIns2 = gpoints4[2] * deltax + xmin
    CollocationPointsPhaseIns3 = gpoints4[3] * deltax + xmin

    CollocationValuesPhaseIns0 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[0, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[0, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[0, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseIns1 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[1, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[1, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[1, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseIns2 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[2, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[2, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[2, uneqspin_indx:], eta, S, chia)
    )

    # NOTE: This CollocationValuesPhaseIns3 disagrees slightly with the value in WF4py at non-zero spin
    CollocationValuesPhaseIns3 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[3, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[3, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[3, uneqspin_indx:], eta, S, chia)
    )

    # See line 1322 of https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__internals_8c_source.html
    CollocationValuesPhaseIns0 = CollocationValuesPhaseIns0 + CollocationValuesPhaseIns2
    CollocationValuesPhaseIns1 = CollocationValuesPhaseIns1 + CollocationValuesPhaseIns2
    CollocationValuesPhaseIns3 = CollocationValuesPhaseIns3 + CollocationValuesPhaseIns2

    A0 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseIns0.shape),
            CollocationPointsPhaseIns0 ** (1.0 / 3.0),
            CollocationPointsPhaseIns0 ** (2.0 / 3.0),
            CollocationPointsPhaseIns0,
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseIns1.shape),
            CollocationPointsPhaseIns1 ** (1.0 / 3.0),
            CollocationPointsPhaseIns1 ** (2.0 / 3.0),
            CollocationPointsPhaseIns1,
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseIns2.shape),
            CollocationPointsPhaseIns2 ** (1.0 / 3.0),
            CollocationPointsPhaseIns2 ** (2.0 / 3.0),
            CollocationPointsPhaseIns2,
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseIns3.shape),
            CollocationPointsPhaseIns3 ** (1.0 / 3.0),
            CollocationPointsPhaseIns3 ** (2.0 / 3.0),
            CollocationPointsPhaseIns3,
        ]
    )

    A = jnp.array([A0, A1, A2, A3])
    b = jnp.array(
        [
            CollocationValuesPhaseIns0,
            CollocationValuesPhaseIns1,
            CollocationValuesPhaseIns2,
            CollocationValuesPhaseIns3,
        ]
    ).T

    coeffscoloc = jnp.linalg.solve(A, b)
    a0 = coeffscoloc[0]
    a1 = coeffscoloc[1]
    a2 = coeffscoloc[2]
    a3 = coeffscoloc[3]

    sigma1 = (-5.0 / 3.0) * a0
    sigma2 = (-5.0 / 4.0) * a1
    sigma3 = (-5.0 / 5.0) * a2
    sigma4 = (-5.0 / 6.0) * a3

    phi_TF2 = (
        phi0
        + phi1 * (fM_s ** (1.0 / 3.0))
        + phi2 * (fM_s ** (2.0 / 3.0))
        + phi3 * fM_s
        + phi4 * (fM_s ** (4.0 / 3.0))
        + phi5 * (fM_s ** (5.0 / 3.0))
        + phi5L * (fM_s ** (5.0 / 3.0)) * jnp.log(fM_s)
        + phi6 * (fM_s**2.0)
        + phi6L * (fM_s**2.0) * jnp.log(fM_s)
        + phi7 * (fM_s ** (7.0 / 3.0))
        + phi8 * (fM_s ** (8.0 / 3.0))
        + phi8L * (fM_s ** (8.0 / 3.0)) * jnp.log(fM_s)
    )

    phi_Ins = phi_TF2 + (
        sigma1 * (fM_s ** (8.0 / 3.0))
        + sigma2 * (fM_s**3.0)
        + sigma3 * (fM_s ** (10.0 / 3.0))
        + sigma4 * (fM_s ** (11.0 / 3.0))
    )

    phiN = -(3.0 * PI ** (-5.0 / 3.0)) / 128.0
    return phi_Ins * phiN * (fM_s ** -(5.0 / 3.0))


def get_intermediate_raw_phase(
    fM_s: Array, theta: Array, coeffs: Array, dPhaseIN, dPhaseRD, cL
) -> Array:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    eta2 = eta * eta
    eta3 = eta2 * eta
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))

    # Spin variables
    chia = chi1 - chi2

    fRD, fdamp, fMECO, fISCO = IMRPhenomX_utils.get_cutoff_fs(m1, m2, chi1, chi2)

    gpoints5 = jnp.array(
        [
            0.0,
            1.0 / 2.0 - 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0 / 2.0,
            1.0 / 2 + 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0,
        ]
    )

    # Intermediate phase collocation points:
    # Default is to use 5 collocation points
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fINmatch = fMECO
    deltaf = (fIMmatch - fINmatch) * 0.03
    fPhaseMatchIN = fINmatch - 1.0 * deltaf
    fPhaseMatchIM = fIMmatch + 0.5 * deltaf

    deltax = fPhaseMatchIM - fPhaseMatchIN
    xmin = fPhaseMatchIN

    CollocationPointsPhaseInt0 = gpoints5[0] * deltax + xmin
    CollocationPointsPhaseInt1 = gpoints5[1] * deltax + xmin
    CollocationPointsPhaseInt2 = gpoints5[2] * deltax + xmin
    CollocationPointsPhaseInt3 = gpoints5[3] * deltax + xmin
    CollocationPointsPhaseInt4 = gpoints5[4] * deltax + xmin

    CollocationValuesPhaseInt0 = dPhaseIN
    CollocationValuesPhaseInt4 = dPhaseRD

    v2IMmRDv4 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[4, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[4, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[4, uneqspin_indx:], eta, S, chia)
    )

    v3IMmRDv4 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[5, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[5, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[5, uneqspin_indx:], eta, S, chia)
    )
    v2IM = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[6, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[6, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[6, uneqspin_indx:], eta, S, chia)
    )

    d43 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[7, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[7, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[7, uneqspin_indx:], eta, S, chia)
    )

    CollocationValuesPhaseRD3 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[11, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[11, uneqspin_indx:], eta, S, chia)
    )

    CollocationValuesPhaseInt1 = (
        0.75 * (v2IMmRDv4 + CollocationValuesPhaseRD3) + 0.25 * v2IM
    )
    CollocationValuesPhaseInt2 = v3IMmRDv4 + CollocationValuesPhaseRD3
    CollocationValuesPhaseInt3 = d43 + CollocationValuesPhaseInt2

    A0 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt0.shape),
            fRD / CollocationPointsPhaseInt0,
            (fRD / CollocationPointsPhaseInt0) * (fRD / CollocationPointsPhaseInt0),
            (fRD / CollocationPointsPhaseInt0) ** 3,
            (fRD / CollocationPointsPhaseInt0) ** 4,
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt1.shape),
            fRD / CollocationPointsPhaseInt1,
            (fRD / CollocationPointsPhaseInt1) * (fRD / CollocationPointsPhaseInt1),
            (fRD / CollocationPointsPhaseInt1) ** 3,
            (fRD / CollocationPointsPhaseInt1) ** 4,
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt2.shape),
            fRD / CollocationPointsPhaseInt2,
            (fRD / CollocationPointsPhaseInt2) * (fRD / CollocationPointsPhaseInt2),
            (fRD / CollocationPointsPhaseInt2) ** 3,
            (fRD / CollocationPointsPhaseInt2) ** 4,
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt3.shape),
            fRD / CollocationPointsPhaseInt3,
            (fRD / CollocationPointsPhaseInt3) * (fRD / CollocationPointsPhaseInt3),
            (fRD / CollocationPointsPhaseInt3) ** 3,
            (fRD / CollocationPointsPhaseInt3) ** 4,
        ]
    )
    A4 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt4.shape),
            fRD / CollocationPointsPhaseInt4,
            (fRD / CollocationPointsPhaseInt4) * (fRD / CollocationPointsPhaseInt4),
            (fRD / CollocationPointsPhaseInt4) ** 3,
            (fRD / CollocationPointsPhaseInt4) ** 4,
        ]
    )

    A = jnp.array([A0, A1, A2, A3, A4])
    b = jnp.array(
        [
            CollocationValuesPhaseInt0
            - (
                (4.0 * cL)
                / (
                    (4.0 * fdamp * fdamp)
                    + (CollocationPointsPhaseInt0 - fRD)
                    * (CollocationPointsPhaseInt0 - fRD)
                )
            ),
            CollocationValuesPhaseInt1
            - (
                (4.0 * cL)
                / (
                    (4.0 * fdamp * fdamp)
                    + (CollocationPointsPhaseInt1 - fRD)
                    * (CollocationPointsPhaseInt1 - fRD)
                )
            ),
            CollocationValuesPhaseInt2
            - (
                (4.0 * cL)
                / (
                    (4.0 * fdamp * fdamp)
                    + (CollocationPointsPhaseInt2 - fRD)
                    * (CollocationPointsPhaseInt2 - fRD)
                )
            ),
            CollocationValuesPhaseInt3
            - (
                (4.0 * cL)
                / (
                    (4.0 * fdamp * fdamp)
                    + (CollocationPointsPhaseInt3 - fRD)
                    * (CollocationPointsPhaseInt3 - fRD)
                )
            ),
            CollocationValuesPhaseInt4
            - (
                (4.0 * cL)
                / (
                    (4.0 * fdamp * fdamp)
                    + (CollocationPointsPhaseInt4 - fRD)
                    * (CollocationPointsPhaseInt4 - fRD)
                )
            ),
        ]
    ).T

    coeffscoloc = jnp.linalg.solve(A, b)

    b0 = coeffscoloc[0]
    b1 = coeffscoloc[1] * fRD
    b2 = coeffscoloc[2] * fRD**2
    b3 = coeffscoloc[3] * fRD**3
    b4 = coeffscoloc[4] * fRD**4
    # print("FRD", fRD)
    # print("frequencies:", (fM_s**-3.0))
    # print(b0 * fM_s)
    # print(
    #     "current test",
    #     +b1 * jnp.log(fM_s)
    #     - b2 * (fM_s**-1.0)
    #     - b3 * (fM_s**-2.0) / 2.0
    #     - (b4 * (fM_s**-3.0) / 3.0),
    # )
    # print(b2)
    # print(-b2 * (fM_s**-1.0))
    # print(-b3 * (fM_s**-2.0) / 2.0)
    # print(-(b4 * (fM_s**-3.0) / 3.0))
    # print(+(2.0 * cL * jnp.arctan((fM_s - fRD) / (2.0 * fdamp**2))))
    #  b0coloc*infreqs + b1coloc*np.log(infreqs) - b2coloc/infreqs - b3coloc/(infreqs*infreqs)/2. - (b4coloc/(infreqs*infreqs*infreqs)/3.) + (2. * cLcoloc * np.arctan((infreqs - fring) / (2. * fdamp)))/fdamp
    # print(
    #     "Intermediate phase",
    #     (
    #         b0 * fM_s
    #         + b1 * jnp.log(fM_s)
    #         - b2 * (fM_s**-1.0)
    #         - b3 * (fM_s**-2.0) / 2.0
    #         - (b4 * (fM_s**-3.0) / 3.0)
    #         + (2.0 * cL * jnp.arctan((fM_s - fRD) / (2.0 * fdamp**2)))
    #     ),
    # )

    return (
        b0 * fM_s
        + b1 * jnp.log(fM_s)
        - b2 * (fM_s**-1.0)
        - b3 * (fM_s**-2.0) / 2.0
        - (b4 * (fM_s**-3.0) / 3.0)
        + (2.0 * cL * jnp.arctan((fM_s - fRD) / (2.0 * fdamp**2)))
    )


def get_mergerringdown_raw_phase(
    fM_s: Array, theta: Array, coeffs: Array  # , f_RD, f_damp
) -> Array:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
    chia = chi1 - chi2

    fRD, fdamp, _, fISCO = IMRPhenomX_utils.get_cutoff_fs(m1, m2, chi1, chi2)
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fPhaseRDMin = fIMmatch
    fPhaseRDMax = fRD + 1.25 * fdamp
    dphase0 = 5.0 / (128.0 * (PI ** (5.0 / 3.0)))

    gpoints5 = jnp.array(
        [
            0.0,
            1.0 / 2.0 - 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0 / 2.0,
            1.0 / 2 + 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0,
        ]
    )

    # Ringdown phase collocation points:
    # Default is to use 5 pseudo-PN coefficients and hence 5 collocation points.
    deltax = fPhaseRDMax - fPhaseRDMin
    xmin = fPhaseRDMin

    CollocationPointsPhaseRD0 = gpoints5[0] * deltax + xmin
    CollocationPointsPhaseRD1 = gpoints5[1] * deltax + xmin
    CollocationPointsPhaseRD2 = gpoints5[2] * deltax + xmin
    CollocationPointsPhaseRD3 = fRD
    CollocationPointsPhaseRD4 = gpoints5[4] * deltax + xmin

    CollocationValuesPhaseRD0 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[8, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[8, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[8, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseRD1 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[9, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[9, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[9, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseRD2 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[10, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[10, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[10, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseRD3 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[11, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[11, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseRD4 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[12, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[12, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[12, uneqspin_indx:], eta, S, chia)
    )

    CollocationValuesPhaseRD4 = CollocationValuesPhaseRD4 + CollocationValuesPhaseRD3
    CollocationValuesPhaseRD2 = CollocationValuesPhaseRD2 + CollocationValuesPhaseRD3
    CollocationValuesPhaseRD1 = CollocationValuesPhaseRD1 + CollocationValuesPhaseRD3
    CollocationValuesPhaseRD0 = CollocationValuesPhaseRD0 + CollocationValuesPhaseRD1

    A0 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseRD0.shape),
            CollocationPointsPhaseRD0 ** (-1.0 / 3.0),
            CollocationPointsPhaseRD0 ** (-2),
            CollocationPointsPhaseRD0 ** (-4),
            -(dphase0)
            / (
                fdamp * fdamp
                + (CollocationPointsPhaseRD0 - fRD) * (CollocationPointsPhaseRD0 - fRD)
            ),
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseRD1.shape),
            CollocationPointsPhaseRD1 ** (-1.0 / 3.0),
            CollocationPointsPhaseRD1 ** (-2),
            CollocationPointsPhaseRD1 ** (-4),
            -(dphase0)
            / (
                fdamp * fdamp
                + (CollocationPointsPhaseRD1 - fRD) * (CollocationPointsPhaseRD1 - fRD)
            ),
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseRD2.shape),
            CollocationPointsPhaseRD2 ** (-1.0 / 3.0),
            CollocationPointsPhaseRD2 ** (-2),
            CollocationPointsPhaseRD2 ** (-4),
            -(dphase0)
            / (
                fdamp * fdamp
                + (CollocationPointsPhaseRD2 - fRD) * (CollocationPointsPhaseRD2 - fRD)
            ),
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseRD3.shape),
            CollocationPointsPhaseRD3 ** (-1.0 / 3.0),
            CollocationPointsPhaseRD3 ** (-2),
            CollocationPointsPhaseRD3 ** (-4),
            -(dphase0)
            / (
                fdamp * fdamp
                + (CollocationPointsPhaseRD3 - fRD) * (CollocationPointsPhaseRD3 - fRD)
            ),
        ]
    )
    A4 = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseRD4.shape),
            CollocationPointsPhaseRD4 ** (-1.0 / 3.0),
            CollocationPointsPhaseRD4 ** (-2),
            CollocationPointsPhaseRD4 ** (-4),
            -(dphase0)
            / (
                fdamp * fdamp
                + (CollocationPointsPhaseRD4 - fRD) * (CollocationPointsPhaseRD4 - fRD)
            ),
        ]
    )

    A = jnp.array([A0, A1, A2, A3, A4])
    b = jnp.array(
        [
            CollocationValuesPhaseRD0,
            CollocationValuesPhaseRD1,
            CollocationValuesPhaseRD2,
            CollocationValuesPhaseRD3,
            CollocationValuesPhaseRD4,
        ]
    ).T

    coeffscoloc = jnp.linalg.solve(A, b)
    c0 = coeffscoloc[0]
    c1 = coeffscoloc[1]
    c2 = coeffscoloc[2]
    c4 = coeffscoloc[3]
    cRD = coeffscoloc[4]

    cL = -(dphase0 * cRD)
    c4ov3 = c4 / 3.0
    cLovfda = cL / fdamp

    phiRD = (
        c0 * fM_s
        + 1.5 * c1 * (fM_s ** (2.0 / 3.0))
        - c2 * (fM_s**-1.0)
        - c4ov3 * (fM_s**-3.0)
        + (cLovfda * jnp.arctan((fM_s - fRD) / fdamp))
    )

    return phiRD, (cL, CollocationValuesPhaseRD0)


# @jax.jit
def Phase(f: Array, theta: Array, coeffs: Array) -> Array:
    """
    Computes the phase of the PhenomD waveform following 1508.07253.
    Sets time and phase of coealence to be zero.

    Returns:
    --------
        phase (array): Phase of the GW as a function of frequency
    """
    # First lets calculate some of the vairables that will be used below
    # Mass variables
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s
    fRD, fdamp, fMECO, fISCO = IMRPhenomX_utils.get_cutoff_fs(m1, m2, chi1, chi2)
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fINmatch = fMECO
    deltaf = (fIMmatch - fINmatch) * 0.03
    # fPhaseInsMin = 0.0026
    # fPhaseInsMax = 1.020 * fMECO
    # fPhaseRDMin = fIMmatch
    # fPhaseRDMax = fRD + 1.25 * fdamp
    f1_Ms = fINmatch - 1.0 * deltaf
    f2_Ms = fIMmatch + 0.5 * deltaf

    phi_Ins = get_inspiral_phase(fM_s, theta, coeffs)
    phi_MRD, (cL, CollocationValuesPhaseRD0) = get_mergerringdown_raw_phase(
        fM_s, theta, coeffs
    )

    # Get the matching points
    phi_Ins_match_f1, dphi_Ins_match_f1 = jax.value_and_grad(get_inspiral_phase)(
        f1_Ms, theta, coeffs
    )
    phi_MRD_match_f2, dphi_MRD_match_f2 = jax.value_and_grad(
        get_mergerringdown_raw_phase, has_aux=True
    )(f2_Ms, theta, coeffs)
    phi_MRD_match_f2, _ = get_mergerringdown_raw_phase(f2_Ms, theta, coeffs)

    # Now find the intermediate phase
    phi_Int_match_f1, dphi_Int_match_f1 = jax.value_and_grad(
        get_intermediate_raw_phase
    )(f1_Ms, theta, coeffs, dphi_Ins_match_f1, CollocationValuesPhaseRD0, cL)
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    phi_Int_func = (
        lambda fM_s_: get_intermediate_raw_phase(
            fM_s_, theta, coeffs, dphi_Ins_match_f1, CollocationValuesPhaseRD0, cL
        )
        + alpha1 * fM_s_
        + alpha0
    )

    phi_Int_match_f2, dphi_Int_match_f2 = jax.value_and_grad(phi_Int_func)(f2_Ms)

    beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2
    beta0 = phi_Int_match_f2 - phi_MRD_match_f2 - beta1 * f2_Ms

    phi_Int_corrected = phi_Int_func(fM_s)
    phi_MRD_corrected = phi_MRD + beta0 + beta1 * fM_s

    phase = (1 / eta) * (
        phi_Ins * jnp.heaviside(f1_Ms - fM_s, 0.5)
        + jnp.heaviside(fM_s - f1_Ms, 0.5)
        * phi_Int_corrected
        * jnp.heaviside(f2_Ms - fM_s, 0.5)
        + phi_MRD_corrected * jnp.heaviside(fM_s - f2_Ms, 0.5)
    )

    return phase


def get_Amp0(fM_s: Array, eta: float) -> Array:
    Amp0 = (
        (2.0 / 3.0 * eta) ** (1.0 / 2.0) * (fM_s) ** (-7.0 / 6.0) * PI ** (-1.0 / 6.0)
    )
    return Amp0


def get_inspiral_Amp(fM_s: Array, theta: Array) -> Array:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    eta2 = eta * eta
    eta3 = eta * eta2
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    # Spin variables
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2

    chi13 = chi12 * chi1
    chi23 = chi22 * chi2

    A0 = 1.0
    A2 = -323.0 / 224.0 + 451.0 * eta / 168.0
    A3 = chi1 * (27.0 * delta / 16.0 - 11.0 * eta / 12.0 + 27.0 / 16.0) + chi2 * (
        -27.0 * delta / 16.0 - 11.0 * eta / 12.0 + 27.0 / 16.0
    )
    A4 = (
        chi12 * (-81.0 * delta / 64.0 + 81.0 * eta / 32.0 - 81.0 / 64.0)
        + chi22 * (81.0 * delta / 64.0 + 81.0 * eta / 32.0 - 81.0 / 64.0)
        + (
            105271.0 * eta2 / 24192.0
            - 1975055.0 * eta / 338688
            - 27312085.0 / 8128512.0
        )
        - 47.0 * eta * chi1 * chi2 / 16.0
    )
    A5 = (
        chi13 * (delta * (3.0 / 16.0 - 3 * eta / 16.0) - 9.0 * eta / 16.0 + 3.0 / 16.0)
        + chi1
        * (
            delta * (287213.0 / 32256.0 - 2083.0 * eta / 8064.0)
            - 2227.0 * eta2 / 2016.0
            - 15569.0 * eta / 1344.0
            + 287213.0 / 32256.0
        )
        + chi23
        * (delta * (3.0 * eta / 16.0 - 3.0 / 16.0) - 9.0 * eta / 16.0 + 3.0 / 16.0)
    )
    (
        +chi2
        * (
            delta * (2083.0 * eta / 8064.0 - 287213.0 / 32256.0)
            - 2227.0 * eta2 / 2016.0
            - 15569.0 * eta / 1344.0
            + 287213.0 / 32256.0
        )
        - 85.0 * PI / 64.0
        + 85.0 * PI * eta / 16.0
    )
    A6 = (
        (
            chi1
            * (
                -17.0 * PI * delta / 12.0
                + (-133249.0 * eta2 / 8064.0 - 319321.0 * eta / 32256.0) * chi2
                + 5.0 * PI * eta / 3.0
                - 17.0 * PI / 12.0
            )
            + chi12
            * (
                delta * (-141359.0 * eta / 32256.0 - 49039.0 / 14336.0)
                + 163199.0 * eta2 / 16128.0
                + 158633.0 * eta / 64512.0
                - 49039.0 / 14336.0
            )
            + chi22
            * (
                delta * (141359.0 * eta / 32256.0 - 49039.0 / 14336.0)
                + 163199.0 * eta2 / 16128.0
                + 158633.0 * eta / 64512.0
                - 49039.0 / 14336.0
            )
        )
        + chi2 * (17.0 * PI * delta / 12.0 + 5 * PI * eta / 3.0 - 17 * PI / 12.0)
        - 177520268561.0 / 8583708672.0
        + (545384828789.0 / 5007163392.0 - 205.0 * PI**2.0 / 48.0) * eta
        - 3248849057.0 * eta2 / 178827264.0
        + 34473079.0 * eta3 / 6386688.0
    )

    # Here we need to compute the rhos
    # A7 = rho1
    # A8 = rho2
    # A9 = rho3

    Amp_Ins = (
        A0
        # A1 is missed since its zero
        + A2 * (fM_s ** (2.0 / 3.0))
        + A3 * fM_s
        + A4 * (fM_s ** (4.0 / 3.0))
        + A5 * (fM_s ** (5.0 / 3.0))
        + A6 * (fM_s**2.0)
        # # Now we add the coefficient terms
        # + A7 * (fM_s ** (7.0 / 3.0))
        # + A8 * (fM_s ** (8.0 / 3.0))
        # + A9 * (fM_s ** 3.0)
    )

    return Amp_Ins


def get_intermediate_Amp(
    fM_s: Array, theta: Array, coeffs: Array, f1, f3, f_RD, f_damp
) -> Array:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s

    return None


def get_mergerringdown_Amp(
    fM_s: Array, theta: Array, coeffs: Array, f_RD, f_damp
) -> Array:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    return None


# @jax.jit
def Amp(f: Array, theta: Array, D=1) -> Array:
    """
    Computes the amplitude of the PhenomD frequency domain waveform following 1508.07253.
    Note that this waveform also assumes that object one is the more massive.

    Returns:
    --------
      Amplitude (array):
    """

    # First lets calculate some of the vairables that will be used below
    # Mass variables
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # And now we can combine them by multiplying by a set of heaviside functions
    # Amp = (
    #     Amp_Ins * jnp.heaviside(f3 - f, 0.5)
    #     + jnp.heaviside(f - f3, 0.5) * Amp_Inter * jnp.heaviside(f4 - f, 0.5)
    #     + Amp_MR * jnp.heaviside(f - f4, 0.5)
    # )

    # Prefactor
    Amp0 = get_Amp0(f * M_s, eta) * (
        2.0 * jnp.sqrt(5.0 / (64.0 * PI))
    )  # This second factor is from lalsuite...

    # Need to add in an overall scaling of M_s^2 to make the units correct
    dist_s = (D * m_per_Mpc) / C
    # return Amp0 * Amp * (M_s ** 2.0) / dist_s
    return None


# @jax.jit
def _gen_IMRPhenomXAS(
    f: Array, theta_intrinsic: Array, theta_extrinsic: Array, coeffs: Array
):
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * gt
    # Lets call the amplitude and phase now
    Psi = Phase(f, theta_intrinsic, coeffs)
    A = Amp(f, theta_intrinsic, D=theta_extrinsic[0])
    h0 = A * jnp.exp(1j * -Psi)
    return h0


# @jax.jit
def gen_IMRPhenomXAS(f: Array, params: Array):
    """
    Generate PhenomXAS frequency domain waveform following 2001.11412.
    Note that this waveform also assumes that object one is the more massive.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    Returns:
    --------
      hp (array): Strain of the plus polarization
      hc (array): Strain of the cross polarization
    """
    # Lets make this easier by starting in Mchirp and eta space
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    theta_intrinsic = jnp.array([m1, m2, params[2], params[3]])
    theta_extrinsic = jnp.array([params[4], params[5], params[6]])

    # h0 = _gen_IMRPhenomXAS(f, theta_intrinsic, theta_extrinsic, coeffs)
    return None
