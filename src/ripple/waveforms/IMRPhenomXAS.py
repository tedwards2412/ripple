# from math import PI
import jax
import jax.numpy as jnp
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ripple.waveforms import IMRPhenomX_utils
from ..typing import Array

from ripple import Mc_eta_to_ms


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
    chis = chi1 + chi2
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

    eqspin_indx = 10
    uneqspin_indx = 38

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

    CollocationValuesPhaseIns3 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[3, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[3, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[3, uneqspin_indx:], eta, S, chia)
    )

    # See line 1322 of https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__internals_8c_source.html
    CollocationValuesPhaseIns0 = CollocationValuesPhaseIns0 + CollocationValuesPhaseIns2
    CollocationValuesPhaseIns1 = CollocationValuesPhaseIns1 + CollocationValuesPhaseIns2
    CollocationValuesPhaseIns3 = CollocationValuesPhaseIns3 + CollocationValuesPhaseIns2

    A0i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseIns0.shape),
            CollocationPointsPhaseIns0 ** (1.0 / 3.0),
            CollocationPointsPhaseIns0 ** (2.0 / 3.0),
            CollocationPointsPhaseIns0,
        ]
    )
    A1i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseIns1.shape),
            CollocationPointsPhaseIns1 ** (1.0 / 3.0),
            CollocationPointsPhaseIns1 ** (2.0 / 3.0),
            CollocationPointsPhaseIns1,
        ]
    )
    A2i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseIns2.shape),
            CollocationPointsPhaseIns2 ** (1.0 / 3.0),
            CollocationPointsPhaseIns2 ** (2.0 / 3.0),
            CollocationPointsPhaseIns2,
        ]
    )
    A3i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseIns3.shape),
            CollocationPointsPhaseIns3 ** (1.0 / 3.0),
            CollocationPointsPhaseIns3 ** (2.0 / 3.0),
            CollocationPointsPhaseIns3,
        ]
    )

    Acoloc = jnp.array([A0i, A1i, A2i, A3i]).T
    bcoloc = jnp.array(
        [
            CollocationValuesPhaseIns0,
            CollocationValuesPhaseIns1,
            CollocationValuesPhaseIns2,
            CollocationValuesPhaseIns3,
        ]
    ).T

    coeffscoloc = jnp.linalg.solve(Acoloc, bcoloc)
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

    # FIXME: The phase seems to be off by a factor proportional to f^(-1/3)
    # which corresponds to the phi4 term
    phiN = -(3.0 * jnp.pi ** (-5.0 / 3.0)) / 128.0 / eta
    return phi_Ins * phiN * (fM_s ** -(5.0 / 3.0))


def get_intermediate_raw_phase(
    fM_s: Array, theta: Array, coeffs: Array, dPhaseIN, dPhaseRD
) -> Array:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    eta2 = eta * eta
    eta3 = eta2 * eta
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    fRD, fdamp, fMECO, fISCO = IMRPhenomX_utils.get_cutoff_fs(m1, m2, chi1, chi2)

    gpoints5 = jnp.array(
        [
            1.0,
            1.0 / 2 + 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0 / 2,
            1.0 / 2 - 1.0 / (2 * jnp.sqrt(2.0)),
            0.0,
        ]
    )
    # Note that they do not use 4.1 from 2001.11412, they actually use
    # (Cos(i PI / 3) + 1)/2

    # Intermediate phase collocation points:
    # Default is to use 5 collocation points
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fINmatch = fMECO
    deltaf = (fIMmatch - fINmatch) * 0.03
    fPhaseMatchIN = fINmatch - 1.0 * deltaf
    fPhaseMatchIM = fIMmatch + 0.5 * deltaf
    # fPhaseInsMin = 0.0026
    # fPhaseInsMax = 1.020 * fMECO
    # fPhaseRDMin = fIMmatch
    # fPhaseRDMax = fRD + 1.25 * fdamp

    deltax = fPhaseMatchIM - fPhaseMatchIN
    xmin = fPhaseMatchIN

    CollocationPointsPhaseInt0 = gpoints5[0] * deltax + xmin
    CollocationPointsPhaseInt1 = gpoints5[1] * deltax + xmin
    CollocationPointsPhaseInt2 = gpoints5[2] * deltax + xmin
    CollocationPointsPhaseInt3 = gpoints5[3] * deltax + xmin
    CollocationPointsPhaseInt4 = gpoints5[4] * deltax + xmin

    CollocationValuesPhaseInt0 = dphaseIN
    CollocationValuesPhaseInt4 = dPhaseRD

    # v2IMmRDv4 = (
    #     (
    #         (
    #             eta
    #             * (
    #                 0.9951733419499662
    #                 + 101.21991715215253 * eta
    #                 + 632.4731389009143 * eta2
    #             )
    #         )
    #         / (
    #             0.00016803066316882238
    #             + 0.11412314719189287 * eta
    #             + 1.8413983770369362 * eta2
    #             + 1.0 * eta2 * eta
    #         )
    #     )
    #     + (
    #         (
    #             totchi
    #             * (
    #                 18.694178521101332
    #                 + 16.89845522539974 * totchi
    #                 + 4941.31613710257 * eta2 * totchi
    #                 + eta * (-697.6773920613674 - 147.53381808989846 * totchi2)
    #                 + 0.3612417066833153 * totchi2
    #                 + eta2
    #                 * eta
    #                 * (
    #                     3531.552143264721
    #                     - 14302.70838220423 * totchi
    #                     + 178.85850322465944 * totchi2
    #                 )
    #             )
    #         )
    #         / (2.965640445745779 - 2.7706595614504725 * totchi + 1.0 * totchi2)
    #     )
    #     + (
    #         dchi
    #         * delta
    #         * eta2
    #         * (356.74395864902294 + 1693.326644293169 * eta2 * totchi)
    #     )
    # )
    # v3IMmRDv4 = (
    #     (
    #         (
    #             eta
    #             * (
    #                 -5.126358906504587
    #                 - 227.46830225846668 * eta
    #                 + 688.3609087244353 * eta2
    #                 - 751.4184178636324 * eta2 * eta
    #             )
    #         )
    #         / (-0.004551938711031158 - 0.7811680872741462 * eta + 1.0 * eta2)
    #     )
    #     + (
    #         (
    #             totchi
    #             * (
    #                 0.1549280856660919
    #                 - 0.9539250460041732 * totchi
    #                 - 539.4071941841604 * eta2 * totchi
    #                 + eta * (73.79645135116367 - 8.13494176717772 * totchi2)
    #                 - 2.84311102369862 * totchi2
    #                 + eta2
    #                 * eta
    #                 * (
    #                     -936.3740515136005
    #                     + 1862.9097047992134 * totchi
    #                     + 224.77581754671272 * totchi2
    #                 )
    #             )
    #         )
    #         / (-1.5308507364054487 + 1.0 * totchi)
    #     )
    #     + (2993.3598520496153 * dchi * delta * eta2 * eta2 * eta2)
    # )
    # v2IM = (
    #     (
    #         (
    #             -82.54500000000004
    #             - 5.58197349185435e6 * eta
    #             - 3.5225742421184325e8 * eta2
    #             + 1.4667258334378073e9 * eta2 * eta
    #         )
    #         / (
    #             1.0
    #             + 66757.12830903867 * eta
    #             + 5.385164380400193e6 * eta2
    #             + 2.5176585751772933e6 * eta2 * eta
    #         )
    #     )
    #     + (
    #         (
    #             totchi
    #             * (
    #                 19.416719811164853
    #                 - 36.066611959079935 * totchi
    #                 - 0.8612656616290079 * totchi2
    #                 + eta2
    #                 * (
    #                     170.97203068800542
    #                     - 107.41099349364234 * totchi
    #                     - 647.8103976942541 * totchi2 * totchi
    #                 )
    #                 + 5.95010003393006 * totchi2 * totchi
    #                 + eta2
    #                 * eta
    #                 * (
    #                     -1365.1499998427248
    #                     + 1152.425940764218 * totchi
    #                     + 415.7134909564443 * totchi2
    #                     + 1897.5444343138167 * totchi2 * totchi
    #                     - 866.283566780576 * totchi2 * totchi2
    #                 )
    #                 + 4.984750041013893 * totchi2 * totchi2
    #                 + eta
    #                 * (
    #                     207.69898051583655
    #                     - 132.88417400679026 * totchi
    #                     - 17.671713040498304 * totchi2
    #                     + 29.071788188638315 * totchi2 * totchi
    #                     + 37.462217031512786 * totchi2 * totchi2
    #                 )
    #             )
    #         )
    #         / (-1.1492259468169692 + 1.0 * totchi)
    #     )
    #     + (
    #         dchi
    #         * delta
    #         * eta2
    #         * eta
    #         * (
    #             7343.130973149263
    #             - 20486.813161100774 * eta
    #             + 515.9898508588834 * totchi
    #         )
    #     )
    # )
    # RDv4 = 10.0  # CollocationValuesPhaseRD3
    # d43 = (
    #     (
    #         (
    #             0.4248820426833804
    #             - 906.746595921514 * eta
    #             - 282820.39946006844 * eta2
    #             - 967049.2793750163 * eta2 * eta
    #             + 670077.5414916876 * eta2 * eta2
    #         )
    #         / (1.0 + 1670.9440812294847 * eta + 19783.077247023448 * eta2)
    #     )
    #     + (
    #         (
    #             totchi
    #             * (
    #                 0.22814271667259703
    #                 + 1.1366593671801855 * totchi
    #                 + eta2
    #                 * eta
    #                 * (
    #                     3499.432393555856
    #                     - 877.8811492839261 * totchi
    #                     - 4974.189172654984 * totchi2
    #                 )
    #                 + eta * (12.840649528989287 - 61.17248283184154 * totchi2)
    #                 + 0.4818323187946999 * totchi2
    #                 + eta2
    #                 * (
    #                     -711.8532052499075
    #                     + 269.9234918621958 * totchi
    #                     + 941.6974723887743 * totchi2
    #                 )
    #                 + eta2
    #                 * eta2
    #                 * (
    #                     -4939.642457025497
    #                     - 227.7672020783411 * totchi
    #                     + 8745.201037897836 * totchi2
    #                 )
    #             )
    #         )
    #         / (-1.2442293719740283 + 1.0 * totchi)
    #     )
    #     + (dchi * delta * (-514.8494071830514 + 1493.3851099678195 * eta) * eta2 * eta)
    # )

    CollocationValuesPhaseInt1 = 0.75 * (v2IMmRDv4 + RDv4) + 0.25 * v2IM
    CollocationValuesPhaseInt2 = v3IMmRDv4 + RDv4
    CollocationValuesPhaseInt3 = d43 + CollocationValuesPhaseInt2

    A0i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt0.shape),
            fRD / CollocationPointsPhaseInt0,
            (fRD / CollocationPointsPhaseInt0) * (fRD / CollocationPointsPhaseInt0),
            (fRD / CollocationPointsPhaseInt0) ** 3,
            (fRD / CollocationPointsPhaseInt0) ** 4,
        ]
    )
    A1i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt1.shape),
            fRD / CollocationPointsPhaseInt1,
            (fRD / CollocationPointsPhaseInt1) * (fRD / CollocationPointsPhaseInt1),
            (fRD / CollocationPointsPhaseInt1) ** 3,
            (fRD / CollocationPointsPhaseInt1) ** 4,
        ]
    )
    A2i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt2.shape),
            fRD / CollocationPointsPhaseInt2,
            (fRD / CollocationPointsPhaseInt2) * (fRD / CollocationPointsPhaseInt2),
            (fRD / CollocationPointsPhaseInt2) ** 3,
            (fRD / CollocationPointsPhaseInt2) ** 4,
        ]
    )
    A3i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt3.shape),
            fRD / CollocationPointsPhaseInt3,
            (fRD / CollocationPointsPhaseInt3) * (fRD / CollocationPointsPhaseInt3),
            (fRD / CollocationPointsPhaseInt3) ** 3,
            (fRD / CollocationPointsPhaseInt3) ** 4,
        ]
    )
    A4i = jnp.array(
        [
            jnp.ones(CollocationPointsPhaseInt4.shape),
            fRD / CollocationPointsPhaseInt4,
            (fRD / CollocationPointsPhaseInt4) * (fRD / CollocationPointsPhaseInt4),
            (fRD / CollocationPointsPhaseInt4) ** 3,
            (fRD / CollocationPointsPhaseInt4) ** 4,
        ]
    )

    Acoloc = jnp.array([A0i, A1i, A2i, A3i, A4i]).T
    bcoloc = jnp.array(
        [
            CollocationValuesPhaseInt0
            - (
                (4.0 * cLcoloc)
                / (
                    (2.0 * fdamp) * (2.0 * fdamp)
                    + (CollocationPointsPhaseInt0 - fring)
                    * (CollocationPointsPhaseInt0 - fring)
                )
            ),
            CollocationValuesPhaseInt1
            - (
                (4.0 * cLcoloc)
                / (
                    (2.0 * fdamp) * (2.0 * fdamp)
                    + (CollocationPointsPhaseInt1 - fring)
                    * (CollocationPointsPhaseInt1 - fring)
                )
            ),
            CollocationValuesPhaseInt2
            - (
                (4.0 * cLcoloc)
                / (
                    (2.0 * fdamp) * (2.0 * fdamp)
                    + (CollocationPointsPhaseInt2 - fring)
                    * (CollocationPointsPhaseInt2 - fring)
                )
            ),
            CollocationValuesPhaseInt3
            - (
                (4.0 * cLcoloc)
                / (
                    (2.0 * fdamp) * (2.0 * fdamp)
                    + (CollocationPointsPhaseInt3 - fring)
                    * (CollocationPointsPhaseInt3 - fring)
                )
            ),
            CollocationValuesPhaseInt4
            - (
                (4.0 * cLcoloc)
                / (
                    (2.0 * fdamp) * (2.0 * fdamp)
                    + (CollocationPointsPhaseInt4 - fring)
                    * (CollocationPointsPhaseInt4 - fring)
                )
            ),
        ]
    ).T

    # coeffscoloc = np.linalg.solve(Acoloc, bcoloc)

    # b0coloc = coeffscoloc[:,0]
    # b1coloc = coeffscoloc[:,1] * fring
    # b2coloc = coeffscoloc[:,2] * fring * fring
    # b3coloc = coeffscoloc[:,3] * fring * fring * fring
    # b4coloc = coeffscoloc[:,4] * fring * fring * fring * fring

    return None


def get_mergerringdown_raw_phase(
    fM_s: Array, theta: Array, coeffs: Array, f_RD, f_damp
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

    #######################################################################
    #######################################################################
    fRD, fdamp, fMECO, fISCO = IMRPhenomX_utils.get_cutoff_fs(m1, m2, chi1, chi2)
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fPhaseRDMin = fIMmatch
    fPhaseRDMax = fRD + 1.25 * fdamp

    gpoints5 = jnp.array(
        [
            1.0,
            1.0 / 2 + 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0 / 2,
            1.0 / 2 - 1.0 / (2 * jnp.sqrt(2.0)),
            0.0,
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
    # (eta*(0.7207992174994245 - 1.237332073800276*eta + 6.086871214811216*eta2))/(0.006851189888541745 + 0.06099184229137391*eta - 0.15500218299268662*eta2 + 1.*eta3);

    eqspin_indx = 10
    uneqspin_indx = 38

    CollocationValuesPhaseRD0 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[9, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[9, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[9, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseRD1 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[10, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[10, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[10, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseRD2 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[11, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[11, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseRD3 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[12, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[12, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[12, uneqspin_indx:], eta, S, chia)
    )
    CollocationValuesPhaseRD4 = (
        IMRPhenomX_utils.nospin_CPvalue(coeffs[13, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CPvalue(coeffs[13, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CPvalue(coeffs[13, uneqspin_indx:], eta, S, chia)
    )

    # These are the nonspinning parts from the lalsuite code
    # For RD1 no spin
    # noSpin = (
    #     (
    #         -9.460253118496386 * eta  # 1
    #         + 9.429314399633007 * eta2  # 2
    #         + 64.69109972468395 * eta3
    #     )  # 3
    # ) / (
    #     -0.0670554310666559 - 0.09987544893382533 * eta + 1.0 * eta2  # 6  # 7  # 8
    #

    # For RD2 no spin
    # ((-8.506898502692536 * eta # 1
    # + 13.936621412517798*eta2) # 2
    # )/(
    #     -0.40919671232073945 # 6
    #     + 1.*eta); # 7

    # # For RD3 no spin
    # noSpin = (-85.86062966719405 # 0
    # - 4616.740713893726*eta # 1
    # - 4925.756920247186*eta2 # 2
    # + 7732.064464348168*eta3 # 3
    # + 12828.269960300782*eta4 # 4
    # - 39783.51698102803*eta5 # 5
    # )/(
    #     1.  # 6
    #     + 50.206318806624004*eta); # 7

    # # For RD4 no spin
    # noSpin = ((7.05731400277692 * eta # 1
    # + 22.455288821807095*eta2 # 2
    # + 119.43820622871043*eta3) # 3
    # )/(
    #     0.26026709603623255 # 6
    #     + 1.*eta); # 7

    # For RD0
    # eqSpin = (
    #     (
    #         0.06519048552628343  # 0
    #         - 25.25397971063995 * eta  # 5
    #         - 308.62513664956975 * eta4  # 19
    #         + 58.59408241189781 * eta2  # 10
    #         + 160.14971486043524 * eta3  # 15
    #     )

    #     * (
    #         -5.215945111216946 * eta  # 6
    #         + 153.95945758807616 * eta2 # 11
    #         - 693.0504179144295 * eta3  # 16
    #         + 835.1725103648205 * eta4  # 20
    #     )
    #     * S

    #     + (
    #         0.20035146870472367  # 2
    #         - 0.28745205203100666 * eta  # 7
    #         - 47.56042058800358 * eta4)  # 21
    #     * S2

    #     * (
    #         5.7756520242745735 * eta  # 8
    #         - 43.97332874253772 * eta2  # 13
    #         + 338.7263666984089 * eta4)  # 22
    #     * S3

    #     + (
    #         -0.2697933899920511  # 4
    #         + 4.917070939324979 * eta  # 9
    #         - 22.384949087140086 * eta4  # 23
    #         - 11.61488280763592 * eta2  # 14
    #     )
    #     * S4
    # ) / (1.0 # 24
    # - 0.6628745847248266 * S # 25
    # )

    # # For RD1 eq spin
    # eqSpin = S * (
    # (+ 0.04497628581617564*S3 # 3

    # + eta * (17.36495157980372) # 5

    # + eta2*(-191.00932194869588 # 10
    # - 62.997389062600035*S # 11
    # + 64.42947340363101*S2) # 12

    # + eta3*(930.3458437154668 # 15
    # + 808.457330742532*S) # 16

    # + eta4*(-774.3633787391745 # 19
    # - 2177.554979351284*S # 20
    # - 1031.846477275069*S2) # 21
    # )
    # )/(
    #     1. # 24
    #     - 0.7267610313751913*S); # 25

    # # For RD2 eq spin
    # eqSpin = S * (
    #     0.046849371468156265*S # 1

    #     + eta*(1.7280582989361533 # 5
    # + 18.41570325463385*S2 # 7
    # - 13.743271480938104*S3) # 8

    # + eta2*(73.8367329022058 # 10
    # - 95.57802408341716*S2 # 12
    #  + 215.78111099820157*S3) # 13

    #  + eta3*(-27.976989112929353 # 15
    #  + 6.404060932334562*S # 16
    #  - 633.1966645925428*S3 # 18
    #  + 109.04824706217418*S2) # 17
    #  )/(
    #     1. # 24
    #     - 0.6862449113932192*S); # 25

    # # For RD3 eq spin
    #     eqSpin = (S*

    #     (33.335857451144356 # 0
    #     - 36.49019206094966*S # 1
    #     - 3.835967351280833*S2 # 2
    #     + 2.302712009652155*S3  # 3
    #     + 1.6533417657003922*S4 # 4

    #    + eta*(-69.19412903018717 # 5
    #      + 26.580344399838758*S # 6
    #      - 15.399770764623746*S2 # 7
    #      + 31.231253209893488*S3 # 8
    #      + 97.69027029734173*S4) # 9

    #     + eta2*(93.64156367505917 # 10
    #     - 18.184492163348665*S  # 11
    #     + 423.48863373726243*S2 # 12
    #     - 104.36120236420928*S3 # 13
    #     - 719.8775484010988*S4) # 14

    #     + eta3*(1497.3545918387515 * S # 16
    #     - 101.72731770500685*S2) # 17

    #      + eta4*(1075.8686153198323 # 19
    #      - 3443.0233614187396*S # 20
    #      - 4253.974688619423*S2 #21
    #      - 608.2901586790335*S3 # 22
    #      + 5064.173605639933*S4)) # 23

    #     )/(
    #         -1.3705601055555852 # 24
    #         + 1.*S # 25
    #     );

    # # For RD 4 eq spin
    # eqSpin = S*(

    #     + eta*(-7.9407123129681425 # 5
    #     + 9.486783128047414*S) # 6

    #     eta2*(134.88158268621922  # 10
    #     - 56.05992404859163*S) # 11

    #     + eta3*(-316.26970506215554 # 15
    #     + 90.31815139272628*S) # 16

    #     )/(

    #         1. # 24
    #         - 0.7162058321905909*S); # 25

    # For RD0 uneq spin
    # uneqSpin = dchi*delta*eta
    # * -23.504907495268824 * eta; # 1

    # # For RD 1 uneq spin
    # dchi*delta*eta * (-36.66374091965371 * eta # 1
    # + 91.60477826830407*eta2) # 2

    # # For RD 2 uneq spin
    # uneqSpin = 641.8965762829259 * eta4 # 3
    # *dchi*delta*eta;

    # # For RD 3 uneq spin
    # uneqSpin = dchi*delta*eta*
    # (22.363215261437862 # 0
    # + 156.08206945239374*eta) # 1

    # # For RD 4 uneq spin
    # uneqSpin = 43.82713604567481 * eta2 # 2
    # *dchi*delta*eta;

    ###############################################################
    ###############################################################
    ###############################################################

    # CollocationValuesPhaseRD4 = CollocationValuesPhaseRD4 + CollocationValuesPhaseRD3
    # CollocationValuesPhaseRD2 = CollocationValuesPhaseRD2 + CollocationValuesPhaseRD3
    # CollocationValuesPhaseRD1 = CollocationValuesPhaseRD1 + CollocationValuesPhaseRD3
    # CollocationValuesPhaseRD0 = CollocationValuesPhaseRD0 + CollocationValuesPhaseRD1

    # A0i = np.array(
    #     [
    #         np.ones(CollocationPointsPhaseRD0.shape),
    #         CollocationPointsPhaseRD0 ** (-1.0 / 3.0),
    #         CollocationPointsPhaseRD0 ** (-2),
    #         CollocationPointsPhaseRD0 ** (-4),
    #         -(dphase0)
    #         / (
    #             fdamp * fdamp
    #             + (CollocationPointsPhaseRD0 - fring)
    #             * (CollocationPointsPhaseRD0 - fring)
    #         ),
    #     ]
    # )
    # A1i = np.array(
    #     [
    #         np.ones(CollocationPointsPhaseRD1.shape),
    #         CollocationPointsPhaseRD1 ** (-1.0 / 3.0),
    #         CollocationPointsPhaseRD1 ** (-2),
    #         CollocationPointsPhaseRD1 ** (-4),
    #         -(dphase0)
    #         / (
    #             fdamp * fdamp
    #             + (CollocationPointsPhaseRD1 - fring)
    #             * (CollocationPointsPhaseRD1 - fring)
    #         ),
    #     ]
    # )
    # A2i = np.array(
    #     [
    #         np.ones(CollocationPointsPhaseRD2.shape),
    #         CollocationPointsPhaseRD2 ** (-1.0 / 3.0),
    #         CollocationPointsPhaseRD2 ** (-2),
    #         CollocationPointsPhaseRD2 ** (-4),
    #         -(dphase0)
    #         / (
    #             fdamp * fdamp
    #             + (CollocationPointsPhaseRD2 - fring)
    #             * (CollocationPointsPhaseRD2 - fring)
    #         ),
    #     ]
    # )
    # A3i = np.array(
    #     [
    #         np.ones(CollocationPointsPhaseRD3.shape),
    #         CollocationPointsPhaseRD3 ** (-1.0 / 3.0),
    #         CollocationPointsPhaseRD3 ** (-2),
    #         CollocationPointsPhaseRD3 ** (-4),
    #         -(dphase0)
    #         / (
    #             fdamp * fdamp
    #             + (CollocationPointsPhaseRD3 - fring)
    #             * (CollocationPointsPhaseRD3 - fring)
    #         ),
    #     ]
    # )
    # A4i = np.array(
    #     [
    #         np.ones(CollocationPointsPhaseRD4.shape),
    #         CollocationPointsPhaseRD4 ** (-1.0 / 3.0),
    #         CollocationPointsPhaseRD4 ** (-2),
    #         CollocationPointsPhaseRD4 ** (-4),
    #         -(dphase0)
    #         / (
    #             fdamp * fdamp
    #             + (CollocationPointsPhaseRD4 - fring)
    #             * (CollocationPointsPhaseRD4 - fring)
    #         ),
    #     ]
    # )

    # Acoloc = np.array([A0i, A1i, A2i, A3i, A4i]).transpose(2, 0, 1)
    # bcoloc = np.array(
    #     [
    #         CollocationValuesPhaseRD0,
    #         CollocationValuesPhaseRD1,
    #         CollocationValuesPhaseRD2,
    #         CollocationValuesPhaseRD3,
    #         CollocationValuesPhaseRD4,
    #     ]
    # ).T

    # coeffscoloc = np.linalg.solve(Acoloc, bcoloc)
    # c0coloc = coeffscoloc[:, 0]
    # c1coloc = coeffscoloc[:, 1]
    # c2coloc = coeffscoloc[:, 2]
    # c4coloc = coeffscoloc[:, 3]
    # cRDcoloc = coeffscoloc[:, 4]
    # cLcoloc = -(dphase0 * cRDcoloc)
    # phaseRD = CollocationValuesPhaseRD0

    return None


@jax.jit
def Phase(f: Array, theta: Array) -> Array:
    """
    Computes the phase of the PhenomD waveform following 1508.07253.
    Sets time and phase of coealence to be zero.

    Returns:
    --------
        phase (array): Phase of the GW as a function of frequency
    """
    # First lets calculate some of the vairables that will be used below
    # Mass variables
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s

    phi_Ins = get_inspiral_phase(fM_s, theta, IMRPhenomX_utils.PhenomX_coeff_table)
    dphaseIN = jax.grad(get_inspiral_phase)(fM_s, theta, coeffs)
    phi_Int = get_intermediate_raw_phase(
        fM_s,
        theta,
        IMRPhenomX_utils.PhenomX_coeff_table,
        X,
    )
    # phi_Ins = phi_Ins * phiN * (fM_s ** -(5.0 / 3.0))

    # And now we can combine them by multiplying by a set of heaviside functions
    # phase = (
    #     phi_Ins * jnp.heaviside(f1 - f, 0.5)
    #     + jnp.heaviside(f - f1, 0.5) * phi_Inter * jnp.heaviside(f2 - f, 0.5)
    #     + phi_MR * jnp.heaviside(f - f2, 0.5)
    # )

    return None


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


@jax.jit
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


@jax.jit
def _gen_IMRPhenomXAS(
    f: Array, theta_intrinsic: Array, theta_extrinsic: Array, coeffs: Array
):
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * gt
    # Lets call the amplitude and phase now
    Psi = Phase(f, theta_intrinsic)
    A = Amp(f, theta_intrinsic, D=theta_extrinsic[0])
    h0 = A * jnp.exp(1j * -Psi)
    return h0


@jax.jit
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
