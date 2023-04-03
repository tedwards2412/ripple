# from math import PI
import jax
import jax.numpy as jnp
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ripple.waveforms import IMRPhenomX_utils
from ..typing import Array

from ripple import Mc_eta_to_ms

eqspin_indx = 10
uneqspin_indx = 39

# Format Choices:
# - All frequencies in Hz -> labelled as f, otherwise fM_s
# - Update variable names of collocation points and values


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

    _, _, fMs_MECO, _ = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)

    fMs_PhaseInsMin = 0.0026
    fMs_PhaseInsMax = 1.020 * fMs_MECO

    deltax = fMs_PhaseInsMax - fMs_PhaseInsMin
    xmin = fMs_PhaseInsMin

    CP_phase_Ins0 = gpoints4[0] * deltax + xmin
    CP_phase_Ins1 = gpoints4[1] * deltax + xmin
    CP_phase_Ins2 = gpoints4[2] * deltax + xmin
    CP_phase_Ins3 = gpoints4[3] * deltax + xmin

    CV_phase_Ins0 = (
        IMRPhenomX_utils.nospin_CV(coeffs[0, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[0, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[0, uneqspin_indx:], eta, S, chia)
    )
    CV_phase_Ins1 = (
        IMRPhenomX_utils.nospin_CV(coeffs[1, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[1, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[1, uneqspin_indx:], eta, S, chia)
    )
    CV_phase_Ins2 = (
        IMRPhenomX_utils.nospin_CV(coeffs[2, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[2, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[2, uneqspin_indx:], eta, S, chia)
    )

    # NOTE: This CV_phase_Ins3 disagrees slightly with the value in WF4py at non-zero spin
    CV_phase_Ins3 = (
        IMRPhenomX_utils.nospin_CV(coeffs[3, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[3, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[3, uneqspin_indx:], eta, S, chia)
    )

    # See line 1322 of https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__internals_8c_source.html
    CV_phase_Ins0 = CV_phase_Ins0 + CV_phase_Ins2
    CV_phase_Ins1 = CV_phase_Ins1 + CV_phase_Ins2
    CV_phase_Ins3 = CV_phase_Ins3 + CV_phase_Ins2

    A0 = jnp.array(
        [
            jnp.ones(CP_phase_Ins0.shape),
            CP_phase_Ins0 ** (1.0 / 3.0),
            CP_phase_Ins0 ** (2.0 / 3.0),
            CP_phase_Ins0,
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CP_phase_Ins1.shape),
            CP_phase_Ins1 ** (1.0 / 3.0),
            CP_phase_Ins1 ** (2.0 / 3.0),
            CP_phase_Ins1,
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CP_phase_Ins2.shape),
            CP_phase_Ins2 ** (1.0 / 3.0),
            CP_phase_Ins2 ** (2.0 / 3.0),
            CP_phase_Ins2,
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CP_phase_Ins3.shape),
            CP_phase_Ins3 ** (1.0 / 3.0),
            CP_phase_Ins3 ** (2.0 / 3.0),
            CP_phase_Ins3,
        ]
    )

    A = jnp.array([A0, A1, A2, A3])
    b = jnp.array(
        [
            CV_phase_Ins0,
            CV_phase_Ins1,
            CV_phase_Ins2,
            CV_phase_Ins3,
        ]
    ).T

    coeffs_Ins = jnp.linalg.solve(A, b)

    sigma1 = (-5.0 / 3.0) * coeffs_Ins[0]
    sigma2 = (-5.0 / 4.0) * coeffs_Ins[1]
    sigma3 = (-5.0 / 5.0) * coeffs_Ins[2]
    sigma4 = (-5.0 / 6.0) * coeffs_Ins[3]

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
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)

    # Spin variables
    chia = chi1 - chi2

    fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1, m2, chi1, chi2
    )

    gpoints5 = jnp.array(
        [
            0.0,
            1.0 / 2.0 - 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0 / 2.0,
            1.0 / 2 + 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0,
        ]
    )

    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    fMs_PhaseMatchIN = fMs_INmatch - 1.0 * deltafMs
    fPhaseMatchIM = fMs_IMmatch + 0.5 * deltafMs

    deltax = fPhaseMatchIM - fMs_PhaseMatchIN
    xmin = fMs_PhaseMatchIN

    CP_phase_Int0 = gpoints5[0] * deltax + xmin
    CP_phase_Int1 = gpoints5[1] * deltax + xmin
    CP_phase_Int2 = gpoints5[2] * deltax + xmin
    CP_phase_Int3 = gpoints5[3] * deltax + xmin
    CP_phase_Int4 = gpoints5[4] * deltax + xmin

    CV_phase_Int0 = dPhaseIN
    CV_phase_Int4 = dPhaseRD

    # NOTE: This is different to WF4py and driving the difference in CV_phase_Int1
    v2IMmRDv4 = (
        IMRPhenomX_utils.nospin_CV(coeffs[4, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[4, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[4, uneqspin_indx:], eta, StotR, chia)
    )

    v3IMmRDv4 = (
        IMRPhenomX_utils.nospin_CV(coeffs[5, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[5, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[5, uneqspin_indx:], eta, StotR, chia)
    )
    v2IM = (
        IMRPhenomX_utils.nospin_CV(coeffs[6, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[6, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[6, uneqspin_indx:], eta, StotR, chia)
    )

    # NOTE: This is different to WF4py and driving the difference in CV_phase_Int3
    d43 = (
        IMRPhenomX_utils.nospin_CV(coeffs[7, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[7, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[7, uneqspin_indx:], eta, StotR, chia)
    )

    CV_phase_RD3 = (
        IMRPhenomX_utils.nospin_CV(coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[11, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[11, uneqspin_indx:], eta, StotR, chia)
    )

    CV_phase_Int1 = 0.75 * (v2IMmRDv4 + CV_phase_RD3) + 0.25 * v2IM
    CV_phase_Int2 = v3IMmRDv4 + CV_phase_RD3
    CV_phase_Int3 = d43 + CV_phase_Int2

    A0 = jnp.array(
        [
            jnp.ones(CP_phase_Int0.shape),
            fMs_RD / CP_phase_Int0,
            (fMs_RD / CP_phase_Int0) * (fMs_RD / CP_phase_Int0),
            (fMs_RD / CP_phase_Int0) ** 3,
            (fMs_RD / CP_phase_Int0) ** 4,
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CP_phase_Int1.shape),
            fMs_RD / CP_phase_Int1,
            (fMs_RD / CP_phase_Int1) * (fMs_RD / CP_phase_Int1),
            (fMs_RD / CP_phase_Int1) ** 3,
            (fMs_RD / CP_phase_Int1) ** 4,
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CP_phase_Int2.shape),
            fMs_RD / CP_phase_Int2,
            (fMs_RD / CP_phase_Int2) * (fMs_RD / CP_phase_Int2),
            (fMs_RD / CP_phase_Int2) ** 3,
            (fMs_RD / CP_phase_Int2) ** 4,
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CP_phase_Int3.shape),
            fMs_RD / CP_phase_Int3,
            (fMs_RD / CP_phase_Int3) * (fMs_RD / CP_phase_Int3),
            (fMs_RD / CP_phase_Int3) ** 3,
            (fMs_RD / CP_phase_Int3) ** 4,
        ]
    )
    A4 = jnp.array(
        [
            jnp.ones(CP_phase_Int4.shape),
            fMs_RD / CP_phase_Int4,
            (fMs_RD / CP_phase_Int4) * (fMs_RD / CP_phase_Int4),
            (fMs_RD / CP_phase_Int4) ** 3,
            (fMs_RD / CP_phase_Int4) ** 4,
        ]
    )

    A = jnp.array([A0, A1, A2, A3, A4])
    b = jnp.array(
        [
            CV_phase_Int0
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int0 - fMs_RD) * (CP_phase_Int0 - fMs_RD)
                )
            ),
            CV_phase_Int1
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int1 - fMs_RD) * (CP_phase_Int1 - fMs_RD)
                )
            ),
            CV_phase_Int2
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int2 - fMs_RD) * (CP_phase_Int2 - fMs_RD)
                )
            ),
            CV_phase_Int3
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int3 - fMs_RD) * (CP_phase_Int3 - fMs_RD)
                )
            ),
            CV_phase_Int4
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int4 - fMs_RD) * (CP_phase_Int4 - fMs_RD)
                )
            ),
        ]
    ).T

    coeffs_Int = jnp.linalg.solve(A, b)

    b0 = coeffs_Int[0]
    b1 = coeffs_Int[1] * fMs_RD
    b2 = coeffs_Int[2] * fMs_RD**2
    b3 = coeffs_Int[3] * fMs_RD**3
    b4 = coeffs_Int[4] * fMs_RD**4

    return (
        b0 * fM_s
        + b1 * jnp.log(fM_s)
        - b2 * (fM_s**-1.0)
        - b3 * (fM_s**-2.0) / 2.0
        - (b4 * (fM_s**-3.0) / 3.0)
        + (2.0 * cL * jnp.arctan(((fM_s - fMs_RD)) / (2.0 * fMs_damp))) / fMs_damp
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
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)

    fMs_RD, fMs_damp, _, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_PhaseRDMin = fMs_IMmatch
    fMs_PhaseRDMax = fMs_RD + 1.25 * fMs_damp
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
    deltax = fMs_PhaseRDMax - fMs_PhaseRDMin
    xmin = fMs_PhaseRDMin

    CP_phase_RD0 = gpoints5[0] * deltax + xmin
    CP_phase_RD1 = gpoints5[1] * deltax + xmin
    CP_phase_RD2 = gpoints5[2] * deltax + xmin
    CP_phase_RD3 = fMs_RD
    CP_phase_RD4 = gpoints5[4] * deltax + xmin

    CV_phase_RD0 = (
        IMRPhenomX_utils.nospin_CV(coeffs[8, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[8, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[8, uneqspin_indx:], eta, StotR, chia)
    )
    CV_phase_RD1 = (
        IMRPhenomX_utils.nospin_CV(coeffs[9, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[9, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[9, uneqspin_indx:], eta, StotR, chia)
    )
    CV_phase_RD2 = (
        IMRPhenomX_utils.nospin_CV(coeffs[10, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[10, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[10, uneqspin_indx:], eta, StotR, chia)
    )
    CV_phase_RD3 = (
        IMRPhenomX_utils.nospin_CV(coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[11, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[11, uneqspin_indx:], eta, StotR, chia)
    )
    CV_phase_RD4 = (
        IMRPhenomX_utils.nospin_CV(coeffs[12, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(coeffs[12, eqspin_indx:uneqspin_indx], eta, StotR)
        + IMRPhenomX_utils.Uneqspin_CV(coeffs[12, uneqspin_indx:], eta, StotR, chia)
    )

    CV_phase_RD4 = CV_phase_RD4 + CV_phase_RD3
    CV_phase_RD2 = CV_phase_RD2 + CV_phase_RD3
    CV_phase_RD1 = CV_phase_RD1 + CV_phase_RD3
    CV_phase_RD0 = CV_phase_RD0 + CV_phase_RD1

    A0 = jnp.array(
        [
            jnp.ones(CP_phase_RD0.shape),
            CP_phase_RD0 ** (-1.0 / 3.0),
            CP_phase_RD0 ** (-2),
            CP_phase_RD0 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD0 - fMs_RD) * (CP_phase_RD0 - fMs_RD)),
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CP_phase_RD1.shape),
            CP_phase_RD1 ** (-1.0 / 3.0),
            CP_phase_RD1 ** (-2),
            CP_phase_RD1 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD1 - fMs_RD) * (CP_phase_RD1 - fMs_RD)),
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CP_phase_RD2.shape),
            CP_phase_RD2 ** (-1.0 / 3.0),
            CP_phase_RD2 ** (-2),
            CP_phase_RD2 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD2 - fMs_RD) * (CP_phase_RD2 - fMs_RD)),
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CP_phase_RD3.shape),
            CP_phase_RD3 ** (-1.0 / 3.0),
            CP_phase_RD3 ** (-2),
            CP_phase_RD3 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD3 - fMs_RD) * (CP_phase_RD3 - fMs_RD)),
        ]
    )
    A4 = jnp.array(
        [
            jnp.ones(CP_phase_RD4.shape),
            CP_phase_RD4 ** (-1.0 / 3.0),
            CP_phase_RD4 ** (-2),
            CP_phase_RD4 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD4 - fMs_RD) * (CP_phase_RD4 - fMs_RD)),
        ]
    )

    A = jnp.array([A0, A1, A2, A3, A4])
    b = jnp.array(
        [
            CV_phase_RD0,
            CV_phase_RD1,
            CV_phase_RD2,
            CV_phase_RD3,
            CV_phase_RD4,
        ]
    ).T

    coeffs_RD = jnp.linalg.solve(A, b)
    c0, c1, c2, c4, cRD = coeffs_RD
    cL = -(dphase0 * cRD)
    c4ov3 = c4 / 3.0
    cLovfda = cL / fMs_damp

    phiRD = (
        c0 * fM_s
        + 1.5 * c1 * (fM_s ** (2.0 / 3.0))
        - c2 * (fM_s**-1.0)
        - c4ov3 * (fM_s**-3.0)
        + (cLovfda * jnp.arctan((fM_s - fMs_RD) / fMs_damp))
    )

    return phiRD, (cL, CV_phase_RD0)


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
    fMs_RD, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    f1_Ms = fMs_INmatch - 1.0 * deltafMs
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs

    phi_Ins = get_inspiral_phase(fM_s, theta, coeffs)
    phi_MRD, (cL, CV_phase_RD0) = get_mergerringdown_raw_phase(fM_s, theta, coeffs)

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
    )(f1_Ms, theta, coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL)
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    phi_Int_func = (
        lambda fM_s_: get_intermediate_raw_phase(
            fM_s_, theta, coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL
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
        + phi_MRD_corrected
        * jnp.heaviside(fM_s - f2_Ms, 0.5)
        * jnp.heaviside(IMRPhenomX_utils.fM_cut - fM_s, 0.5)
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


def Amp(f: Array, theta: Array, D=1.0) -> Array:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    eta2 = eta * eta
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)

    # Spin variables
    chia = chi1 - chi2

    fM_s = f * M_s
    fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1, m2, chi1, chi2
    )
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M_s**2 / ((D * m_per_Mpc) / C)
    ampNorm = jnp.sqrt(2.0 * eta / 3.0) * (PI ** (-1.0 / 6.0))

    gamma2 = (
        (
            (0.8312293675316895 + 7.480371544268765 * eta - 18.256121237800397 * eta2)
            / (1.0 + 10.915453595496611 * eta - 30.578409433912874 * eta2)
        )
        + (
            (
                StotR
                * (
                    0.5869408584532747
                    + eta * (-0.1467158405070222 - 2.8489481072076472 * StotR)
                    + 0.031852563636196894 * StotR
                    + eta2 * (0.25295441250444334 + 4.6849496672664594 * StotR)
                )
            )
            / (3.8775263105069953 - 3.41755361841226 * StotR + 1.0 * StotR**2)
        )
        + (-0.00548054788508203 * chia * delta * eta)
    )
    gamma3 = (
        (
            (
                1.3666000000000007
                - 4.091333144596439 * eta
                + 2.109081209912545 * eta2
                - 4.222259944408823 * eta2 * eta
            )
            / (1.0 - 2.7440263888207594 * eta)
        )
        + (
            (
                0.07179105336478316
                + eta2 * (2.331724812782498 - 0.6330998412809531 * StotR)
                + eta * (-0.8752427297525086 + 0.4168560229353532 * StotR)
                - 0.05633734476062242 * StotR
            )
            * StotR
        )
        + (0.0 * delta * chia)
    )
    fAmpRDMin = jnp.where(
        gamma2 <= 1.0,
        jnp.fabs(
            fMs_RD
            + fMs_damp * gamma3 * (jnp.sqrt(1.0 - gamma2 * gamma2) - 1.0) / gamma2
        ),
        jnp.fabs(fMs_RD + fMs_damp * (-1.0) * gamma3 / gamma2),
    )
    v1RD = (
        (
            (0.03689164742964719 + 25.417967754401182 * eta + 162.52904393600332 * eta2)
            / (1.0 + 61.19874463331437 * eta - 29.628854485544874 * eta2)
        )
        + (
            (
                StotR
                * (
                    -0.14352506969368556
                    + 0.026356911108320547 * StotR
                    + 0.19967405175523437 * StotR**2
                    - 0.05292913111731128 * StotR**2 * StotR
                    + eta2
                    * eta
                    * (
                        -48.31945248941757
                        - 3.751501972663298 * StotR
                        + 81.9290740950083 * StotR**2
                        + 30.491948143930266 * StotR**2 * StotR
                        - 132.77982622925845 * StotR**2 * StotR**2
                    )
                    + eta
                    * (
                        -4.805034453745424
                        + 1.11147906765112 * StotR
                        + 6.176053843938542 * StotR**2
                        - 0.2874540719094058 * StotR**2 * StotR
                        - 8.990840289951514 * StotR**2 * StotR**2
                    )
                    - 0.18147275151697131 * StotR**2 * StotR**2
                    + eta2
                    * (
                        27.675454081988036
                        - 2.398327419614959 * StotR
                        - 47.99096500250743 * StotR**2
                        - 5.104257870393138 * StotR**2 * StotR
                        + 72.08174136362386 * StotR**2 * StotR**2
                    )
                )
            )
            / (-1.4160870461211452 + 1.0 * StotR)
        )
        + (-0.04426571511345366 * chia * delta * eta2)
    )
    F1 = fAmpRDMin

    gamma1 = (
        (v1RD / (fMs_damp * gamma3))
        * (
            F1 * F1
            - 2.0 * F1 * fMs_RD
            + fMs_RD * fMs_RD
            + fMs_damp * fMs_damp * gamma3 * gamma3
        )
        * jnp.exp(((F1 - fMs_RD) * gamma2) / (fMs_damp * gamma3))
    )
    gammaR = gamma2 / (fMs_damp * gamma3)
    gammaD2 = (gamma3 * fMs_damp) * (gamma3 * fMs_damp)
    gammaD13 = fMs_damp * gamma1 * gamma3
    fAmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fAmpMatchIN = fAmpInsMax

    # TaylorF2 PN Amplitude Coefficients
    pnInitial = 1.0
    pnOneThird = 0.0
    pnTwoThirds = ((-969 + 1804 * eta) / 672.0) * (PI ** (2.0 / 3.0))
    pnThreeThirds = (
        (
            81 * (chi1 + chi2)
            + 81 * chi1 * delta
            - 81 * chi2 * delta
            - 44 * (chi1 + chi2) * eta
        )
        / 48.0
    ) * PI
    pnFourThirds = (
        (
            -27312085
            - 10287648 * chi1**2 * (1 + delta)
            + 24
            * (
                428652 * chi2**2 * (-1 + delta)
                + (
                    -1975055
                    + 10584 * (81 * chi1**2 - 94 * chi1 * chi2 + 81 * chi2**2)
                )
                * eta
                + 1473794 * eta2
            )
        )
        / 8.128512e6
    ) * (PI ** (4.0 / 3.0))
    pnFiveThirds = (
        (
            -6048 * chi1**2 * chi1 * (-1 - delta + (3 + delta) * eta)
            + chi2
            * (
                -((287213 + 6048 * chi2**2) * (-1 + delta))
                + 4 * (-93414 + 1512 * chi2**2 * (-3 + delta) + 2083 * delta) * eta
                - 35632 * eta2
            )
            + chi1
            * (287213 * (1 + delta) - 4 * eta * (93414 + 2083 * delta + 8908 * eta))
            + 42840 * (-1 + 4 * eta) * PI
        )
        / 32256.0
    ) * (PI ** (5.0 / 3.0))
    pnSixThirds = (
        (
            (
                -1242641879927
                + 12.0
                * (
                    28.0
                    * (
                        -3248849057.0
                        + 11088
                        * (
                            163199 * chi1**2
                            - 266498 * chi1 * chi2
                            + 163199 * chi2**2
                        )
                    )
                    * eta2
                    + 27026893936 * eta2 * eta
                    - 116424
                    * (
                        147117
                        * (-(chi2**2 * (-1.0 + delta)) + chi1**2 * (1.0 + delta))
                        + 60928 * (chi1 + chi2 + chi1 * delta - chi2 * delta) * PI
                    )
                    + eta
                    * (
                        545384828789.0
                        - 77616
                        * (
                            638642 * chi1 * chi2
                            + chi1**2 * (-158633 + 282718 * delta)
                            - chi2**2 * (158633.0 + 282718.0 * delta)
                            - 107520.0 * (chi1 + chi2) * PI
                            + 275520 * PI * PI
                        )
                    )
                )
            )
            / 6.0085960704e10
        )
        * PI
        * PI
    )

    # Now the collocation points for the inspiral
    CollocationValuesAmpIns1 = (
        (
            (
                -0.015178276424448592
                - 0.06098548699809163 * eta
                + 0.4845148547154606 * eta2
            )
            / (1.0 + 0.09799277215675059 * eta)
        )
        + (
            (
                (0.02300153747158323 + 0.10495263104245876 * eta2) * S
                + (0.04834642258922544 - 0.14189350657140673 * eta) * eta * S**2 * S
                + (0.01761591799745109 - 0.14404522791467844 * eta2) * S**2
            )
            / (1.0 - 0.7340448493183307 * S)
        )
        + (
            chia
            * delta
            * eta2
            * eta2
            * (0.0018724905795891192 + 34.90874132485147 * eta)
        )
    )
    CollocationValuesAmpIns2 = (
        (
            (-0.058572000924124644 - 1.1970535595488723 * eta + 8.4630293045015 * eta2)
            / (1.0 + 15.430818840453686 * eta)
        )
        + (
            (
                (
                    -0.08746408292050666
                    + eta * (-0.20646621646484237 - 0.21291764491897636 * S)
                    + eta2 * (0.788717372588848 + 0.8282888482429105 * S)
                    - 0.018924013869130434 * S
                )
                * S
            )
            / (-1.332123330797879 + 1.0 * S)
        )
        + (
            chia
            * delta
            * eta2
            * eta2
            * (0.004389995099201855 + 105.84553997647659 * eta)
        )
    )
    CollocationValuesAmpIns3 = (
        (
            (
                -0.16212854591357853
                + 1.617404703616985 * eta
                - 3.186012733446088 * eta2
                + 5.629598195000046 * eta2 * eta
            )
            / (1.0 + 0.04507019231274476 * eta)
        )
        + (
            (
                S
                * (
                    1.0055835408962206
                    + eta2 * (18.353433894421833 - 18.80590889704093 * S)
                    - 0.31443470118113853 * S
                    + eta * (-4.127597118865669 + 5.215501942120774 * S)
                    + eta2 * eta * (-41.0378120175805 + 19.099315016873643 * S)
                )
            )
            / (5.852706459485663 - 5.717874483424523 * S + 1.0 * S**2)
        )
        + (
            chia
            * delta
            * eta2
            * eta2
            * (0.05575955418803233 + 208.92352600701068 * eta)
        )
    )

    CollocationPointsAmpIns1 = 0.50 * fAmpMatchIN
    CollocationPointsAmpIns2 = 0.75 * fAmpMatchIN
    CollocationPointsAmpIns3 = 1.00 * fAmpMatchIN

    V1 = CollocationValuesAmpIns1
    V2 = CollocationValuesAmpIns2
    V3 = CollocationValuesAmpIns3

    F1 = CollocationPointsAmpIns1
    F2 = CollocationPointsAmpIns2
    F3 = CollocationPointsAmpIns3

    rho1 = (
        -((F2 ** (8.0 / 3.0)) * (F3 * F3 * F3) * V1)
        + F2 * F2 * F2 * (F3 ** (8.0 / 3.0)) * V1
        + (F1 ** (8.0 / 3.0)) * (F3 * F3 * F3) * V2
        - F1 * F1 * F1 * (F3 ** (8.0 / 3.0)) * V2
        - (F1 ** (8.0 / 3.0)) * (F2 * F2 * F2) * V3
        + F1 * F1 * F1 * (F2 ** (8.0 / 3.0)) * V3
    ) / (
        (F1 ** (7.0 / 3.0))
        * (jnp.cbrt(F1) - jnp.cbrt(F2))
        * (F2 ** (7.0 / 3.0))
        * (jnp.cbrt(F1) - jnp.cbrt(F3))
        * (jnp.cbrt(F2) - jnp.cbrt(F3))
        * (F3 ** (7.0 / 3.0))
    )
    rho2 = (
        (F2 ** (7.0 / 3.0)) * (F3 * F3 * F3) * V1
        - F2 * F2 * F2 * (F3 ** (7.0 / 3.0)) * V1
        - (F1 ** (7.0 / 3.0)) * (F3 * F3 * F3) * V2
        + F1 * F1 * F1 * (F3 ** (7.0 / 3.0)) * V2
        + (F1 ** (7.0 / 3.0)) * (F2 * F2 * F2) * V3
        - F1 * F1 * F1 * (F2 ** (7.0 / 3.0)) * V3
    ) / (
        (F1 ** (7.0 / 3.0))
        * (jnp.cbrt(F1) - jnp.cbrt(F2))
        * (F2 ** (7.0 / 3.0))
        * (jnp.cbrt(F1) - jnp.cbrt(F3))
        * (jnp.cbrt(F2) - jnp.cbrt(F3))
        * (F3 ** (7.0 / 3.0))
    )
    rho3 = (
        (F2 ** (8.0 / 3.0)) * (F3 ** (7.0 / 3.0)) * V1
        - (F2 ** (7.0 / 3.0)) * (F3 ** (8.0 / 3.0)) * V1
        - (F1 ** (8.0 / 3.0)) * (F3 ** (7.0 / 3.0)) * V2
        + (F1 ** (7.0 / 3.0)) * (F3 ** (8.0 / 3.0)) * V2
        + (F1 ** (8.0 / 3.0)) * (F2 ** (7.0 / 3.0)) * V3
        - (F1 ** (7.0 / 3.0)) * (F2 ** (8.0 / 3.0)) * V3
    ) / (
        (F1 ** (7.0 / 3.0))
        * (jnp.cbrt(F1) - jnp.cbrt(F2))
        * (F2 ** (7.0 / 3.0))
        * (jnp.cbrt(F1) - jnp.cbrt(F3))
        * (jnp.cbrt(F2) - jnp.cbrt(F3))
        * (F3 ** (7.0 / 3.0))
    )

    # Now the intermediate region
    F1 = fAmpMatchIN
    F4 = fAmpRDMin

    d1 = (
        (
            (chi2 * (81 - 81 * delta - 44 * eta) + chi1 * (81 * (1 + delta) - 44 * eta))
            * PI
        )
        / 48.0
        + ((-969 + 1804 * eta) * (PI ** (2.0 / 3.0))) / (1008.0 * (F1 ** (1.0 / 3.0)))
        + (
            (
                -27312085
                - 10287648 * chi2**2
                + 10287648 * chi2**2 * delta
                - 10287648 * chi1**2 * (1 + delta)
                + 24
                * (
                    -1975055
                    + 857304 * chi1**2
                    - 994896 * chi1 * chi2
                    + 857304 * chi2**2
                )
                * eta
                + 35371056 * eta2
            )
            * (PI ** (4.0 / 3.0))
            * (F1 ** (1.0 / 3.0))
        )
        / 6.096384e6
        + (
            5
            * (PI ** (5.0 / 3.0))
            * (
                -6048 * chi1**2 * chi1 * (-1 - delta + (3 + delta) * eta)
                + chi1
                * (
                    287213 * (1 + delta)
                    - 4 * (93414 + 2083 * delta) * eta
                    - 35632 * eta2
                )
                + chi2
                * (
                    -((287213 + 6048 * chi2**2) * (-1 + delta))
                    + 4
                    * (-93414 + 1512 * chi2**2 * (-3 + delta) + 2083 * delta)
                    * eta
                    - 35632 * eta2
                )
                + 42840 * (-1 + 4 * eta) * PI
            )
            * (F1 ** (2.0 / 3.0))
        )
        / 96768.0
        - (
            (PI ** (2.0))
            * (
                -336
                * (
                    -3248849057
                    + 1809550512 * chi1**2
                    - 2954929824 * chi1 * chi2
                    + 1809550512 * chi2**2
                )
                * eta2
                - 324322727232 * eta2 * eta
                + 7
                * (
                    177520268561
                    + 29362199328 * chi2**2
                    - 29362199328 * chi2**2 * delta
                    + 29362199328 * chi1**2 * (1 + delta)
                    + 12160253952 * (chi1 + chi2 + chi1 * delta - chi2 * delta) * PI
                )
                + 12
                * eta
                * (
                    -545384828789
                    + 49568837472 * chi1 * chi2
                    - 12312458928 * chi2**2
                    - 21943440288 * chi2**2 * delta
                    + 77616 * chi1**2 * (-158633 + 282718 * delta)
                    - 8345272320 * (chi1 + chi2) * PI
                    + 21384760320 * (PI ** (2.0))
                )
            )
            * F1
        )
        / 3.0042980352e10
        + (7.0 / 3.0) * (F1 ** (4.0 / 3.0)) * rho1
        + (8.0 / 3.0) * (F1 ** (5.0 / 3.0)) * rho2
        + 3 * F1 * F1 * rho3
    )
    d4 = (
        -jnp.exp(-gamma2 * (F4 - fMs_RD) / (fMs_damp * gamma3))
        * gamma1
        * (
            (F4 - fMs_RD) * (F4 - fMs_RD) * gamma2
            + 2.0 * fMs_damp * (F4 - fMs_RD) * gamma3
            + fMs_damp * fMs_damp * gamma2 * gamma3 * gamma3
        )
        / (
            ((F4 - fMs_RD) * (F4 - fMs_RD) + (fMs_damp * gamma3) * (fMs_damp * gamma3))
            * (
                (F4 - fMs_RD) * (F4 - fMs_RD)
                + (fMs_damp * gamma3) * (fMs_damp * gamma3)
            )
        )
    )
    inspF1 = (
        pnInitial
        + (F1 ** (1.0 / 3.0)) * pnOneThird
        + (F1 ** (2.0 / 3.0)) * pnTwoThirds
        + F1 * pnThreeThirds
        + F1
        * (
            (F1 ** (1.0 / 3.0)) * pnFourThirds
            + (F1 ** (2.0 / 3.0)) * pnFiveThirds
            + F1 * pnSixThirds
            + F1 * ((F1 ** (1.0 / 3.0)) * rho1 + (F1 ** (2.0 / 3.0)) * rho2 + F1 * rho3)
        )
    )
    rdF4 = (
        jnp.exp(-(F4 - fMs_RD) * gammaR)
        * (gammaD13)
        / ((F4 - fMs_RD) * (F4 - fMs_RD) + gammaD2)
    )

    # Use d1 and d4 calculated above to get the derivative of the amplitude on the boundaries
    d1 = ((7.0 / 6.0) * (F1 ** (1.0 / 6.0)) / inspF1) - (
        (F1 ** (7.0 / 6.0)) * d1 / (inspF1 * inspF1)
    )
    d4 = ((7.0 / 6.0) * (F4 ** (1.0 / 6.0)) / rdF4) - (
        (F4 ** (7.0 / 6.0)) * d4 / (rdF4 * rdF4)
    )

    # Use a 4th order polynomial in intermediate - good extrapolation, recommended default fit
    F2 = F1 + (1.0 / 2.0) * (F4 - F1)
    F3 = 0.0

    V1 = (F1 ** (-7.0 / 6)) * inspF1
    V2 = (
        (
            (
                1.4873184918202145
                + 1974.6112656679577 * eta
                + 27563.641024162127 * eta2
                - 19837.908020966777 * eta2 * eta
            )
            / (1.0 + 143.29004876335128 * eta + 458.4097306093354 * eta2)
        )
        + (
            (
                StotR
                * (
                    27.952730865904343
                    + eta * (-365.55631765202895 - 260.3494489873286 * StotR)
                    + 3.2646808851249016 * StotR
                    + 3011.446602208493 * eta2 * StotR
                    - 19.38970173389662 * StotR**2
                    + eta2
                    * eta
                    * (
                        1612.2681322644232
                        - 6962.675551371755 * StotR
                        + 1486.4658089990298 * StotR**2
                    )
                )
            )
            / (12.647425554323242 - 10.540154508599963 * StotR + 1.0 * StotR**2)
        )
        + (chia * delta * (-0.016404056649860943 - 296.473359655246 * eta) * eta2)
    )
    V3 = 0.0
    V4 = (F4 ** (-7.0 / 6)) * rdF4

    V1 = 1.0 / V1
    V2 = 1.0 / V2
    V4 = 1.0 / V4

    # Reconstruct the phenomenological coefficients for the intermediate ansatz
    F12 = F1 * F1
    F13 = F12 * F1
    F14 = F13 * F1
    F15 = F14 * F1

    F22 = F2 * F2
    F23 = F22 * F2
    F24 = F23 * F2

    F42 = F4 * F4
    F43 = F42 * F4
    F44 = F43 * F4
    F45 = F44 * F4

    F1mF2 = F1 - F2
    F1mF4 = F1 - F4
    F2mF4 = F2 - F4

    F1mF22 = F1mF2 * F1mF2
    F2mF42 = F2mF4 * F2mF4
    F1mF43 = F1mF4 * F1mF4 * F1mF4

    delta0 = (
        -(d4 * F12 * F1mF22 * F1mF4 * F2 * F2mF4 * F4)
        + d1 * F1 * F1mF2 * F1mF4 * F2 * F2mF42 * F42
        + F42
        * (
            F2 * F2mF42 * (-4 * F12 + 3 * F1 * F2 + 2 * F1 * F4 - F2 * F4) * V1
            + F12 * F1mF43 * V2
        )
        + F12 * F1mF22 * F2 * (F1 * F2 - 2 * F1 * F4 - 3 * F2 * F4 + 4 * F42) * V4
    ) / (F1mF22 * F1mF43 * F2mF42)
    delta1 = (
        d4 * F1 * F1mF22 * F1mF4 * F2mF4 * (2 * F2 * F4 + F1 * (F2 + F4))
        + F4
        * (
            -(d1 * F1mF2 * F1mF4 * F2mF42 * (2 * F1 * F2 + (F1 + F2) * F4))
            - 2
            * F1
            * (
                F44 * (V1 - V2)
                + 3 * F24 * (V1 - V4)
                + F14 * (V2 - V4)
                + 4 * F23 * F4 * (-V1 + V4)
                + 2 * F13 * F4 * (-V2 + V4)
                + F1
                * (
                    2 * F43 * (-V1 + V2)
                    + 6 * F22 * F4 * (V1 - V4)
                    + 4 * F23 * (-V1 + V4)
                )
            )
        )
    ) / (F1mF22 * F1mF43 * F2mF42)
    delta2 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4 * (F12 + F2 * F4 + 2 * F1 * (F2 + F4)))
        + d1 * F1mF2 * F1mF4 * F2mF42 * (F1 * F2 + 2 * (F1 + F2) * F4 + F42)
        - 4 * F12 * F23 * V1
        + 3 * F1 * F24 * V1
        - 4 * F1 * F23 * F4 * V1
        + 3 * F24 * F4 * V1
        + 12 * F12 * F2 * F42 * V1
        - 4 * F23 * F42 * V1
        - 8 * F12 * F43 * V1
        + F1 * F44 * V1
        + F45 * V1
        + F15 * V2
        + F14 * F4 * V2
        - 8 * F13 * F42 * V2
        + 8 * F12 * F43 * V2
        - F1 * F44 * V2
        - F45 * V2
        - F1mF22
        * (
            F13
            + F2 * (3 * F2 - 4 * F4) * F4
            + F12 * (2 * F2 + F4)
            + F1 * (3 * F2 - 4 * F4) * (F2 + 2 * F4)
        )
        * V4
    ) / (F1mF22 * F1mF43 * F2mF42)
    delta3 = (
        d4 * F1mF22 * F1mF4 * F2mF4 * (2 * F1 + F2 + F4)
        - d1 * F1mF2 * F1mF4 * F2mF42 * (F1 + F2 + 2 * F4)
        + 2
        * (
            F44 * (-V1 + V2)
            + 2 * F12 * F2mF42 * (V1 - V4)
            + 2 * F22 * F42 * (V1 - V4)
            + 2 * F13 * F4 * (V2 - V4)
            + F24 * (-V1 + V4)
            + F14 * (-V2 + V4)
            + 2
            * F1
            * F4
            * (F42 * (V1 - V2) + F22 * (V1 - V4) + 2 * F2 * F4 * (-V1 + V4))
        )
    ) / (F1mF22 * F1mF43 * F2mF42)
    delta4 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4)
        + d1 * F1mF2 * F1mF4 * F2mF42
        - 3 * F1 * F22 * V1
        + 2 * F23 * V1
        + 6 * F1 * F2 * F4 * V1
        - 3 * F22 * F4 * V1
        - 3 * F1 * F42 * V1
        + F43 * V1
        + F13 * V2
        - 3 * F12 * F4 * V2
        + 3 * F1 * F42 * V2
        - F43 * V2
        - F1mF22 * (F1 + 2 * F2 - 3 * F4) * V4
    ) / (F1mF22 * F1mF43 * F2mF42)
    delta5 = 0.0

    Overallamp = amp0 * ampNorm

    amplitudeIMR = jnp.where(
        fM_s <= fAmpMatchIN,
        (
            pnInitial
            + (fM_s ** (1.0 / 3.0)) * pnOneThird
            + (fM_s ** (2.0 / 3.0)) * pnTwoThirds
            + fM_s * pnThreeThirds
            + fM_s
            * (
                (fM_s ** (1.0 / 3.0)) * pnFourThirds
                + (fM_s ** (2.0 / 3.0)) * pnFiveThirds
                + fM_s * pnSixThirds
                + fM_s
                * (
                    (fM_s ** (1.0 / 3.0)) * rho1
                    + (fM_s ** (2.0 / 3.0)) * rho2
                    + fM_s * rho3
                )
            )
        ),
        jnp.where(
            fM_s <= fAmpRDMin,
            (fM_s ** (7.0 / 6.0))
            / (
                delta0
                + fM_s
                * (
                    delta1
                    + fM_s
                    * (delta2 + fM_s * (delta3 + fM_s * (delta4 + fM_s * delta5)))
                )
            ),
            jnp.where(
                fM_s <= IMRPhenomX_utils.fM_CUT,
                jnp.exp(-(fM_s - fMs_RD) * gammaR)
                * (gammaD13)
                / ((fM_s - fMs_RD) * (fM_s - fMs_RD) + gammaD2),
                0.0,
            ),
        ),
    )

    return Overallamp * amplitudeIMR * (fM_s ** (-7.0 / 6.0))


# @jax.jit
# Removed theta_extrinsic and only return the phase for now
def _gen_IMRPhenomXAS(
    f: Array, theta_intrinsic: Array, theta_extrinsic: Array, coeffs: Array
):
    f_ref = jnp.amin(f, axis=0)
    m1, m2, chi1, chi2 = theta_intrinsic
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
    chia = chi1 - chi2
    fM_s = f * M_s
    fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1, m2, chi1, chi2
    )
    Psi = Phase(f, theta_intrinsic, coeffs)

    lina, linb, psi4tostrain = IMRPhenomX_utils.calc_phaseatpeak(
        eta, StotR, chia, delta
    )
    dphi22Ref = (
        jax.grad(Phase)((fMs_RD - fMs_damp) / M_s, theta_intrinsic, coeffs) / M_s
    )
    linb = linb - dphi22Ref - 2.0 * PI * (500.0 + psi4tostrain)
    phifRef = (
        -(Phase(f_ref, theta_intrinsic, coeffs) + linb * (f_ref * M_s) + lina)
        + PI / 4.0
        + PI
    )
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] - theta_extrinsic[2]
    Psi = Psi + (linb * fM_s) + lina + phifRef - 2 * PI + ext_phase_contrib

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
      h0 (array): Complex gravitational wave strain
    """
    # Lets make this easier by starting in Mchirp and eta space
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    theta_intrinsic = jnp.array([m1, m2, params[2], params[3]])
    theta_extrinsic = jnp.array([params[4], params[5], params[6]])
    coeffs = IMRPhenomX_utils.PhenomX_coeff_table

    h0 = _gen_IMRPhenomXAS(f, theta_intrinsic, theta_extrinsic, coeffs)
    return h0
