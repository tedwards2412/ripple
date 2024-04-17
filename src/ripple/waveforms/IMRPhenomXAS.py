# from math import PI
import jax
import jax.numpy as jnp
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ripple.waveforms import IMRPhenomX_utils
from ..typing import Array

from ripple import Mc_eta_to_ms

eqspin_indx = 10
uneqspin_indx = 39

amp_eqspin_indx = 8
amp_uneqspin_indx = 36


def get_inspiral_phase(fM_s: Array, theta: Array, phase_coeffs: Array) -> Array:
    """
    Calculate the inspiral phase for the IMRPhenomD waveform.
    """
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
    phi5L = ((5.0 * (46374.0 - 6552.0 * eta) * PI) / 4536.0) * PI ** (5.0 / 3.0) + (
        (
            -732985.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
            - 560.0 * (-1213.0 * (chi1 + chi2) + 63.0 * (chi1 - chi2) * delta) * eta
            + 85680.0 * (chi1 + chi2) * eta2
        )
        / 4536.0
    ) * PI ** (5.0 / 3.0)
    phi6L = (-6848.0 / 63.0) * PI**2.0
    phi6 = (
        (
            11583231236531.0 / 4.69421568e9
            - (5.0 * eta * (3147553127.0 + 588.0 * eta * (-45633.0 + 102260.0 * eta)))
            / 3.048192e6
            - (6848.0 * EulerGamma) / 21.0
            - (640.0 * PI**2.0) / 3.0
            + (2255.0 * eta * PI**2.0) / 12.0
            - (13696.0 * jnp.log(2.0)) / 21.0
            - (6848.0 * jnp.log(PI)) / 63.0
        )
        * PI**2.0
        + (
            (
                5
                * (
                    227.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                    - 156.0 * (chi1 + chi2) * eta
                )
                * PI
            )
            / 3.0
        )
        * PI**2.0
        + (
            (
                5.0
                * (
                    20.0 * chi1L2L * eta * (11763.0 + 12488.0 * eta)
                    + 7.0
                    * chi2L2
                    * (
                        -15103.0 * (-1 + delta)
                        + 2.0 * (-21683.0 + 6580.0 * delta) * eta
                        - 9808.0 * eta2
                    )
                    - 7.0
                    * chi1L2
                    * (
                        -15103.0 * (1 + delta)
                        + 2.0 * (21683.0 + 6580.0 * delta) * eta
                        + 9808.0 * eta2
                    )
                )
            )
            / 4032.0
        )
        * PI**2.0
    )
    phi7 = (
        ((5.0 * (15419335.0 + 168.0 * (75703.0 - 29618.0 * eta) * eta) * PI) / 254016.0)
        * PI ** (7.0 / 3.0)
        + (
            (
                5.0
                * (
                    -5030016755.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                    + 4.0
                    * (
                        2113331119.0 * (chi1 + chi2)
                        + 675484362.0 * (chi1 - chi2) * delta
                    )
                    * eta
                    - 1008.0
                    * (208433.0 * (chi1 + chi2) + 25011.0 * (chi1 - chi2) * delta)
                    * eta2
                    + 90514368.0 * (chi1 + chi2) * eta3
                )
            )
            / 6.096384e6
        )
        * PI ** (7.0 / 3.0)
        + (
            -5.0
            * (
                57.0 * chi1L2 * (1 + delta - 2 * eta)
                + 220.0 * chi1L2L * eta
                - 57.0 * chi2L2 * (-1 + delta + 2 * eta)
            )
            * PI
        )
        * PI ** (7.0 / 3.0)
        + (
            (
                14585.0 * (-(chi2L3 * (-1 + delta)) + chi1L3 * (1 + delta))
                - 5.0
                * (
                    chi2L3 * (8819.0 - 2985.0 * delta)
                    + 8439.0 * chi1 * chi2L2 * (-1.0 + delta)
                    - 8439.0 * chi1L2 * chi2 * (1.0 + delta)
                    + chi1L3 * (8819.0 + 2985.0 * delta)
                )
                * eta
                + 40.0
                * (chi1 + chi2)
                * (17.0 * chi1L2 - 14.0 * chi1L2L + 17.0 * chi2L2)
                * eta2
            )
            / 48.0
        )
        * PI ** (7.0 / 3.0)
    )
    phi8 = (
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
            * (-1.0 + jnp.log(PI))
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
        IMRPhenomX_utils.nospin_CV(phase_coeffs[0, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(phase_coeffs[0, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(phase_coeffs[0, uneqspin_indx:], eta, S, chia)
    )
    CV_phase_Ins1 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[1, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(phase_coeffs[1, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(phase_coeffs[1, uneqspin_indx:], eta, S, chia)
    )
    CV_phase_Ins2 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[2, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(phase_coeffs[2, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(phase_coeffs[2, uneqspin_indx:], eta, S, chia)
    )

    # NOTE: This CV_phase_Ins3 disagrees slightly with the value in WF4py at non-zero spin
    CV_phase_Ins3 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[3, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(phase_coeffs[3, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(phase_coeffs[3, uneqspin_indx:], eta, S, chia)
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
    fM_s: Array, theta: Array, phase_coeffs: Array, dPhaseIN, dPhaseRD, cL
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
        IMRPhenomX_utils.nospin_CV(phase_coeffs[4, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[4, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[4, uneqspin_indx:], eta, StotR, chia
        )
    )

    v3IMmRDv4 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[5, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[5, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[5, uneqspin_indx:], eta, StotR, chia
        )
    )
    v2IM = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[6, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[6, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[6, uneqspin_indx:], eta, StotR, chia
        )
    )

    # NOTE: This is different to WF4py and driving the difference in CV_phase_Int3
    d43 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[7, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[7, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[7, uneqspin_indx:], eta, StotR, chia
        )
    )

    CV_phase_RD3 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[11, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[11, uneqspin_indx:], eta, StotR, chia
        )
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
    fM_s: Array, theta: Array, phase_coeffs: Array
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
        IMRPhenomX_utils.nospin_CV(phase_coeffs[8, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[8, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[8, uneqspin_indx:], eta, StotR, chia
        )
    )
    CV_phase_RD1 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[9, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[9, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[9, uneqspin_indx:], eta, StotR, chia
        )
    )
    CV_phase_RD2 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[10, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[10, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[10, uneqspin_indx:], eta, StotR, chia
        )
    )
    CV_phase_RD3 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[11, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[11, uneqspin_indx:], eta, StotR, chia
        )
    )
    CV_phase_RD4 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[12, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[12, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[12, uneqspin_indx:], eta, StotR, chia
        )
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
def Phase(f: Array, theta: Array, phase_coeffs: Array) -> Array:
    """
    Computes the phase of the PhenomD waveform following 1508.07253.
    Sets time and phase of coealence to be zero.

    Returns:
    --------
        phase (array): Phase of the GW as a function of frequency
    """
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

    # Calculate the inspiral and raw merger phase (required for the intemediate phase)
    phi_Ins = get_inspiral_phase(fM_s, theta, phase_coeffs)
    phi_MRD, (cL, CV_phase_RD0) = get_mergerringdown_raw_phase(
        fM_s, theta, phase_coeffs
    )

    # Get matching points
    # Here we want to evaluate the gradient and the phase of the raw phase functions
    # in order to enforce C1 continuity at the transition frequencies.
    # This procedure is identical to IMRPhenomD, see IMRPhenomD.py for more details
    phi_Ins_match_f1, dphi_Ins_match_f1 = jax.value_and_grad(get_inspiral_phase)(
        f1_Ms, theta, phase_coeffs
    )
    phi_MRD_match_f2, dphi_MRD_match_f2 = jax.value_and_grad(
        get_mergerringdown_raw_phase, has_aux=True
    )(f2_Ms, theta, phase_coeffs)
    phi_MRD_match_f2, _ = get_mergerringdown_raw_phase(f2_Ms, theta, phase_coeffs)

    # Now find the intermediate phase
    phi_Int_match_f1, dphi_Int_match_f1 = jax.value_and_grad(
        get_intermediate_raw_phase
    )(f1_Ms, theta, phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL)
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    phi_Int_func = (
        lambda fM_s_: get_intermediate_raw_phase(
            fM_s_, theta, phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL
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
        * jnp.heaviside(IMRPhenomX_utils.fM_CUT - fM_s, 0.5)
    )

    return phase


def get_Amp0(fM_s: Array, eta: float) -> Array:
    Amp0 = (
        (2.0 / 3.0 * eta) ** (1.0 / 2.0) * (fM_s) ** (-7.0 / 6.0) * PI ** (-1.0 / 6.0)
    )
    return Amp0


def get_inspiral_Amp(fM_s: Array, theta: Array, amp_coeffs: Array) -> Array:
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
    chia = chi1 - chi2

    _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    fMs_AmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fMs_AmpMatchIN = fMs_AmpInsMax

    A0 = 1.0
    # A1 = 0.0
    A2 = ((-969.0 + 1804.0 * eta) / 672.0) * (PI ** (2.0 / 3.0))
    A3 = (
        (
            81.0 * (chi1 + chi2)
            + 81.0 * chi1 * delta
            - 81.0 * chi2 * delta
            - 44.0 * (chi1 + chi2) * eta
        )
        / 48.0
    ) * PI
    A4 = (
        (
            -27312085.0
            - 10287648.0 * chi1**2.0 * (1.0 + delta)
            + 24.0
            * (
                428652.0 * chi2**2 * (-1 + delta)
                + (
                    -1975055.0
                    + 10584.0 * (81.0 * chi1**2.0 - 94.0 * chi1 * chi2 + 81.0 * chi2**2)
                )
                * eta
                + 1473794.0 * eta2
            )
        )
        / 8.128512e6
    ) * (PI ** (4.0 / 3.0))
    A5 = (
        (
            -6048.0 * chi1**2.0 * chi1 * (-1.0 - delta + (3.0 + delta) * eta)
            + chi2
            * (
                -((287213.0 + 6048.0 * chi2**2) * (-1.0 + delta))
                + 4
                * (-93414.0 + 1512.0 * chi2**2.0 * (-3.0 + delta) + 2083.0 * delta)
                * eta
                - 35632.0 * eta2
            )
            + chi1
            * (
                287213.0 * (1.0 + delta)
                - 4.0 * eta * (93414.0 + 2083.0 * delta + 8908.0 * eta)
            )
            + 42840.0 * (-1.0 + 4.0 * eta) * PI
        )
        / 32256.0
    ) * (PI ** (5.0 / 3.0))
    A6 = (
        (
            -1242641879927.0
            + 12.0
            * (
                28.0
                * (
                    -3248849057.0
                    + 11088.0
                    * (
                        163199.0 * chi1**2.0
                        - 266498.0 * chi1 * chi2
                        + 163199.0 * chi2**2.0
                    )
                )
                * eta2
                + 27026893936.0 * eta2 * eta
                - 116424.0
                * (
                    147117.0
                    * (-(chi2**2.0 * (-1.0 + delta)) + chi1**2.0 * (1.0 + delta))
                    + 60928.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) * PI
                )
                + eta
                * (
                    545384828789.0
                    - 77616.0
                    * (
                        638642.0 * chi1 * chi2
                        + chi1**2.0 * (-158633.0 + 282718.0 * delta)
                        - chi2**2.0 * (158633.0 + 282718.0 * delta)
                        - 107520.0 * (chi1 + chi2) * PI
                        + 275520.0 * PI**2
                    )
                )
            )
        )
        / 6.0085960704e10
    ) * PI**2

    # Now we need to get the higher order components

    CV_Amp_Ins0 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[0, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[0, amp_eqspin_indx:amp_uneqspin_indx], eta, S
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[0, amp_uneqspin_indx:], eta, S, chia
        )
    )
    CV_Amp_Ins1 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[1, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[1, amp_eqspin_indx:amp_uneqspin_indx], eta, S
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[1, amp_uneqspin_indx:], eta, S, chia
        )
    )
    CV_Amp_Ins2 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[2, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[2, amp_eqspin_indx:amp_uneqspin_indx], eta, S
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[2, amp_uneqspin_indx:], eta, S, chia
        )
    )

    CP_Amp_Ins0 = 0.50 * fMs_AmpMatchIN
    CP_Amp_Ins1 = 0.75 * fMs_AmpMatchIN
    CP_Amp_Ins2 = 1.00 * fMs_AmpMatchIN

    rho1 = (
        -((CP_Amp_Ins1 ** (8.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins0)
        + CP_Amp_Ins1**3 * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins0
        + (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins1
        - CP_Amp_Ins0**3 * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins1
        - (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins1**3) * CV_Amp_Ins2
        + CP_Amp_Ins0**3 * (CP_Amp_Ins1 ** (8.0 / 3.0)) * CV_Amp_Ins2
    ) / (
        (CP_Amp_Ins0 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins1))
        * (CP_Amp_Ins1 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins2))
        * (jnp.cbrt(CP_Amp_Ins1) - jnp.cbrt(CP_Amp_Ins2))
        * (CP_Amp_Ins2 ** (7.0 / 3.0))
    )
    rho2 = (
        (CP_Amp_Ins1 ** (7.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins0
        - CP_Amp_Ins1**3 * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins0
        - (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins1
        + CP_Amp_Ins0**3 * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins1
        + (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins1**3) * CV_Amp_Ins2
        - CP_Amp_Ins0**3 * (CP_Amp_Ins1 ** (7.0 / 3.0)) * CV_Amp_Ins2
    ) / (
        (CP_Amp_Ins0 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins1))
        * (CP_Amp_Ins1 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins2))
        * (jnp.cbrt(CP_Amp_Ins1) - jnp.cbrt(CP_Amp_Ins2))
        * (CP_Amp_Ins2 ** (7.0 / 3.0))
    )
    rho3 = (
        (CP_Amp_Ins1 ** (8.0 / 3.0)) * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins0
        - (CP_Amp_Ins1 ** (7.0 / 3.0)) * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins0
        - (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins1
        + (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins1
        + (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins1 ** (7.0 / 3.0)) * CV_Amp_Ins2
        - (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins1 ** (8.0 / 3.0)) * CV_Amp_Ins2
    ) / (
        (CP_Amp_Ins0 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins1))
        * (CP_Amp_Ins1 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins2))
        * (jnp.cbrt(CP_Amp_Ins1) - jnp.cbrt(CP_Amp_Ins2))
        * (CP_Amp_Ins2 ** (7.0 / 3.0))
    )

    Amp_Ins = (
        A0
        # A1 is missed since its zero
        + A2 * (fM_s ** (2.0 / 3.0))
        + A3 * fM_s
        + A4 * (fM_s ** (4.0 / 3.0))
        + A5 * (fM_s ** (5.0 / 3.0))
        + A6 * (fM_s**2.0)
        # # Now we add the coefficient terms
        + rho1 * (fM_s ** (7.0 / 3.0))
        + rho2 * (fM_s ** (8.0 / 3.0))
        + rho3 * (fM_s**3.0)
    )

    return Amp_Ins


def get_intermediate_Amp(
    fM_s: Array, theta: Array, amp_coeffs: Array, fMs_AmpRDMin
) -> Array:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    eta2 = eta * eta
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)

    # Spin variables
    chia = chi1 - chi2

    # Now the intermediate region
    _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    fMs_AmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fMs_AmpMatchIN = fMs_AmpInsMax
    FMs1 = fMs_AmpMatchIN
    # This needs to come from outside
    FMs4 = fMs_AmpRDMin

    inspFMs1, d1 = jax.value_and_grad(get_inspiral_Amp)(FMs1, theta, amp_coeffs)
    rdFMs4, d4 = jax.value_and_grad(get_mergerringdown_Amp, has_aux=True)(
        FMs4, theta, amp_coeffs
    )
    rdFMs4 = rdFMs4[0]

    # Use d1 and d4 calculated above to get the derivative of the amplitude on the boundaries
    d1 = ((7.0 / 6.0) * (FMs1 ** (1.0 / 6.0)) / inspFMs1) - (
        (FMs1 ** (7.0 / 6.0)) * d1 / (inspFMs1 * inspFMs1)
    )
    d4 = ((7.0 / 6.0) * (FMs4 ** (1.0 / 6.0)) / rdFMs4) - (
        (FMs4 ** (7.0 / 6.0)) * d4 / (rdFMs4 * rdFMs4)
    )

    # Use a 4th order polynomial in intermediate - good extrapolation, recommended default fit
    FMs2 = FMs1 + (1.0 / 2.0) * (FMs4 - FMs1)

    V1 = (FMs1 ** (-7.0 / 6)) * inspFMs1

    V2 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[3, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[3, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[3, amp_uneqspin_indx:], eta, StotR, chia
        )
    )
    V4 = (FMs4 ** (-7.0 / 6)) * rdFMs4

    V1 = 1.0 / V1
    V2 = 1.0 / V2
    V4 = 1.0 / V4

    # Reconstruct the phenomenological coefficients for the intermediate ansatz
    F12 = FMs1 * FMs1
    F13 = F12 * FMs1
    F14 = F13 * FMs1
    F15 = F14 * FMs1

    F22 = FMs2 * FMs2
    F23 = F22 * FMs2
    F24 = F23 * FMs2

    F42 = FMs4 * FMs4
    F43 = F42 * FMs4
    F44 = F43 * FMs4
    F45 = F44 * FMs4

    F1mF2 = FMs1 - FMs2
    F1mF4 = FMs1 - FMs4
    F2mF4 = FMs2 - FMs4

    F1mF22 = F1mF2 * F1mF2
    F2mF42 = F2mF4 * F2mF4
    F1mF43 = F1mF4 * F1mF4 * F1mF4

    delta0 = (
        -(d4 * F12 * F1mF22 * F1mF4 * FMs2 * F2mF4 * FMs4)
        + d1 * FMs1 * F1mF2 * F1mF4 * FMs2 * F2mF42 * F42
        + F42
        * (
            FMs2
            * F2mF42
            * (-4 * F12 + 3 * FMs1 * FMs2 + 2 * FMs1 * FMs4 - FMs2 * FMs4)
            * V1
            + F12 * F1mF43 * V2
        )
        + F12
        * F1mF22
        * FMs2
        * (FMs1 * FMs2 - 2 * FMs1 * FMs4 - 3 * FMs2 * FMs4 + 4 * F42)
        * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta1 = (
        d4 * FMs1 * F1mF22 * F1mF4 * F2mF4 * (2 * FMs2 * FMs4 + FMs1 * (FMs2 + FMs4))
        + FMs4
        * (
            -(d1 * F1mF2 * F1mF4 * F2mF42 * (2 * FMs1 * FMs2 + (FMs1 + FMs2) * FMs4))
            - 2
            * FMs1
            * (
                F44 * (V1 - V2)
                + 3 * F24 * (V1 - V4)
                + F14 * (V2 - V4)
                + 4 * F23 * FMs4 * (-V1 + V4)
                + 2 * F13 * FMs4 * (-V2 + V4)
                + FMs1
                * (
                    2 * F43 * (-V1 + V2)
                    + 6 * F22 * FMs4 * (V1 - V4)
                    + 4 * F23 * (-V1 + V4)
                )
            )
        )
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta2 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4 * (F12 + FMs2 * FMs4 + 2 * FMs1 * (FMs2 + FMs4)))
        + d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1 * FMs2 + 2 * (FMs1 + FMs2) * FMs4 + F42)
        - 4 * F12 * F23 * V1
        + 3 * FMs1 * F24 * V1
        - 4 * FMs1 * F23 * FMs4 * V1
        + 3 * F24 * FMs4 * V1
        + 12 * F12 * FMs2 * F42 * V1
        - 4 * F23 * F42 * V1
        - 8 * F12 * F43 * V1
        + FMs1 * F44 * V1
        + F45 * V1
        + F15 * V2
        + F14 * FMs4 * V2
        - 8 * F13 * F42 * V2
        + 8 * F12 * F43 * V2
        - FMs1 * F44 * V2
        - F45 * V2
        - F1mF22
        * (
            F13
            + FMs2 * (3 * FMs2 - 4 * FMs4) * FMs4
            + F12 * (2 * FMs2 + FMs4)
            + FMs1 * (3 * FMs2 - 4 * FMs4) * (FMs2 + 2 * FMs4)
        )
        * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta3 = (
        d4 * F1mF22 * F1mF4 * F2mF4 * (2 * FMs1 + FMs2 + FMs4)
        - d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1 + FMs2 + 2 * FMs4)
        + 2
        * (
            F44 * (-V1 + V2)
            + 2 * F12 * F2mF42 * (V1 - V4)
            + 2 * F22 * F42 * (V1 - V4)
            + 2 * F13 * FMs4 * (V2 - V4)
            + F24 * (-V1 + V4)
            + F14 * (-V2 + V4)
            + 2
            * FMs1
            * FMs4
            * (F42 * (V1 - V2) + F22 * (V1 - V4) + 2 * FMs2 * FMs4 * (-V1 + V4))
        )
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta4 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4)
        + d1 * F1mF2 * F1mF4 * F2mF42
        - 3 * FMs1 * F22 * V1
        + 2 * F23 * V1
        + 6 * FMs1 * FMs2 * FMs4 * V1
        - 3 * F22 * FMs4 * V1
        - 3 * FMs1 * F42 * V1
        + F43 * V1
        + F13 * V2
        - 3 * F12 * FMs4 * V2
        + 3 * FMs1 * F42 * V2
        - F43 * V2
        - F1mF22 * (FMs1 + 2 * FMs2 - 3 * FMs4) * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    Amp_Int = (fM_s ** (7.0 / 6.0)) / (
        delta0 + fM_s * (delta1 + fM_s * (delta2 + fM_s * (delta3 + fM_s * delta4)))
    )

    return Amp_Int


def get_mergerringdown_Amp(
    fM_s: Array,
    theta: Array,
    amp_coeffs: Array,
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
    chia = chi1 - chi2

    fMs_RD, fMs_damp, _, _ = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)

    gamma2 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[4, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[4, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[4, amp_uneqspin_indx:], eta, StotR, chia
        )
    )
    gamma3 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[5, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[5, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[5, amp_uneqspin_indx:], eta, StotR, chia
        )
    )
    fMs_AmpRDMin = jnp.where(
        gamma2 <= 1.0,
        jnp.fabs(
            fMs_RD
            + fMs_damp * gamma3 * (jnp.sqrt(1.0 - gamma2 * gamma2) - 1.0) / gamma2
        ),
        jnp.fabs(fMs_RD + fMs_damp * (-1.0) * gamma3 / gamma2),
    )
    v1RD = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[6, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[6, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[6, amp_uneqspin_indx:], eta, StotR, chia
        )
    )
    FMs1 = fMs_AmpRDMin

    gamma1 = (
        (v1RD / (fMs_damp * gamma3))
        * (
            FMs1 * FMs1
            - 2.0 * FMs1 * fMs_RD
            + fMs_RD * fMs_RD
            + fMs_damp * fMs_damp * gamma3 * gamma3
        )
        * jnp.exp(((FMs1 - fMs_RD) * gamma2) / (fMs_damp * gamma3))
    )
    gammaR = gamma2 / (fMs_damp * gamma3)
    gammaD2 = (gamma3 * fMs_damp) * (gamma3 * fMs_damp)
    gammaD13 = fMs_damp * gamma1 * gamma3

    Amp_RD = (
        jnp.exp(-(fM_s - fMs_RD) * gammaR)
        * (gammaD13)
        / ((fM_s - fMs_RD) * (fM_s - fMs_RD) + gammaD2)
    )

    return Amp_RD, fMs_AmpRDMin


def Amp(f: Array, theta: Array, amp_coeffs: Array, D=1.0) -> Array:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s
    _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M_s**2 / ((D * m_per_Mpc) / C)
    ampNorm = jnp.sqrt(2.0 * eta / 3.0) * (PI ** (-1.0 / 6.0))

    fMs_AmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fMs_AmpMatchIN = fMs_AmpInsMax

    # Below
    Overallamp = amp0 * ampNorm

    Amp_Ins = get_inspiral_Amp(fM_s, theta, amp_coeffs)
    Amp_RD, fMs_AmpRDMin = get_mergerringdown_Amp(fM_s, theta, amp_coeffs)
    Amp_Int = get_intermediate_Amp(fM_s, theta, amp_coeffs, fMs_AmpRDMin)

    Amp = (
        Amp_Ins * jnp.heaviside(fMs_AmpMatchIN - fM_s, 0.5)
        + jnp.heaviside(fM_s - fMs_AmpMatchIN, 0.5)
        * Amp_Int
        * jnp.heaviside(fMs_AmpRDMin - fM_s, 0.5)
        + Amp_RD
        * jnp.heaviside(fM_s - fMs_AmpRDMin, 0.5)
        * jnp.heaviside(IMRPhenomX_utils.fM_CUT - fM_s, 0.5)
    )

    return Overallamp * Amp * (fM_s ** (-7.0 / 6.0))


# @jax.jit
def _gen_IMRPhenomXAS(
    f: Array,
    theta_intrinsic: Array,
    theta_extrinsic: Array,
    phase_coeffs: Array,
    amp_coeffs: Array,
    f_ref: float,
):
    m1, m2, chi1, chi2 = theta_intrinsic
    m1_s = m1 * gt
    m2_s = m2 * gt

    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)

    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
    chia = chi1 - chi2

    fM_s = f * M_s
    fMs_RD, fMs_damp, _, _ = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    Psi = Phase(f, theta_intrinsic, phase_coeffs)

    # Generate the linear in f and constant contribution to the phase in order
    # to roll the waveform such that the peak is at the input tc and phic
    lina, linb, psi4tostrain = IMRPhenomX_utils.calc_phaseatpeak(
        eta, StotR, chia, delta
    )
    dphi22Ref = (
        jax.grad(Phase)((fMs_RD - fMs_damp) / M_s, theta_intrinsic, phase_coeffs) / M_s
    )
    linb = linb - dphi22Ref - 2.0 * PI * (500.0 + psi4tostrain)
    phifRef = (
        -(Phase(f_ref, theta_intrinsic, phase_coeffs) + linb * (f_ref * M_s) + lina)
        + PI / 4.0
        + PI
    )
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] - theta_extrinsic[2]
    Psi = Psi + (linb * fM_s) + lina + phifRef - 2 * PI + ext_phase_contrib

    A = Amp(f, theta_intrinsic, amp_coeffs, D=theta_extrinsic[0])
    h0 = A * jnp.exp(1j * Psi)
    return h0


def gen_IMRPhenomXAS(f: Array, params: Array, f_ref: float):
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
    phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    amp_coeffs = IMRPhenomX_utils.PhenomX_amp_coeff_table

    h0 = _gen_IMRPhenomXAS(
        f, theta_intrinsic, theta_extrinsic, phase_coeffs, amp_coeffs, f_ref
    )
    return h0


def gen_IMRPhenomXAS_hphc(f: Array, params: Array, f_ref: float):
    """
    Generate PhenomXAS frequency domain waveform following 2001.11412.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
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
    iota = params[7]
    h0 = gen_IMRPhenomXAS(f, params, f_ref)

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
