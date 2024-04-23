"""
This file implements the TaylorF2 waveform, as described in the LALSuite library.
"""

import jax.numpy as jnp
from ..constants import EulerGamma, gt, m_per_Mpc, PI, MRSUN
from ..typing import Array
from ripple import Mc_eta_to_ms, lambda_tildes_to_lambdas
from .IMRPhenom_tidal_utils import get_quadparam_octparam

###########################
### AUXILIARY FUNCTIONS ###
###########################


def get_3PNSOCoeff(mByM):
    return mByM * (25.0 + 38.0 / 3.0 * mByM)


def get_5PNSOCoeff(mByM):
    return -mByM * (
        1391.5 / 8.4
        - mByM * (1.0 - mByM) * 10.0 / 3.0
        + mByM * (1276.0 / 8.1 + mByM * (1.0 - mByM) * 170.0 / 9.0)
    )


def get_6PNSOCoeff(mByM):
    return PI * mByM * (1490.0 / 3.0 + mByM * 260.0)


def get_7PNSOCoeff(mByM):
    eta = mByM * (1.0 - mByM)
    return mByM * (
        -17097.8035 / 4.8384
        + eta * 28764.25 / 6.72
        + eta * eta * 47.35 / 1.44
        + mByM
        * (-7189.233785 / 1.524096 + eta * 458.555 / 3.024 - eta * eta * 534.5 / 7.2)
    )


def get_4PNS1S2Coeff(eta):
    return 247.0 / 4.8 * eta


def get_4PNS1S2OCoeff(eta):
    return -721.0 / 4.8 * eta


def get_4PNQM2SOCoeff(mByM):
    return -720.0 / 9.6 * mByM * mByM


def get_4PNSelf2SOCoeff(mByM):
    return 1.0 / 9.6 * mByM * mByM


def get_4PNQM2SCoeff(mByM):
    return 240.0 / 9.6 * mByM * mByM


def get_4PNSelf2SCoeff(mByM):
    return -7.0 / 9.6 * mByM * mByM


def get_6PNS1S2OCoeff(eta):
    return (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta


def get_6PNSelf2SCoeff(mByM):
    return (
        (-4108.25 / 6.72 - 108.5 / 1.2 * mByM + 125.5 / 3.6 * mByM * mByM) * mByM * mByM
    )


def get_6PNQM2SCoeff(mByM):
    return (4703.5 / 8.4 + 2935.0 / 6.0 * mByM - 120.0 * mByM * mByM) * mByM * mByM


def get_10PNTidalCoeff(mByM):
    return (-288.0 + 264.0 * mByM) * mByM * mByM * mByM * mByM


def get_12PNTidalCoeff(mByM):
    return (
        (
            -15895.0 / 28.0
            + 4595.0 / 28.0 * mByM
            + 5715.0 / 14.0 * mByM * mByM
            - 325.0 / 7.0 * mByM * mByM * mByM
        )
        * mByM
        * mByM
        * mByM
        * mByM
    )


def get_13PNTidalCoeff(mByM):
    return mByM * mByM * mByM * mByM * 24.0 * (12.0 - 11.0 * mByM) * PI


def get_14PNTidalCoeff(mByM):
    mByM3 = mByM * mByM * mByM
    mByM4 = mByM3 * mByM
    return (
        -mByM4
        * 5.0
        * (
            193986935.0 / 571536.0
            - 14415613.0 / 381024.0 * mByM
            - 57859.0 / 378.0 * mByM * mByM
            - 209495.0 / 1512.0 * mByM3
            + 965.0 / 54.0 * mByM4
            - 4.0 * mByM4 * mByM
        )
    )


def get_15PNTidalCoeff(mByM):
    mByM2 = mByM * mByM
    mByM3 = mByM2 * mByM
    mByM4 = mByM3 * mByM
    return (
        mByM4
        * 1.0
        / 28.0
        * PI
        * (27719.0 - 22415.0 * mByM + 7598.0 * mByM2 - 10520.0 * mByM3)
    )


def get_flux_0PNCoeff(eta):
    return 32.0 * eta * eta / 5.0


def get_energy_0PNCoeff(eta):
    return -eta / 2.0


################
### WAVEFORM ###
################


def get_PNPhasing_F2(
    m1: float, m2: float, S1z: float, S2z: float, lambda1: float, lambda2: float
) -> tuple[dict, dict]:
    """
    Gets dictionaries giving the phasing coefficients to be used in the approximant.
    Keys are the different PN orders, with values being the corresponding coefficient.
    This follows the implementation of XLALSimInspiralPNPhasing_F2 from lalsuite.

    Args:
        m1 (float): Mass of first (heavier) object
        m2 (float): Mass of second (lighter) object
        S1z (float): z-component of spin of first object
        S2z (float): z-component of spin of second object
        lambda1 (float): Tidal deformability first object
        lambda2 (float): Tidal deformability first object

    Returns:
        tuple[dict, dict]: phasing_coeffs, phasing_log_coeffs as defined in the LAL source code, coefficients for various PN orders.
    """

    # Mass variables
    M = m1 + m2
    m1M = m1 / M
    m2M = m2 / M
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # Auxiliary variables
    pfaN = 3.0 / (128.0 * eta)  # prefactor
    S1z_2 = S1z**2  # spin one squared
    S2z_2 = S2z**2  # spin two squared
    S1z_dot_S2z = S1z * S2z  # dot product spins
    qm_def1, _ = get_quadparam_octparam(lambda1)  # quadrupole parameter 1
    qm_def2, _ = get_quadparam_octparam(lambda2)  # quadrupole parameter 2

    # We are going to build a dictionary with coefficients for varying PN orders
    phasing_coeffs = dict()
    phasing_log_coeffs = dict()

    # Basic PN phasing coefficients (LALSimInspiralPNCoefficents)
    phasing_coeffs["0PN"] = 1.0
    phasing_coeffs["1PN"] = 0.0
    phasing_coeffs["2PN"] = 5.0 * (74.3 / 8.4 + 11.0 * eta) / 9.0
    phasing_coeffs["3PN"] = -16.0 * PI
    phasing_coeffs["4PN"] = (
        5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0
    )
    phasing_coeffs["5PN"] = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * PI
    phasing_log_coeffs["5PN"] = 5.0 / 3.0 * (772.9 / 8.4 - 13.0 * eta) * PI
    phasing_log_coeffs["6PN"] = -684.8 / 2.1
    phasing_coeffs["6PN"] = (
        11583.231236531 / 4.694215680
        - 640.0 / 3.0 * PI * PI
        - 684.8 / 2.1 * EulerGamma
        + eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * PI * PI)
        + eta * eta * 76.055 / 1.728
        - eta * eta * eta * 127.825 / 1.296
        + phasing_log_coeffs["6PN"] * jnp.log(4.0)
    )
    phasing_coeffs["7PN"] = PI * (
        770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta * eta
    )

    # Spin terms for phasing
    # Note: lal uses `spinL`` here, but no difference
    phasing_coeffs["7PN"] += get_7PNSOCoeff(m1M) * S1z + get_7PNSOCoeff(m2M) * S2z
    phasing_coeffs["6PN"] += (
        get_6PNSOCoeff(m1M) * S1z
        + get_6PNSOCoeff(m2M) * S2z
        + get_6PNS1S2OCoeff(eta) * S1z * S2z
        + (get_6PNQM2SCoeff(m1M) * qm_def1 + get_6PNSelf2SCoeff(m1M)) * S1z_2
        + (get_6PNQM2SCoeff(m2M) * qm_def2 + get_6PNSelf2SCoeff(m2M)) * S2z_2
    )

    phasing_coeffs["5PN"] += get_5PNSOCoeff(m1M) * S1z + get_5PNSOCoeff(m2M) * S2z
    phasing_log_coeffs["5PN"] += 3.0 * (
        get_5PNSOCoeff(m1M) * S1z + get_5PNSOCoeff(m2M) * S2z
    )

    phasing_coeffs["4PN"] += (
        get_4PNS1S2Coeff(eta) * S1z_dot_S2z
        + get_4PNS1S2OCoeff(eta) * S1z * S2z
        + (get_4PNQM2SOCoeff(m1M) * qm_def1 + get_4PNSelf2SOCoeff(m1M)) * S1z_2
        + (get_4PNQM2SOCoeff(m2M) * qm_def2 + get_4PNSelf2SOCoeff(m2M)) * S2z_2
        + (get_4PNQM2SCoeff(m1M) * qm_def1 + get_4PNSelf2SCoeff(m1M)) * S1z_2
        + (get_4PNQM2SCoeff(m2M) * qm_def2 + get_4PNSelf2SCoeff(m2M)) * S2z_2
    )

    phasing_coeffs["3PN"] += get_3PNSOCoeff(m1M) * S1z + get_3PNSOCoeff(m2M) * S2z

    # Tidal contributions
    phasing_coeffs["15PN"] = lambda1 * get_15PNTidalCoeff(
        m1M
    ) + lambda2 * get_15PNTidalCoeff(m2M)
    # Note, we are setting the 7.5PN contribution to zero, since this is done in LAL as well by default
    phasing_coeffs["15PN"] = 0
    phasing_coeffs["14PN"] = lambda1 * get_14PNTidalCoeff(
        m1M
    ) + lambda2 * get_14PNTidalCoeff(m2M)
    phasing_coeffs["13PN"] = lambda1 * get_13PNTidalCoeff(
        m1M
    ) + lambda2 * get_13PNTidalCoeff(m2M)
    phasing_coeffs["12PN"] = lambda1 * get_12PNTidalCoeff(
        m1M
    ) + lambda2 * get_12PNTidalCoeff(m2M)
    phasing_coeffs["10PN"] = lambda1 * get_10PNTidalCoeff(
        m1M
    ) + lambda2 * get_10PNTidalCoeff(m2M)

    # Multiply all at the end with the prefactor
    for key in phasing_coeffs.keys():
        phasing_coeffs[key] *= pfaN
    for key in phasing_log_coeffs.keys():
        phasing_log_coeffs[key] *= pfaN

    return phasing_coeffs, phasing_log_coeffs


def gen_TaylorF2(f: Array, params: Array, f_ref: float, use_lambda_tildes: bool = True):
    """
    Generate TaylorF2 frequency domain waveform

    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, lambda1, lambda1, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    lambda tilde: Dimensionless tidal deformability first object [between 0 and 5000]
    delta lamda tilde: Dimensionless tidal deformability second object [between 0 and 5000]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    f_ref: Reference frequency for the waveform

    Returns:
    --------
        h0 (array): Strain
    """
    # Lets make this easier by starting in Mchirp and eta space
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    if use_lambda_tildes:
        lambda1, lambda2 = lambda_tildes_to_lambdas(
            jnp.array([params[4], params[5], m1, m2])
        )
    else:
        lambda1, lambda2 = params[4], params[5]

    theta_intrinsic = jnp.array([m1, m2, params[2], params[3], lambda1, lambda2])
    theta_extrinsic = jnp.array([params[6], params[7], params[8]])

    h0 = _gen_TaylorF2(f, theta_intrinsic, theta_extrinsic, f_ref)

    return h0


def gen_TaylorF2_hphc(
    f: Array, params: Array, f_ref: float, use_lambda_tildes: bool = True
):
    """
    Generate PhenomD frequency domain waveform following 1508.07253.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    lambda1: Dimensionless tidal deformability of the primary object [between 0 and 5000]
    lambda2: Dimensionless tidal deformability of the secondary object [between 0 and 5000]
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
    h0 = gen_TaylorF2(f, params, f_ref, use_lambda_tildes=use_lambda_tildes)

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc


def _gen_TaylorF2(
    f: Array, theta_intrinsic: Array, theta_extrinsic: Array, f_ref: float
):
    """
    Generates the TaylorF2 waveform accoding to lal implementation.

    Note: internal units for mass are solar masses, as in LAL.

    Args:
        f (Array): Frequencies at which GW must be evaluated (Hz)
        theta_intrinsic (Array): Intrinsic parameters:
            component mass 1 [M_sun],
            component mass 2 [M_sun],
            spin 1 z-component,
            spin 2 z-component,
            dimensionless tidal deformability 1,
            dimensionless tidal deformability 2
        theta_extrinsic (Array): Extrinsic parameters:
            dist_mpc,
            tc,
            phic
        f_ref (float): Reference frequency

    Returns:
        Array: GW strain, evaluated at given frequencies
    """

    m1, m2, chi1, chi2, lambda1, lambda2 = theta_intrinsic
    dist_mpc, tc, phi_ref = theta_extrinsic
    m1_s = m1 * gt
    m2_s = m2 * gt
    M = m1 + m2
    M_s = (m1 + m2) * gt
    eta = m1_s * m2_s / (M_s**2.0)
    piM = PI * M_s

    # TODO: incorporate this into the waveform
    vISCO = 1.0 / jnp.sqrt(6.0)
    fISCO = vISCO * vISCO * vISCO / piM

    # Get the phasing coefficients
    phasing_coeffs, phasing_log_coeffs = get_PNPhasing_F2(
        m1, m2, chi1, chi2, lambda1, lambda2
    )

    # Coeffs
    pfa7 = phasing_coeffs["7PN"]
    pfa6 = phasing_coeffs["6PN"]
    pfa5 = phasing_coeffs["5PN"]
    pfa4 = phasing_coeffs["4PN"]
    pfa3 = phasing_coeffs["3PN"]
    pfa2 = phasing_coeffs["2PN"]
    pfa1 = phasing_coeffs["1PN"]
    pfaN = phasing_coeffs["0PN"]

    # Log coeffs
    pfl6 = phasing_log_coeffs["6PN"]
    pfl5 = phasing_log_coeffs["5PN"]

    pft15 = phasing_coeffs["15PN"]
    pft14 = phasing_coeffs["14PN"]
    pft13 = phasing_coeffs["13PN"]
    pft12 = phasing_coeffs["12PN"]
    pft10 = phasing_coeffs["10PN"]

    # Flux and energy coefficients
    FTaN = get_flux_0PNCoeff(eta)
    dETaN = 2.0 * get_energy_0PNCoeff(eta)

    r = dist_mpc * m_per_Mpc
    amp0 = -4.0 * m1 * m2 / r * MRSUN * gt * jnp.sqrt(PI / 12.0)

    ref_phasing = 0.0
    if f_ref != 0:
        vref = jnp.cbrt(piM * f_ref)
        logvref = jnp.log(vref)

        v2ref = vref * vref
        v3ref = vref * v2ref
        v4ref = vref * v3ref
        v5ref = vref * v4ref
        v6ref = vref * v5ref
        v7ref = vref * v6ref
        v8ref = vref * v7ref
        v9ref = vref * v8ref
        v10ref = vref * v9ref
        v12ref = v2ref * v10ref
        v13ref = vref * v12ref
        v14ref = vref * v13ref
        v15ref = vref * v14ref
        ref_phasing += pfa7 * v7ref
        ref_phasing += (pfa6 + pfl6 * logvref) * v6ref
        ref_phasing += (pfa5 + pfl5 * logvref) * v5ref
        ref_phasing += pfa4 * v4ref
        ref_phasing += pfa3 * v3ref
        ref_phasing += pfa2 * v2ref
        ref_phasing += pfa1 * vref
        ref_phasing += pfaN

        # Tidal terms in reference phasing
        ref_phasing += pft15 * v15ref
        ref_phasing += pft14 * v14ref
        ref_phasing += pft13 * v13ref
        ref_phasing += pft12 * v12ref
        ref_phasing += pft10 * v10ref

        ref_phasing /= v5ref

    # Build coefficients and factors
    v = jnp.cbrt(piM * f)
    logv = jnp.log(v)
    v2 = v * v
    v3 = v * v2
    v4 = v * v3
    v5 = v * v4
    v6 = v * v5
    v7 = v * v6
    v8 = v * v7
    v9 = v * v8
    v10 = v * v9
    v12 = v2 * v10
    v13 = v * v12
    v14 = v * v13
    v15 = v * v14

    # Build phasing
    phasing = 0.0
    phasing += pfa7 * v7
    phasing += (pfa6 + pfl6 * logv) * v6
    phasing += (pfa5 + pfl5 * logv) * v5
    phasing += pfa4 * v4
    phasing += pfa3 * v3
    phasing += pfa2 * v2
    phasing += pfa1 * v
    phasing += pfaN

    # Tidal terms in phasing
    phasing += pft15 * v15
    phasing += pft14 * v14
    phasing += pft13 * v13
    phasing += pft12 * v12
    phasing += pft10 * v10

    phasing /= v5
    flux = FTaN * v10
    dEnergy = dETaN * v

    shft = 2 * PI * tc
    phasing += shft * f - 2.0 * phi_ref - ref_phasing

    amp = amp0 * jnp.sqrt(-dEnergy / flux) * v

    # Assemble everything in final waveform
    h0 = amp * jnp.cos(phasing - PI / 4) - amp * jnp.sin(phasing - PI / 4) * 1.0j

    return h0
