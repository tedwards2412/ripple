import numpy as np
from math import pi
from waveform_constants import gt, EulerGamma, PhenomD_coeff_table

# All these equations are defined in the papers
# Taken from https://github.com/scottperkins/phenompy/blob/master/utilities.py
def get_fRD_fdamp(m1, m2, chi1, chi2):
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s ** 2.0)
    S = (
        chi1 * m1_s ** 2 + chi2 * m2_s ** 2
    ) / M_s ** 2  # convert chi to spin s in z direction
    S_red = S / (1 - 2 * eta_s)

    a = (
        S
        + 2 * np.sqrt(3) * eta_s
        - 4.399 * eta_s ** 2
        + 9.397 * eta_s ** 3
        - 13.181 * eta_s ** 4
        + (-0.085 * S + 0.102 * S ** 2 - 1.355 * S ** 3 - 0.868 * S ** 4) * eta_s
        + (-5.837 * S - 2.097 * S ** 2 + 4.109 * S ** 3 + 2.064 * S ** 4) * eta_s ** 2
    )

    E_rad_ns = (
        0.0559745 * eta_s
        + 0.580951 * eta_s ** 2
        - 0.960673 * eta_s ** 3
        + 3.35241 * eta_s ** 4
    )

    E_rad = (
        E_rad_ns
        * (1 + S_red * (-0.00303023 - 2.00661 * eta_s + 7.70506 * eta_s ** 2))
        / (1 + S_red * (-0.67144 - 1.47569 * eta_s + 7.30468 * eta_s ** 2))
    )

    MWRD = 1.5251 - 1.1568 * (1 - a) ** 0.1292
    fRD = (1 / (2 * np.pi)) * (MWRD) / (M_s * (1 - E_rad))

    MWdamp = (1.5251 - 1.1568 * (1 - a) ** 0.1292) / (
        2 * (0.700 + 1.4187 * (1 - a) ** (-0.4990))
    )
    fdamp = (1 / (2 * np.pi)) * (MWdamp) / (M_s * (1 - E_rad))
    return fRD, fdamp


def get_transition_frequencies(
    theta: np.ndarray, gamma2: float, gamma3: float
) -> np.ndarray:

    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    f_RD, f_damp = get_fRD_fdamp(m1, m2, chi1, chi2)

    # Phase transition frequencies
    # FIXME: Need to work out what is happening with the units here
    f1 = 0.018 / (M * gt)
    f2 = f_RD / 2

    # Amplitude transition frequencies
    f3 = 0.018 / (M * gt)
    f4 = abs(f_RD + f_damp * gamma3 * (np.sqrt(1 - (gamma2 ** 2))) / gamma2)

    return f1, f2, f3, f4, f_RD, f_damp


def get_coeffs(theta: np.ndarray) -> np.ndarray:
    # Retrives the coefficients needed to produce the waveform

    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    eta = m1 * m2 / (M ** 2.0)

    chi_eff = (m1 * chi1 + m2 * chi2) / M
    chiPN = chi_eff - (38.0 * eta / 113.0) * (chi1 + chi2)

    def calc_coeff(i, eta, chiPN):
        coeff = (
            PhenomD_coeff_table[i, 0]
            + PhenomD_coeff_table[i, 1] * eta
            + (chiPN - 1.0)
            * (
                PhenomD_coeff_table[i, 2]
                + PhenomD_coeff_table[i, 3] * eta
                + PhenomD_coeff_table[i, 4] * (eta ** 2.0)
            )
            + (chiPN - 1.0) ** 2.0
            * (
                PhenomD_coeff_table[i, 5]
                + PhenomD_coeff_table[i, 6] * eta
                + PhenomD_coeff_table[i, 7] * (eta ** 2.0)
            )
            + (chiPN - 1) ** 3.0
            * (
                PhenomD_coeff_table[i, 8]
                + PhenomD_coeff_table[i, 9] * eta
                + PhenomD_coeff_table[i, 10] * (eta ** 2.0)
            )
        )

        return coeff

    rho1 = calc_coeff(0, eta, chiPN)
    rho2 = calc_coeff(1, eta, chiPN)
    rho3 = calc_coeff(2, eta, chiPN)
    v2 = calc_coeff(3, eta, chiPN)
    gamma1 = calc_coeff(4, eta, chiPN)
    gamma2 = calc_coeff(5, eta, chiPN)
    gamma3 = calc_coeff(6, eta, chiPN)
    sig1 = calc_coeff(7, eta, chiPN)
    sig2 = calc_coeff(8, eta, chiPN)
    sig3 = calc_coeff(9, eta, chiPN)
    sig4 = calc_coeff(10, eta, chiPN)
    beta1 = calc_coeff(11, eta, chiPN)
    beta2 = calc_coeff(12, eta, chiPN)
    beta3 = calc_coeff(13, eta, chiPN)
    a1 = calc_coeff(14, eta, chiPN)
    a2 = calc_coeff(15, eta, chiPN)
    a3 = calc_coeff(16, eta, chiPN)
    a4 = calc_coeff(17, eta, chiPN)
    a5 = calc_coeff(18, eta, chiPN)

    coeffs = np.array(
        [
            rho1,
            rho2,
            rho3,
            v2,
            gamma1,
            gamma2,
            gamma3,
            sig1,
            sig2,
            sig3,
            sig4,
            beta1,
            beta2,
            beta3,
            a1,
            a2,
            a3,
            a4,
            a5,
        ]
    )

    return coeffs


def Phase(f: np.ndarray, theta: np.ndarray) -> np.ndarray:
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
    eta = m1_s * m2_s / (M_s ** 2.0)
    delta = (m1_s - m2_s) / M_s

    # Spin variable
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    chi_eff = (m1_s * chi1 + m2_s * chi2) / M_s
    chiPN = chi_eff - (38.0 * eta / 113.0) * (chi1 + chi2)

    coeffs = get_coeffs(theta)
    print("Here are the coeffs", coeffs)

    # Next we need to calculate the transition frequencies
    f1, f2, _, _, f_RD, f_damp = get_transition_frequencies(theta, coeffs[5], coeffs[6])

    # First lets construct the phase in the inspiral (region I)
    phi0 = 1.0
    phi1 = 0.0
    phi2 = 3715.0 / 756.0 + 55.0 * eta / 9.0
    phi3 = (
        -16.0 * pi
        + 113.0 * delta * chi_a / 3.0
        + (113.0 / 3.0 - 76.0 * eta / 3.0) * chi_s
    )
    phi4 = (
        15293365.0 / 508032.0
        + 27145.0 * eta / 504.0
        + 3085.0 * (eta ** 2.0) / 72.0
        + (-405.0 / 8.0 + 200.0 * eta) * (chi_a ** 2.0)
        - 405.0 * delta * chi_a * chi_s / 4.0
        + (-405.0 / 8.0 + 5.0 * eta / 2.0) * (chi_s ** 2.0)
    )
    phi5 = (1.0 + np.log(pi * M_s * f)) * (
        38645.0 * pi / 756.0
        - 64.0 * pi * eta / 9.0
        + delta * (-732985.0 / 2268.0 - 140.0 * eta / 9.0) * chi_a
        + (732985.0 / 2268.0 + 24260.0 * eta / 81.0 + 340.0 * (eta ** 2.0) / 9.0)
        * chi_s
    )
    phi6 = (
        11583231236531.0 / 4694215680.0
        - 6848.0 * EulerGamma / 21.0
        - 640.0 * (pi ** 2.0) / 3.0
        + (-15737765635.0 / 3048192.0 + 2255.0 * (pi ** 2.0) / 12.0) * eta
        + 76055.0 * (eta ** 2.0) / 1728.0
        - 127825.0 * (eta ** 3.0) / 1296.0
        - 6848.0 * np.log(64.0 * pi * M_s * f) / 63.0
        + 2270.0 * pi * delta * chi_a / 3.0
        + (2270.0 * pi / 3.0 - 520.0 * pi * eta) * chi_s
    )
    phi7 = (
        77096675.0 * pi / 254016.0
        + 378515.0 * pi * eta / 1512.0
        - 74045.0 * pi * (eta ** 2.0) / 756.0
        + delta
        * (
            -25150083775.0 / 3048192.0
            + 26804935.0 * eta / 6048.0
            - 1985.0 * (eta ** 2.0) / 48.0
        )
        * chi_a
        + (
            -25150083775.0 / 3048192.0
            + 10566655595.0 * eta / 762048.0
            - 1042165.0 * (eta ** 2.0) / 3024.0
            + 5345.0 * (eta ** 3.0) / 36.0
        )
        * chi_s
    )

    # Add frequency dependence here
    TF2_pre = 3.0 * ((pi * f * M_s) ** -(5.0 / 3.0)) / (128.0 * eta)
    phi_TF2 = TF2_pre * (
        phi0
        + phi1 * ((pi * f * M_s) ** (1.0 / 3.0))
        + phi2 * ((pi * f * M_s) ** (2.0 / 3.0))
        + phi3 * ((pi * f * M_s) ** (3.0 / 3.0))
        + phi4 * ((pi * f * M_s) ** (4.0 / 3.0))
        + phi5 * ((pi * f * M_s) ** (5.0 / 3.0))
        + phi6 * ((pi * f * M_s) ** (6.0 / 3.0))
        + phi7 * ((pi * f * M_s) ** (7.0 / 3.0))
    )
    phi_Ins = (
        phi_TF2
        + (
            coeffs[7] * (pi * f * M_s)
            + (3.0 / 4.0) * coeffs[8] * ((pi * f * M_s) ** (4.0 / 3.0))
            + (3.0 / 5.0) * coeffs[9] * ((pi * f * M_s) ** (5.0 / 3.0))
            + (1.0 / 2.0) * coeffs[10] * ((pi * f * M_s) ** 2.0)
        )
        / eta
    )

    # Next lets construct the phase of the late inspiral (region IIa)
    # Sig0 is found by matching the phase between the region I and IIa
    # FIXME: Is there a better way to do the matching? This is probably where the error is beta ==
    # beta0 = eta * np.interp(f1, f, phi_Ins) - (
    #     coeffs[11] * (f1 * M * gt)
    #     + coeffs[12] * np.log((f1 * M * gt))
    #     - coeffs[13] * ((f1 * M * gt) ** -3.0) / 3.0
    # )
    # phi_IIa = (
    #     beta0
    #     + (
    #         coeffs[11] * (f * M * gt)
    #         + coeffs[12] * np.log((f * M * gt))
    #         - coeffs[13] * ((f * M * gt) ** -3.0) / 3.0
    #     )
    # ) / eta

    # # # And finally, we construct the phase of the merger-ringdown (region IIb)
    # sig0 = eta * np.interp(f2, f, phi_IIa) - (
    #     coeffs[14] * (f * M * gt)
    #     - coeffs[15] * ((f * M * gt) ** -1.0)
    #     + 4.0 * coeffs[16] * ((f * M * gt) ** (3.0 / 4.0)) / 3.0
    #     + coeffs[17] * np.arctan((f - coeffs[18] * f_RD) / f_damp)
    # )
    # phi_IIb = (
    #     sig0
    #     + (
    #         coeffs[14] * (f * M * gt)
    #         - coeffs[15] * ((f * M * gt) ** -1.0)
    #         + 4.0 * coeffs[16] * ((f * M * gt) ** (3.0 / 4.0)) / 3.0
    #         + coeffs[17] * np.arctan((f - coeffs[18] * f_RD) / f_damp)
    #     )
    # ) / eta

    # # And now we can combine them by multiplying by a set of heaviside functions
    # print("Within function Ins", phi_Ins * np.heaviside(f1 - f, 0.5))
    # print(
    #     "Within function IIa",
    #     np.heaviside(f - f1, 0.5) * phi_IIa * np.heaviside(f2 - f, 0.5),
    # )
    # print("Within function IIb", phi_IIb * np.heaviside(f - f2, 0.5))
    # print("Within function", f1, f2, f_RD, f_damp)
    phase = (
        phi_Ins
        * np.heaviside(f1 - f, 0.5)
        # + np.heaviside(f - f1, 0.5) * phi_IIa * np.heaviside(f2 - f, 0.5)
        # + phi_IIb * np.heaviside(f - f2, 0.5)
    )
    # phase = phi_TF2

    return phase, f1, f2


def Amp(f: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes the amplitude of the PhenomD frequency domain waveform following 1508.07253.
    Note that this waveform also assumes that object one is the more massive.
    Therefore the more massive object is always considered a BH

    Returns:
    --------
      Amplitude (array):
    """

    # First lets calculate some of the vairables that will be used below
    # Mass variables
    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    eta = m1 * m2 / (M ** 2.0)
    delta = (m1 - m2) / M

    # Spin variables
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    chi_eff = (m1 * chi1 + m2 * chi2) / M
    chiPN = chi_eff - (38.0 * eta / 113.0) * (chi1 + chi2)

    coeffs = get_coeffs(theta)

    _, _, f3, f4, f_RD, f_damp = get_transition_frequencies(theta, coeffs[5], coeffs[6])

    ### Need to check units here
    Amp0 = (
        eta ** (1.0 / 2.0)(f + 1e-100) ** (-7.0 / 6.0)
        * (2.0 / 3.0) ** (1.0 / 2.0)
        * pi ** (3.0 / 2.0)
        * np.sqrt(5.0 / 24.0)
    )

    # First lets construct the Amplitude in the inspiral (region I)
    A0 = 1.0
    A1 = 0.0
    A2 = -323.0 / 224.0 + 451.0 * eta / 168.0
    A3 = 27.0 * delta * chi_a / 8.0 + (27.0 / 8.0 - 11.0 * eta / 6.0) * chi_s
    A4 = (
        -27312085.0 / 8128512.0
        - 1975055.0 * eta / 338688.0
        + 105271.0 * (eta ** 2.0) / 24192.0
        + (-81.0 / 32.0 + 8.0 * eta) * (chi_a ** 2.0)
        - 81.0 / 16.0 * delta * chi_a * chi_s
        + (-81.0 / 32.0 + 17.0 * eta / 8.0) * (chi_s ** 2.0)
    )
    A5 = (
        -85.0 * pi / 64.0
        + 85.0 * pi * eta / 16.0
        + delta * (285197.0 / 16128.0 - 1579.0 * eta / 4032.0) * chi_a
        + (285197.0 / 16128.0 - 15317.0 * eta / 672.0 - 2227.0 * (eta ** 2.0) / 1008.0)
        * chi_s
    )
    A6 = (
        -177520268561.0 / 8583708672.0
        + (545384828789.0 / 5007163392.0 - 205.0 * (pi ** 2.0) / 48.0) * eta
        - 3248849057.0 * (eta ** 2.0) / 178827264.0
        + 34473079.0 * (eta ** 3.0) / 6386688.0
        + (
            1614569.0 / 64512.0
            - 1873643.0 * eta / 16128.0
            + 2167.0 * (eta ** 2.0) / 42.0
        )
        * (chi_a ** 2.0)
        + (31.0 * pi / 12.0 - 8.0 * pi * eta / 3.0) * chi_s
        + (
            1614569.0 / 64512.0
            - 61391.0 * eta / 1344.0
            + 57451.0 * (eta ** 2.0) / 4032.0
        )
        * (chi_s ** 2.0)
        + delta
        * chi_a
        * (31.0 * pi / 12.0 + (1614569.0 / 32256.0 - 165961.0 * eta / 2688.0) * chi_s)
    )

    Amp_PN = Amp0 * (
        A0
        + A1 * ((pi * f) ** (1.0 / 3.0))
        + A2 * ((pi * f) ** (2.0 / 3.0))
        + A3 * ((pi * f) ** (3.0 / 3.0))
        + A4 * ((pi * f) ** (4.0 / 3.0))
        + A5 * ((pi * f) ** (5.0 / 3.0))
        + A6 * ((pi * f) ** (6.0 / 3.0))
    )

    Amp_Ins = Amp_PN + Amp0 * (
        +coeffs[0] * ((pi * f) ** (7.0 / 3.0))
        + coeffs[1] * ((pi * f) ** (8.0 / 3.0))
        + coeffs[2] * ((pi * f) ** (9.0 / 3.0))
    )

    # Next lets construct the phase of the late inspiral (region IIa)
    # This part is a little harder since we need to solve a system of equations for deltas
    Amp_IIa = 1.0

    # And finally, we construct the phase of the merger-ringdown (region IIb)
    Amp_IIb = Amp0 * (
        (
            coeffs[4]
            * coeffs[6]
            * f_damp
            / ((f - f_RD) ** 2.0 + (coeffs[6] * f_damp) ** 2)
        )
        * np.exp(-coeffs[5] * (f - f_RD) / coeffs[6] / f_damp)
    )

    # And now we can combine them by multiplying by a set of heaviside functions
    Amp = (
        Amp_Ins * np.heaviside(f3 - f, 0.5)
        + np.heaviside(f - f3, 0.5) * Amp_IIa * np.heaviside(f - f4, 0.5)
        + Amp_IIb * np.heaviside(f4 - f, 0.5)
    )

    pre = 3.6686934875530996e-19  # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec

    return Amp