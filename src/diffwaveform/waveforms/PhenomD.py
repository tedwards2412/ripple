import numpy as np
from math import pi
from waveform_constants import gt, EulerGamma, PhenomD_coeff_table


def get_coeffs(theta: np.ndarray) -> np.ndarray:
    # This

    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    eta = m1 * m2 / (M ** 2)

    chi_eff = (m1 * chi1 + m2 * chi2) / M
    chiPN = chi_eff - (38 * eta / 113) * (chi1 + chi2)

    def calc_coeff(i, eta, chiPN):
        coeff = (
            PhenomD_coeff_table[i, 0]
            + PhenomD_coeff_table[i, 1]
            * eta
            * (chiPN - 1)
            * (
                PhenomD_coeff_table[i, 2]
                + PhenomD_coeff_table[i, 3] * eta
                + PhenomD_coeff_table[i, 4] * eta ** 2
            )
            + (chiPN - 1) ** 2
            * (
                PhenomD_coeff_table[i, 5]
                + PhenomD_coeff_table[i, 6] * eta
                + PhenomD_coeff_table[i, 7] * eta ** 2
            )
            + (chiPN - 1) ** 3
            * (
                PhenomD_coeff_table[i, 8]
                + PhenomD_coeff_table[i, 9] * eta
                + PhenomD_coeff_table[i, 10] * eta ** 2
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


def Psi(f: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes the phase of the PhenomD waveform following 1508.07253.
    Sets time and phase of coealence to be zero.

    Returns:
    --------
        phase (array): Phase of the GW as a function of frequency
    """

    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    eta = m1 * m2 / (M ** 2)

    chi_eff = (m1 * chi1 + m2 * chi2) / M
    chiPN = chi_eff - (38 * eta / 113) * (chi1 + chi2)

    coeffs = PhenomD.get_coeffs(theta)

    # First lets construct the phase in the inspiral (region I)
    phi0 = 1

    phi_TF2 = phi0 + phi1 + phi2 + phi3 + phi4 + phi5 + phi6 + phi7
    phi_Ins = (
        phi_TF2
        + (
            coeffs[7] * f
            + (3.0 / 4.0) * coeffs[8] * (f ** (4.0 / 3.0))
            + (3.0 / 5.0) * coeffs[9] * (f ** (5.0 / 3.0))
            + (1.0 / 2.0) * coeffs[10] * (f ** 2.0)
        )
        / eta
    )

    # Next lets construct the phase of the late inspiral (region IIa)

    # And finally, we construct the phase of the merger-ringdown (region IIb)

    return


def Amp(f: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes the amplitude of the PhenomD frequency domain waveform following 1508.07253.
    Note that this waveform also assumes that object one is the more massive.
    Therefore the more massive object is always considered a BH

    Returns:
    --------
      Amplitude (array):
    """

    m1, m2, chi1, chi2 = theta

    M = m1 + m2
    eta = m1 * m2 / (M ** 2)
    pre = 3.6686934875530996e-19  # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
    ### Need to check units here
    A0 = (
        eta ** (1 / 2)(f + 1e-100) ** (-7.0 / 6.0)
        * (2.0 / 3.0) ** (1.0 / 2.0)
        * pi ** (3.0 / 2.0)
        * np.sqrt(5.0 / 24.0)
    )

    return pre * A0