# import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax
from math import pi
from waveform_constants import gt, EulerGamma
from PhenomD_utils import (
    get_transition_frequencies,
    get_coeffs,
    get_delta0,
    get_delta1,
    get_delta2,
    get_delta3,
    get_delta4,
)


def get_inspiral_phase(fM_s: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
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

    coeffs = get_coeffs(theta)

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
    phi5 = (1.0 + jnp.log(pi * fM_s)) * (
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
        - 6848.0 * jnp.log(64.0 * pi * fM_s) / 63.0
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
    TF2_pre = 3.0 * ((pi * fM_s) ** -(5.0 / 3.0)) / (128.0 * eta)
    phi_TF2 = TF2_pre * (
        phi0
        + phi1 * ((pi * fM_s) ** (1.0 / 3.0))
        + phi2 * ((pi * fM_s) ** (2.0 / 3.0))
        + phi3 * ((pi * fM_s) ** (3.0 / 3.0))
        + phi4 * ((pi * fM_s) ** (4.0 / 3.0))
        + phi5 * ((pi * fM_s) ** (5.0 / 3.0))
        + phi6 * ((pi * fM_s) ** (6.0 / 3.0))
        + phi7 * ((pi * fM_s) ** (7.0 / 3.0))
    )
    phi_Ins = (
        phi_TF2
        + (
            coeffs[7] * fM_s
            + (3.0 / 4.0) * coeffs[8] * (fM_s ** (4.0 / 3.0))
            + (3.0 / 5.0) * coeffs[9] * (fM_s ** (5.0 / 3.0))
            + (1.0 / 2.0) * coeffs[10] * (fM_s ** 2.0)
        )
        / eta
    )
    return phi_Ins


def get_IIa_raw_phase(fM_s: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    coeffs = get_coeffs(theta)

    phi_IIa_raw = (
        coeffs[11] * fM_s
        + coeffs[12] * jnp.log(fM_s)
        - coeffs[13] * (fM_s ** -3.0) / 3.0
    ) / eta

    return phi_IIa_raw


def get_IIb_raw_phase(fM_s: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    coeffs = get_coeffs(theta)
    _, _, _, _, f_RD, f_damp = get_transition_frequencies(theta, coeffs[5], coeffs[6])
    f_RDM_s = f_RD * M_s
    f_dampM_s = f_damp * M_s

    phi_IIb_raw = (
        coeffs[14] * fM_s
        - coeffs[15] * (fM_s ** -1.0)
        + 4.0 * coeffs[16] * (fM_s ** (3.0 / 4.0)) / 3.0
        + coeffs[17] * jnp.arctan((fM_s - coeffs[18] * f_RDM_s) / f_dampM_s)
    ) / eta

    return phi_IIb_raw


def get_Amp0(fM_s: jnp.ndarray, eta: jnp.float64) -> jnp.ndarray:
    Amp0 = (
        eta ** (1.0 / 2.0)
        * (fM_s) ** (-7.0 / 6.0)
        * (2.0 / 3.0) ** (1.0 / 2.0)
        * pi ** (-1.0 / 6.0)
    )
    return Amp0


def get_inspiral_Amp(fM_s: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)
    delta = (m1_s - m2_s) / M_s

    # Spin variables
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0

    # And the coefficients
    coeffs = get_coeffs(theta)

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

    Amp_PN = (
        A0
        + A1 * ((pi * fM_s) ** (1.0 / 3.0))
        + A2 * ((pi * fM_s) ** (2.0 / 3.0))
        + A3 * ((pi * fM_s) ** (3.0 / 3.0))
        + A4 * ((pi * fM_s) ** (4.0 / 3.0))
        + A5 * ((pi * fM_s) ** (5.0 / 3.0))
        + A6 * ((pi * fM_s) ** (6.0 / 3.0))
    )

    Amp_Ins = Amp_PN + (
        +coeffs[0] * (fM_s ** (7.0 / 3.0))
        + coeffs[1] * (fM_s ** (8.0 / 3.0))
        + coeffs[2] * (fM_s ** (9.0 / 3.0))
    )
    return Amp_Ins


def get_IIa_Amp(fM_s: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s

    # And the coefficients
    coeffs = get_coeffs(theta)

    # Frequency breaks
    _, _, f1, f3, _, _ = get_transition_frequencies(theta, coeffs[5], coeffs[6])
    f2 = (f1 + f3) / 2

    # For this region, we also need to calculate the the values and derivatives
    # of the Ins and IIb regions
    v1, d1 = jax.value_and_grad(get_inspiral_Amp)(f1 * M_s, theta)
    v3, d3 = jax.value_and_grad(get_IIb_Amp)(f3 * M_s, theta)

    # Here we need the delta solutions
    delta0 = get_delta0(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta1 = get_delta1(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta2 = get_delta2(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta3 = get_delta3(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta4 = get_delta4(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)

    Amp_IIa = (
        delta0
        + delta1 * fM_s
        + delta2 * (fM_s ** 2.0)
        + delta3 * (fM_s ** 3.0)
        + delta4 * (fM_s ** 4.0)
    )

    return Amp_IIa


def get_IIb_Amp(fM_s: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    m1, m2, _, _ = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s

    # And the coefficients
    coeffs = get_coeffs(theta)

    # Frequency breaks
    _, _, _, _, f_RD, f_damp = get_transition_frequencies(theta, coeffs[5], coeffs[6])

    Amp_IIb = (
        coeffs[4]
        * coeffs[6]
        * (f_damp * M_s)
        / ((fM_s - (f_RD * M_s)) ** 2.0 + (coeffs[6] * (f_damp * M_s)) ** 2)
    ) * jnp.exp(-coeffs[5] * (fM_s - (f_RD * M_s)) / coeffs[6] / (f_damp * M_s))
    return Amp_IIb


def Phase(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
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

    coeffs = get_coeffs(theta)

    # Next we need to calculate the transition frequencies
    f1, f2, _, _, f_RD, f_damp = get_transition_frequencies(theta, coeffs[5], coeffs[6])

    phi_Ins = get_inspiral_phase(f * M_s, theta)

    # Next lets construct the phase of the late inspiral (region IIa)
    # beta0 is found by matching the phase between the region I and IIa
    # C(1) continuity must be preserved. We therefore need to solve for an additional
    # contribution to beta1
    # Note that derivatives seem to be d/d(fM_s), not d/df

    # Here I've now defined
    # phi_IIa(f1*M_s) + beta0 + beta1_correction*(f1*M_s) = phi_Ins(f1)
    # ==> phi_IIa'(f1*M_s) + beta1_correction = phi_Ins'(f1*M_s)
    # ==> beta1_correction = phi_Ins'(f1*M_s) - phi_IIa'(f1*M_s)
    # ==> beta0 = phi_Ins(f1*M_s) - phi_IIa(f1*M_s) - beta1_correction*(f1*M_s)
    phi_Ins_f1, dphi_Ins_f1 = jax.value_and_grad(get_inspiral_phase)(f1 * M_s, theta)
    phi_IIa_f1, dphi_IIa_f1 = jax.value_and_grad(get_IIa_raw_phase)(f1 * M_s, theta)

    beta1_correction = dphi_Ins_f1 - dphi_IIa_f1
    beta0 = phi_Ins_f1 - beta1_correction * (f1 * M_s) - phi_IIa_f1

    phi_IIa_func = (
        lambda fM_s: get_IIa_raw_phase(fM_s, theta) + beta0 + (beta1_correction * fM_s)
    )
    phi_IIa = phi_IIa_func(f * M_s)

    # And finally, we do the same thing to get the phase of the merger-ringdown (region IIb)
    # phi_IIb(f2*M_s) + a0 + a1_correction*(f2*M_s) = phi_IIa(f2*M_s)
    # ==> phi_IIb'(f2*M_s) + a1_correction = phi_IIa'(f2*M_s)
    # ==> a1_correction = phi_IIa'(f2*M_s) - phi_IIb'(f2*M_s)
    # ==> a0 = phi_IIa(f2*M_s) - phi_IIb(f2*M_s) - beta1_correction*(f2*M_s)
    phi_IIa_f2, dphi_IIa_f2 = jax.value_and_grad(phi_IIa_func)(f2 * M_s)
    phi_IIb_f2, dphi_IIb_f2 = jax.value_and_grad(get_IIb_raw_phase)(f2 * M_s, theta)

    a1_correction = dphi_IIa_f2 - dphi_IIb_f2
    a0 = phi_IIa_f2 - a1_correction * (f2 * M_s) - phi_IIb_f2

    phi_IIb = get_IIb_raw_phase(f * M_s, theta) + a0 + a1_correction * (f * M_s)

    # And now we can combine them by multiplying by a set of heaviside functions
    phase = (
        phi_Ins * jnp.heaviside(f1 - f, 0.5)
        + jnp.heaviside(f - f1, 0.5) * phi_IIa * jnp.heaviside(f2 - f, 0.5)
        + phi_IIb * jnp.heaviside(f - f2, 0.5)
    )

    return phase


def Amp(f: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
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
    eta = m1_s * m2_s / (M_s ** 2.0)

    coeffs = get_coeffs(theta)

    _, _, f3, f4, _, _ = get_transition_frequencies(theta, coeffs[5], coeffs[6])

    # First we get the inspiral amplitude
    Amp_Ins = get_inspiral_Amp(f * M_s, theta)

    # Next lets construct the phase of the late inspiral (region IIa)
    # Note that this part is a little harder since we need to solve a system of equations for deltas
    Amp_IIa = get_IIa_Amp(f * M_s, theta)

    # And finally, we construct the phase of the merger-ringdown (region IIb)
    Amp_IIb = get_IIb_Amp(f * M_s, theta)

    # And now we can combine them by multiplying by a set of heaviside functions
    Amp = (
        Amp_Ins * jnp.heaviside(f3 - f, 0.5)
        + jnp.heaviside(f - f3, 0.5) * Amp_IIa * jnp.heaviside(f4 - f, 0.5)
        + Amp_IIb * jnp.heaviside(f - f4, 0.5)
    )

    # Prefactor
    Amp0 = get_Amp0(f * M_s, eta)

    # Need to add in an overall scaling of M_s^2 to make the units correct
    return Amp0 * Amp * (M_s ** 2.0)