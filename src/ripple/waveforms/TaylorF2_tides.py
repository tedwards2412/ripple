# from math import pi

import jax
import jax.numpy as jnp

from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ..typing import Array
from ripple import Mc_eta_to_ms


def Phif3hPN_TLN(f, theta):
    """
    Computes the phase of the TaylorF2 waveform + TLN parameters. Sets time and phase of coealence to be zero.
    Parameters:
    f (array): Frequency at which to output the phase [Hz]
    M (float): Total Mass [Msun]
    eta (float): m1*m2/(m1+m2)**2
    chi_1 (float): Spin of object 1 in z direction
    chi_2 (float): Spin of object 2 in z direction
    Love1 (float): Love number of object 1
    Love2 (float): Love number of object 2

    Returns:
    phase (array): Phase of the GW as a function of frequency
    """
    M, eta, chi_1, chi_2, Love1, Love2 = theta
    vlso = 1.0 / jnp.sqrt(6.0)
    lambda_1 = 1.0
    lambda_2 = 1.0
    kappa_1 = 1.0
    kappa_2 = 1.0

    chi_s = 0.5 * (chi_1 + chi_2)
    chi_a = 0.5 * (chi_1 - chi_2)
    k_s = 0.5 * (kappa_1 + kappa_2)
    k_a = 0.5 * (kappa_1 - kappa_2)

    lambda_s = 0.5 * (lambda_1 + lambda_2)
    lambda_a = 0.5 * (lambda_1 - lambda_2)
    delta = jnp.sqrt(1.0 - 4.0 * eta)

    m1 = (M + jnp.sqrt(M**2 - 4 * (eta * M**2))) / 2
    m2 = (M - jnp.sqrt(M**2 - 4 * (eta * M**2))) / 2

    v = (PI * M * (f + 1e-100) * gt) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    v10 = v5 * v5
    v12 = v10 * v2
    eta2 = eta**2
    eta3 = eta**3

    ## ------------------------- Non spinning part of the waveform
    ## Background GR
    psi_NS_0PN = 1.0
    psi_NS_1PN = (3715.0 / 756.0 + 55.0 * eta / 9.0) * v2
    psi_NS_15PN = -16.0 * PI * v3
    psi_NS_2PN = (
        15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta2 / 72.0
    ) * v4
    psi_NS_25PN = (
        PI * (38645.0 / 756.0 - 65.0 * eta / 9.0) * (1 + 3.0 * jnp.log(v / vlso)) * v5
    )
    psi_NS_3PN = (
        (
            11583231236531.0 / 4694215680.0
            - 640.0 * PI**2 / 3.0
            - 6848.0 * EulerGamma / 21.0
        )
        + (2255.0 * PI**2 / 12.0 - 15737765635.0 / 3048192.0) * eta
        + 76055.0 * eta2 / 1728.0
        - 127825.0 * eta3 / 1296.0
        - 6848.0 * jnp.log(4.0 * v) / 21.0
    ) * v6
    psi_NS_35PN = (
        PI
        * (77096675.0 / 254016.0 + 378515.0 * eta / 1512.0 - 74045.0 * eta2 / 756.0)
        * v7
    )

    ## Tidal Love Numbers
    Lambda_t = (
        16.0
        * ((m1 + 12 * m2) * m1**4 * Love1 + (m2 + 12 * m1) * m2**4 * Love2)
        / (13.0 * M**5.0)
    )

    psi_TLN_5PN = -(39.0 * Lambda_t / 2.0) * v10
    # psi_TLN_6PN = (-3115.0 * Lambda_t / 64.0 + 6595.0 * delta_Lambda_t / 364.0) * v12

    ## ------------------------- Spining part of the waveform (aligned spins)
    psi_S_15PN = (
        (113.0 / 3.0 - 76.0 * eta / 3.0) * chi_s + 113.0 * delta * chi_a / 3.0
    ) * v3

    psi_S_2PN = (
        -(5.0 / 8.0)
        * (1.0 + 156.0 * eta + 80.0 * delta * k_a + 80.0 * (1.0 - 2.0 * eta) * k_s)
        * (chi_s**2)
    )
    psi_S_2PN -= (
        (5.0 / 8.0)
        * (1.0 - 160.0 * eta + 80.0 * delta * k_a + 80.0 * (1.0 - 2.0 * eta) * k_s)
        * (chi_a**2)
    )
    psi_S_2PN -= (
        (5.0 / 4.0)
        * (delta + 80.0 * delta * k_s + 80.0 * (1.0 - 2.0 * eta) * k_a)
        * chi_s
        * chi_a
    )
    psi_S_2PN *= v4

    psi_S_25PN = (
        -(732985.0 / 2268.0 - 24260.0 * eta / 81.0 - 340.0 * eta2 / 9.0) * chi_s
        - (732985.0 / 2268.0 + 140.0 * eta / 9.0) * delta * chi_a
    ) * v5
    psi_S_25PN_log = 3.0 * psi_S_25PN * jnp.log(v / vlso)

    psi_S_3PN = (2270.0 / 3.0 - 520.0 * eta) * PI * chi_s + (
        2270.0 * PI / 3.0
    ) * delta * chi_a
    psi_S_3PN += (
        (
            (26015.0 / 14.0 - 88510.0 * eta / 21.0 - 480.0 * eta2) * k_a
            + delta
            * (
                -1344475.0 / 1008.0
                + 745.0 * eta / 18.0
                + (26015.0 / 14.0 - 1495.0 * eta / 3.0) * k_s
            )
        )
        * chi_s
        * chi_a
    )
    psi_S_3PN += (
        -1344475.0 / 2016.0
        + 829705.0 * eta / 504.0
        + 3415.0 * eta2 / 9.0
        + (26015.0 / 28.0 - 44255.0 * eta / 21.0 - 240.0 * eta2) * k_s
        + delta * (26015.0 / 28.0 - 1495.0 * eta / 6.0) * k_a
    ) * (chi_s) ** 2
    psi_S_3PN += (
        -1344475.0 / 2016.0
        + 267815.0 * eta / 252.0
        - 240.0 * eta2
        + (26015.0 / 28.0 - 44255.0 * eta / 21.0 - 240.0 * eta2) * k_s
        + delta * (26015.0 / 28.0 - 1495.0 * eta / 6.0) * k_a
    ) * (chi_a) ** 2
    psi_S_3PN *= v6

    psi_S_35PN = (
        -25150083775.0 / 3048192.0
        + 10566655595.0 * eta / 762048.0
        - 1042165 * eta2 / 3024.0
        + 5345.0 * eta3 / 36.0
    ) * chi_s
    psi_S_35PN += (
        (-25150083775.0 / 3048192.0 + 26804935.0 * eta / 6048.0 - 1985.0 * eta2 / 48.0)
        * delta
        * chi_a
    )
    psi_S_35PN += (
        265.0 / 24.0
        + 4035.0 * eta / 2.0
        - 20.0 * eta2 / 3.0
        + (3110.0 / 3.0 - 10250.0 * eta / 3.0 + 40.0 * eta2) * k_s
        - 440.0 * (1.0 - 3.0 * eta) * lambda_s
        + delta
        * ((3110.0 / 3.0 - 4030.0 * eta / 3.0) * k_a - 440.0 * (1.0 - eta) * lambda_a)
    ) * (chi_s) ** 3
    psi_S_35PN += (
        (3110.0 / 3.0 - 8470.0 * eta / 3.0) * k_a
        - 440.0 * (1.0 - 3.0 * eta) * lambda_a
        + delta
        * (
            265.0 / 24.0
            - 2070.0 * eta
            + (3110.0 / 3.0 - 750.0 * eta) * k_s
            - 440.0 * (1 - eta) * lambda_s
        )
    ) * (chi_a) ** 3
    psi_S_35PN += (
        (3110.0 - 28970.0 * eta / 3.0 + 80.0 * eta2) * k_a
        - 1320.0 * (1.0 - 3.0 * eta) * lambda_a
        + delta
        * (
            265.0 / 8.0
            + 12055.0 * eta / 6.0
            + (3110.0 - 10310.0 * eta / 3.0) * k_s
            - 1320.0 * (1.0 - eta) * lambda_s
        )
    ) * (chi_s**2 * chi_a)
    psi_S_35PN += (
        265.0 / 8.0
        - 6500.0 * eta / 3.0
        + 40.0 * eta2
        + (3110.0 - 27190.0 * eta / 3.0 + 40.0 * eta2) * k_s
        - 1320.0 * (1.0 - 3 * eta) * lambda_s
        + delta
        * ((3110.0 - 8530.0 * eta / 3.0) * k_a - 1320.0 * (1.0 - eta) * lambda_a)
    ) * (chi_a**2 * chi_s)
    psi_S_35PN *= v7

    psi_NS = (
        psi_NS_0PN
        + psi_NS_1PN
        + psi_NS_15PN
        + psi_NS_2PN
        + psi_NS_25PN
        + psi_NS_3PN
        + psi_NS_35PN
    )
    psi_TLN = psi_TLN_5PN  # + psi_TLN_6PN
    psi_S = (
        psi_S_15PN + psi_S_2PN + psi_S_25PN + psi_S_25PN_log + psi_S_3PN + psi_S_35PN
    )

    return (
        3.0 / 128.0 / eta / v5 * (psi_NS + psi_TLN + psi_S)
    )  # Note that when called with hf3hPN_TLN, we need to include - sign for correct time domain direction


def gen_h0(f, theta, f_ref):
    """
    Computes the Taylor F2 Frequency domain strain waveform with non-standard spin induced quadrupole moment/tidal deformability for object two
    Note that this waveform assumes object 1 is a BH therefore uses the chi*M relation to find C
    Note that this waveform also assumes that object one is the more massive. Therefore the more massive object is always considered a BH
    Parameters:
    f (array): Frequency at which to output the phase [Hz]
    M (float): Total Mass [Msun]
    eta (float): m1*m2/(m1+m2)**2
    s1z (float): Spin of object 1 in z direction
    s2z (float): Spin of object 2 in z direction
    Deff (float): Distance [Mpc]

    Returns:
    Strain (array):
    """
    M, eta, _, _, _, _, Deff, tc, phic = theta
    pre = 3.6686934875530996e-19  # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
    Mchirp = M * eta**0.6
    A0 = (
        Mchirp ** (5.0 / 6.0)
        / (f + 1e-100) ** (7.0 / 6.0)
        / Deff
        / PI ** (2.0 / 3.0)
        * jnp.sqrt(5.0 / 24.0)
    )

    Phase_ = lambda f_: Phif3hPN_TLN(f_, theta[:-3])
    grad_phase = jax.grad(Phase_)
    Phi = Phase_(f)

    # m1 = (M + jnp.sqrt(M**2 - 4 * (eta * M**2))) / 2
    # m2 = (M - jnp.sqrt(M**2 - 4 * (eta * M**2))) / 2

    # c_l = 299792458.0  # speed of light in ms-1
    # Msun = 1.0 / 5.02785e-31
    # G_N = 6.6743e-11 * Msun

    grad_phase_fcut = jax.vmap(grad_phase)(f)
    f_phasecutoff = f[jnp.argmax(grad_phase_fcut)]
    f_ISCO = 4.4e3 * (1 / M)  # Hz
    f_cutoff = min(f_ISCO, f_phasecutoff)
    if f_cutoff <= f[0]:
        raise RuntimeError("Frequency cut off below minimum input frequency")

    t0 = grad_phase(f_cutoff)

    # Lets call the amplitude and phase now
    Phi_ref = Phase_(f_ref)
    Phi -= t0 * (f - f_ref) + Phi_ref

    ext_phase_contrib = 2.0 * PI * f * tc - 2 * phic
    Phi += ext_phase_contrib

    Phi = Phi * jnp.heaviside(f_cutoff - f, 0.0)
    Amp = pre * A0 * jnp.heaviside(f_cutoff - f, 0.0)

    return Amp * jnp.exp(1.0j * Phi)


def gen_taylorF2_tidal_polar(f: Array, params: Array, f_ref: float):
    """
    Generate PhenomD frequency domain waveform following 1508.07253.
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
    h0 = gen_h0(f, params[:-1], f_ref)

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
