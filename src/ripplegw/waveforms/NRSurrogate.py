import jax.numpy as jnp
from ..typing import Array
from ripplegw import Mc_eta_to_ms, ht_to_hf
from jaxNRSur.SurrogateModel import NRHybSur3dq8Model


def gen_NRSurrogate(f: Array, params: Array, f_ref: float):
    """
    Generate NRSurrogate
    params array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
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
    theta_NRSurrogate = jnp.array([m1 / m2, params[2], params[3]])
    M_tot = m1 + m2

    model = NRHybSur3dq8Model()
    h0 = ht_to_hf(f, theta_NRSurrogate, model)

    phase_shift = jnp.exp(-1j * (2 * jnp.pi * f - f_ref * params[5] + params[6]))
    scaling_factor = (M_tot**2.0) / (params[4])

    return h0 * scaling_factor * phase_shift


def gen_NRSurrogate_hphc(f: Array, params: Array, f_ref: float):
    """
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
    h0 = gen_NRSurrogate(f, params, f_ref)

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
