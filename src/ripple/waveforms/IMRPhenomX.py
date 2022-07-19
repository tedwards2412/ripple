from math import pi
import jax
import jax.numpy as jnp
from ..constants import EulerGamma, gt, m_per_Mpc, C
from ..typing import Array
from ripple import Mc_eta_to_ms


# def get_inspiral_phase(fM_s: Array, theta: Array, coeffs: Array) -> Array:
#     """
#     Calculate the inspiral phase for the IMRPhenomD waveform.
#     """
#     # First lets calculate some of the vairables that will be used below
#     # Mass variables
#     m1, m2, chi1, chi2 = theta
#     m1_s = m1 * gt
#     m2_s = m2 * gt
#     M_s = m1_s + m2_s
#     eta = m1_s * m2_s / (M_s ** 2.0)

#     # First lets construct the phase in the inspiral (region I)
#     m1M = m1_s / M_s
#     m2M = m2_s / M_s

    
#     return None


# def get_intermediate_raw_phase(fM_s: Array, theta: Array, coeffs: Array) -> Array:
#     m1, m2, _, _ = theta
#     m1_s = m1 * gt
#     m2_s = m2 * gt
#     M_s = m1_s + m2_s
#     eta = m1_s * m2_s / (M_s ** 2.0)

#     return None


# def get_mergerringdown_raw_phase(fM_s: Array, theta: Array, coeffs: Array, f_RD, f_damp) -> Array:
#     m1, m2, _, _ = theta
#     m1_s = m1 * gt
#     m2_s = m2 * gt
#     M_s = m1_s + m2_s
#     eta = m1_s * m2_s / (M_s ** 2.0)


#     return None


# def get_Amp0(fM_s: Array, eta: float) -> Array:
#     Amp0 = (
#         (2.0 / 3.0 * eta) ** (1.0 / 2.0) * (fM_s) ** (-7.0 / 6.0) * pi ** (-1.0 / 6.0)
#     )
#     return Amp0


def get_inspiral_Amp(fM_s: Array, theta: Array, coeffs: Array) -> Array:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)
    eta2 = eta * eta
    eta3 = eta * eta2
    delta = jnp.sqrt(1.-4.*eta)

    Seta = jnp.sqrt(1.0 - 4.0 * eta)
    SetaPlus1 = 1.0 + Seta

    # Spin variables
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2

    A0 = 1.
    A2 = -323./224. + 451.*eta/168.
    A3 = chi1*(27.*delta/16. - 11.*eta/12. + 27./16.) + chi2*(-27.*delta/16. - 11.*eta/12. + 27./16.)
    A4 = chi12*(-81.*delta/64. + 81.*eta/32. - 81./64.) + chi22*(81.*delta/64. + 81.*eta/32. - 81./64.) + (105271.*eta2/24192. - 1975055.*eta/338688 - 27312085./8128512.) - 47.*eta*chi1*chi2/16.
    A5 = 1.
    # A6 = 

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
        # + A6 * (fM_s ** 2.0)
        # # Now we add the coefficient terms
        # + A7 * (fM_s ** (7.0 / 3.0))
        # + A8 * (fM_s ** (8.0 / 3.0))
        # + A9 * (fM_s ** 3.0)
    )

    return Amp_Ins


# def get_intermediate_Amp(
#     fM_s: Array, theta: Array, coeffs: Array, f1, f3, f_RD, f_damp
# ) -> Array:
#     m1, m2, _, _ = theta
#     m1_s = m1 * gt
#     m2_s = m2 * gt
#     M_s = m1_s + m2_s

#     return None


# def get_mergerringdown_Amp(fM_s: Array, theta: Array, coeffs: Array, f_RD, f_damp) -> Array:
#     m1, m2, _, _ = theta
#     m1_s = m1 * gt
#     m2_s = m2 * gt
#     M_s = m1_s + m2_s
#     return None


# @jax.jit
# def Phase(f: Array, theta: Array) -> Array:
#     """
#     Computes the phase of the PhenomD waveform following 1508.07253.
#     Sets time and phase of coealence to be zero.

#     Returns:
#     --------
#         phase (array): Phase of the GW as a function of frequency
#     """
#     # First lets calculate some of the vairables that will be used below
#     # Mass variables
#     m1, m2, _, _ = theta
#     m1_s = m1 * gt
#     m2_s = m2 * gt
#     M_s = m1_s + m2_s


#     # And now we can combine them by multiplying by a set of heaviside functions
#     # phase = (
#     #     phi_Ins * jnp.heaviside(f1 - f, 0.5)
#     #     + jnp.heaviside(f - f1, 0.5) * phi_Inter * jnp.heaviside(f2 - f, 0.5)
#     #     + phi_MR * jnp.heaviside(f - f2, 0.5)
#     # )

#     return None


# @jax.jit
# def Amp(f: Array, theta: Array, D=1) -> Array:
#     """
#     Computes the amplitude of the PhenomD frequency domain waveform following 1508.07253.
#     Note that this waveform also assumes that object one is the more massive.

#     Returns:
#     --------
#       Amplitude (array):
#     """

#     # First lets calculate some of the vairables that will be used below
#     # Mass variables
#     m1, m2, _, _ = theta
#     m1_s = m1 * gt
#     m2_s = m2 * gt
#     M_s = m1_s + m2_s
#     eta = m1_s * m2_s / (M_s ** 2.0)

#     # And now we can combine them by multiplying by a set of heaviside functions
#     # Amp = (
#     #     Amp_Ins * jnp.heaviside(f3 - f, 0.5)
#     #     + jnp.heaviside(f - f3, 0.5) * Amp_Inter * jnp.heaviside(f4 - f, 0.5)
#     #     + Amp_MR * jnp.heaviside(f - f4, 0.5)
#     # )

#     # Prefactor
#     Amp0 = get_Amp0(f * M_s, eta) * (
#         2.0 * jnp.sqrt(5.0 / (64.0 * pi))
#     )  # This second factor is from lalsuite...

#     # Need to add in an overall scaling of M_s^2 to make the units correct
#     dist_s = (D * m_per_Mpc) / C
#     # return Amp0 * Amp * (M_s ** 2.0) / dist_s
#     return None


# @jax.jit
# def _gen_IMRPhenomXAS(
#     f: Array, theta_intrinsic: Array, theta_extrinsic: Array, coeffs: Array
# ):
#     M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * gt
#     # Lets call the amplitude and phase now
#     Psi = Phase(f, theta_intrinsic)
#     A = Amp(f, theta_intrinsic, D=theta_extrinsic[0])
#     h0 = A * jnp.exp(1j * -Psi)
#     return h0


# @jax.jit
# def gen_IMRPhenomXAS(f: Array, params: Array):
#     """
#     Generate PhenomXAS frequency domain waveform following 2001.11412.
#     Note that this waveform also assumes that object one is the more massive.
#     vars array contains both intrinsic and extrinsic variables
#     theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
#     Mchirp: Chirp mass of the system [solar masses]
#     eta: Symmetric mass ratio [between 0.0 and 0.25]
#     chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
#     chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
#     D: Luminosity distance to source [Mpc]
#     tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
#     phic: Phase of coalesence

#     Returns:
#     --------
#       hp (array): Strain of the plus polarization
#       hc (array): Strain of the cross polarization
#     """
#     # Lets make this easier by starting in Mchirp and eta space
#     m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
#     theta_intrinsic = jnp.array([m1, m2, params[2], params[3]])
#     theta_extrinsic = jnp.array([params[4], params[5], params[6]])

#     # h0 = _gen_IMRPhenomXAS(f, theta_intrinsic, theta_extrinsic, coeffs)
#     return None
