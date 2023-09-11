"""This file implements the NRTidalv2 corrections, see http://arxiv.org/abs/1905.06011"""

# FIXME I think I used f everywhere, but some functions should really need fM_s as in IMRPhenomD
# FIXME make sure that jax differentiable is OK

import jax
import jax.numpy as jnp

from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ..typing import Array
from ripple import Mc_eta_to_ms
import sys


# FIXME - move these coefficients below to separate utils file?

# Eq. (20)
C_1 = 3115./1248.
C_THREE_HALVES = - PI
C_2 = 28024205./3302208.
C_FIVE_HALVES = - 4283. * PI / 1092.

# Eq. (21)

N_FIVE_HALVES = 90.550822
N_3 = -60.253578
D_1 = -15.111208
D_2 = 8.0641096

# Combinations, see Eq. (19)

N_1 = C_1 + D_1
N_THREE_HALVES = (C_1 * C_THREE_HALVES - C_FIVE_HALVES - C_THREE_HALVES * D_1 + N_FIVE_HALVES) / (C_1)
N_2 = C_2 + C_1 * D_1 + D_2
D_THREE_HALVES = - (C_FIVE_HALVES + C_THREE_HALVES * D_1 - N_FIVE_HALVES) / (C_1)

# D: just below Eq. (24)

D = 13477.8

def _get_kappa_eff_term(theta):
    """
    Internal function that computes one term for kappa_eff, then can swap arguments around for the other kappa term.
    """
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    return (3./13.) * (1. + 12. * (m2_s / m1_s)) * (m1_s/M_s) ** (5.) * lambda1
    

def get_kappa_eff(theta):
    # Sum the two terms to get total kappa
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    # Parameters with A <-> B
    theta     = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    theta_rev = jnp.array([m2, m1, chi2, chi1, lambda2, lambda1])
    
    return _get_kappa_eff_term(theta) + _get_kappa_eff_term(theta_rev)
    
    
def get_tidal_phase(f: Array, theta: Array, kappa_T_eff: float) -> Array:
    
    # Mass variables
    m1, m2, _, _, _, _ = theta 
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    
    # Build the pade approximant (see Eq. (18) - (21) in NRTidal)
    
    pade_num = 1. + N_1 * f + N_THREE_HALVES * f ** (3. / 2.) + N_2 * f ** 2. + N_FIVE_HALVES * f ** (5. / 2.) + N_3 * f ** 3
    pade_denom = 1. + D_1 * f + D_THREE_HALVES * f ** (3. / 2.) + D_2 * f ** (2.)
    pade = pade_num / pade_denom
    
    # Complete result
    psi_T = - kappa_T_eff * (39./(16. * eta)) * f ** (5./2.) * pade
    
    return psi_T

def _get_spin_phase_correction_term(f: Array, theta: Array) -> Array:
    # Already rescaled below in global function
    m1, m2, chi1, chi2, lambda1, lambda2 = theta 
    # Convert the mass variables
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    
    # Compute the auxiliary variables
    X_1 = m1_s / M_s
    X_2 = m2_s / M_s
    
    log_lambda1 = jnp.log(lambda1)
    
    log_C_Q  = 0.1940 + 0.09163 * log_lambda1 + 0.04812 * log_lambda1 ** 2. - 0.004286 * log_lambda1 ** 3 + 0.00012450 * log_lambda1 ** 4
    log_C_Oc = 0.003131 + 2.071 * log_C_Q - 0.7152 * log_C_Q ** 2 + 0.2458 * log_C_Q ** 3 - 0.03309 * log_C_Q ** 4
    
    C_Q_hat  = jnp.exp(log_C_Q)
    C_Oc_hat = jnp.exp(log_C_Oc)
    
    # Get the coefficients of the corrections
    psi_SS_2  = - 50. * C_Q_hat * X_1 ** 2 * chi1 ** 2
    psi_SS_3  = (5. / 84.) * (9407. + 8218. * X_1 - 2016. * X_1 ** 2) * C_Q_hat * X_1 ** 2 * chi1 ** 2
    psi_SS_35 = 10. * ( ( X_1 ** 2 + (308. / 3.) * X_1 ) * chi1 + (X_2 ** 2 - 89. / 3. * X_2) * chi2 - 40. * PI) * C_Q_hat * X_1 ** 2 * chi1 ** 2 - 440. * C_Oc_hat * X_1 ** 3 * chi1 ** 3
    
    psi_SS = (3. / (128. * eta)) * (psi_SS_2 * f ** (-1./2.) + psi_SS_3 * f ** (1./2.) + psi_SS_35 * f)
    
    # Override - making SS contribution zero
    psi_SS = jnp.zeros_like(f)
    
    return psi_SS

def get_spin_phase_correction(f: Array, theta: Array) -> Array:
    
    m1, m2, chi1, chi2, lambda1, lambda2 = theta 
    
    theta     = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    theta_rev = jnp.array([m2, m1, chi2, chi1, lambda2, lambda1])
    
    return _get_spin_phase_correction_term(f, theta) + _get_spin_phase_correction_term(f, theta_rev)
    
    
def get_tidal_amplitude(f: Array, theta: Array, kappa_T_eff: float, dL=1):
    
    # Mass variables
    m1, m2, _, _, _, _ = theta 
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    
    # Build pade approximant
    pade_num = 1. + (449. / 108.) * f + (22672. / 9.) * f ** 2.89
    pade_denom = 1. + D * f ** 4
    pade = pade_num / pade_denom
    
    # Result
    A_T = - ( (5. * PI * eta) / (24.)) ** (1. / 2.) * 9 * M_s**2  * kappa_T_eff * f ** (13. / 4.) * pade
    # FIXME check if this is correct? Copied from IMRPhenomD
    dist_s = (dL * m_per_Mpc) / C
    return A_T / dist_s

def _get_f_merger(theta, kappa_T_eff):
    
    # TODO - remove later on?
    
    # Already rescaled below in global function
    m1, m2, chi1, chi2, lambda1, lambda2 = theta 
    # Convert the mass variables
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    M = m1 + m2
    eta = m1_s * m2_s / (M_s**2.0)
    
    # Compute the auxiliary variables
    X_1 = m1_s / M_s
    X_2 = m2_s / M_s
    
    # FIXME add the correction here
    
    n_1 = 3.354e-2
    n_2 = 4.315e-5
    d_1 = 7.542e-2
    d_2 = 2.236e-4
    
    omega_hat = 0.3586 * (X_2/X_1) ** (1./2.) * (1. + n_1 * kappa_T_eff + n_2 * kappa_T_eff ** 2)/(1. + d_1 * kappa_T_eff + d_2 * kappa_T_eff ** 2)
    
    f_merger = omega_hat / M_s
    
    return f_merger
    
    

# FIXME - what about the Planck taper? See Eq 25 of NRTidalv2 paper
def get_planck_taper(f: Array, theta: Array, kappa_T_eff):
    
    # Already rescaled below in global function
    m1, m2, chi1, chi2, lambda1, lambda2 = theta 
    # Convert the mass variables
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    
    # Compute the auxiliary variables
    X_1 = m1_s / M_s
    X_2 = m2_s / M_s
    
    # FIXME add the correction here
    
    print(kappa_T_eff)
    
    n_1 = 3.354e-2
    n_2 = 4.3153e-5
    d_1 = 7.542e-2
    d_2 = 2.236e-4
    
    omega_hat = 0.3586 * (X_2/X_1) ** (1./2.) * (1. + n_1 * kappa_T_eff + n_2 * kappa_T_eff ** 2)/(1. + d_1 * kappa_T_eff + d_2 * kappa_T_eff ** 2)
    
    # Safety override: 
    A_P = jnp.ones_like(f)
    
    return A_P


def gen_NRTidalv2(f: Array, params: Array, f_ref: float, IMRphenom: str) -> Array:
    """
    Generate NRTidalv2 frequency domain waveform following 1508.07253.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    lambda1: Dimensionless tidal deformability of primary object
    lambda2: Dimensionless tidal deformability of secondary object
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    f_ref: Reference frequency for the waveform
    
    IMRphenom: string selecting the BBH approximant

    Returns:
    --------
      h0 (array): Strain
    """
    
    # Get parameters
    # FIXME now this is pretty cumbersome
    theta_intrinsic, theta_extrinsic = params[:6], params[6:]
    m1, m2, chi1, chi2, lambda1, lambda2 = theta_intrinsic
    
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    
    
    # Compute auxiliary quantities like kappa
    kappa_T_eff = get_kappa_eff(theta=theta_intrinsic)
        
    # Get the parameters that are passed to the BBH waveform, all except lambdas
    bbh_params = jnp.concatenate((jnp.array([m1, m2, chi1, chi2]), theta_extrinsic))
    
    if IMRphenom == "IMRPhenomD":
        from ripple.waveforms.IMRPhenomD import (
            gen_IMRPhenomD as bbh_waveform_generator,
        )
    ## FIXME what to do in other cases?
    # if IMRphenom == "IMRPhenomXAS":
    #     from ripple.waveforms.IMRPhenomXAS import (
    #         gen_IMRPhenomXAS_hphc as bbh_waveform_generator,
    #     )
    # if IMRphenom == "IMRPhenomPv2":
    #     from ripple.waveforms.IMRPhenomPv2 import (
    #         gen_IMRPhenomPv2_hphc as bbh_waveform_generator,
    #     )
    
    # Generate BBH waveform strain and get its amplitude and phase
    h0_bbh = bbh_waveform_generator(f, bbh_params, f_ref)
    A_bbh = jnp.abs(h0_bbh)
    psi_bbh = h0_bbh / jnp.abs(h0_bbh)
    
    # Add the tidal amplitude
    A_T = get_tidal_amplitude(f * M_s, theta_intrinsic, kappa_T_eff, dL=theta_extrinsic[0])
    
    
    A_P = get_planck_taper(f * M_s, theta_intrinsic, kappa_T_eff)
    
    # Add tidal phase
    psi_T = get_tidal_phase(f * M_s, theta_intrinsic, kappa_T_eff)
    # FIXME - get correct SS terms
    psi_SS = get_spin_phase_correction(f * M_s, theta_intrinsic)
    
    h0 = A_P * (h0_bbh + A_T * jnp.exp(1j * - psi_bbh)) * jnp.exp(1.j * -(psi_T + psi_SS))
    
    return h0


def gen_NRTidalv2_hphc(f: Array, params: Array, f_ref: float, IMRphenom: str) -> Array:
    """
    vars array contains both intrinsic and extrinsic variables
    
    theta = [Mchirp, eta, chi1, chi2, lambda1, lambda2, D, tc, phic, inclination]
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
    iota = params[-1]
    h0 = gen_NRTidalv2(f, params[:-1], f_ref, IMRphenom=IMRphenom)
    
    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc