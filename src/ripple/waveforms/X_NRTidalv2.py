"""This file implements the NRTidalv2 corrections, see http://arxiv.org/abs/1905.06011"""

import jax
import jax.numpy as jnp
# import numpy as np

# from .IMRPhenomD_QNMdata import fM_CUT
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ..typing import Array
from ripple import Mc_eta_to_ms


# FIXME - move these coefficients below to separate utils file?

# Eq. (20)
C_1 = 3115./1248.
C_2 = 28024205./3302208.
C_THREE_HALVES = - PI
C_FIVE_HALVES = - 4283. * PI / 1092.

# Eq. (21)

N_FIVE_HALVES = 90.550822
D_1 = -15.111208
N_3 = -60.253578
D_2 = 8.0641096

# Combinations, see Eq. (19)

N_1 = C_1 + D_1
N_THREE_HALVES = (C_1 * C_THREE_HALVES - C_FIVE_HALVES - C_THREE_HALVES * D_1 + N_FIVE_HALVES) / (C_1)
N_2 = C_2 + C_1 * D_1 + D_2
D_THREE_HALVES = - (C_FIVE_HALVES + C_THREE_HALVES * D_1 - N_FIVE_HALVES) / (C_1)

# D: just below Eq. (24)

D = 13477.8

def get_kappa_eff_term(theta):
    # FIXME - do I need to get the units? Check with other parts of ripple code, think not?
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    M = m1 + m2
    return 3./13. * (1. + 12. * (m2 / m1)) * (m1/M) ** (5.) * lambda1
    

def get_kappa_eff(theta):
    # Sum the two terms to get total kappa
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    # Parameters with A <-> B
    theta_rev = jnp.array([m2, m1, chi2, chi1, lambda2, lambda1])
    
    return get_kappa_eff_term(theta) + get_kappa_eff_term(theta_rev)
    
    
def get_tidal_phase(f: Array, theta: Array, kappa_T_eff: float) -> Array:
    
    # FIXME - have to change the units?
    
    # Mass variables
    m1, m2, _, _, _, _ = theta 
    M = m1 + m2
    eta = m1 * m2 / (M**2.0)
    
    # Build the pade approximant (see Eq. (18) - (21) in NRTidal)
    
    pade_num = 1. + N_1 * f + N_THREE_HALVES * f ** (3. / 2.) + N_FIVE_HALVES * f ** (5. / 2.) + N_3 * f ** 3
    pade_denom = 1. + D_1 * f + D_THREE_HALVES * f ** (3. / 2.) + D_2 * f ** (2.)
    pade = pade_num / pade_denom
    
    # Result
    psi_T = - kappa_T_eff * (39./(16. * eta)) * f ** (5./2.) * pade
    
    return psi_T
    
def get_tidal_amplitude(f: Array, theta: Array, kappa_T_eff: float, dL=1):
    
    # Mass variables
    m1, m2, _, _, _, _ = theta 
    M = m1 + m2
    eta = m1 * m2 / (M**2.0)
    
    # FIXME add the correction here
    
    # Build pade
    pade_num = 1. + (449. / 108.) * f + (22672. / 9.) * f ** 2.89
    pade_denom = 1. + D * f ** 4
    pade = pade_num / pade_denom
    
    # Result
    A_T = - ( (5. * PI * eta) / (24.)) ** (1. / 2.) * 9 * M**2 / dL * f ** (13. / 4.) * pade
    
    return A_T

# FIXME - what about the Planck taper? See Eq 25 of NRTidalv2 paper
def get_planck_taper(f: Array, theta: Array):
    
    # FIXME add the correction here
    
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
    print(params)
    theta_intrinsic, theta_extrinsic = params[:6], params[6:]
    print(theta_intrinsic)
    print(theta_extrinsic)
    m1, m2, chi1, chi2, lambda1, lambda2 = theta_intrinsic
    
    
    # Compute auxiliary quantities like kappa
    kappa_T_eff = get_kappa_eff(theta=theta_intrinsic)
        
    # Get the parameters that are passed to the BBH waveform, all except lambdas
    bbh_params = jnp.array([m1, m2, chi1, chi2]) + theta_extrinsic
    
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
    
    # Generate BBH waveform strain and get phase
    h0_bbh = bbh_waveform_generator(f, bbh_params, f_ref)
    psi_bbh = h0_bbh / jnp.abs(h0_bbh)
    
    # Build BNS waveform
    h0 = h0_bbh
    
    # Add the tidal amplitude
    A_T = get_tidal_amplitude(f, theta_intrinsic, kappa_T_eff, dL=theta_intrinsic[0])
    h0 += A_T * jnp.exp(1j * -psi_bbh)
    
    # Add tidal phase
    psi_T = get_tidal_phase(f, theta_intrinsic, kappa_T_eff)
    h0 *= jnp.exp(1j * -psi_T)
    
    # Add the Planck taper:
    A_P = get_planck_taper(f, theta_intrinsic)
    h0 *= A_P
    
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