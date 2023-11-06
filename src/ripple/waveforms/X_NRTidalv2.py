"""This file implements the NRTidalv2 corrections, see http://arxiv.org/abs/1905.06011"""

# FIXME make sure that jax differentiable is OK

import jax
import jax.numpy as jnp

from ..constants import EulerGamma, gt, m_per_Mpc, C, PI, TWO_PI, MSUN, MRSUN
from ..typing import Array
from ripple import Mc_eta_to_ms, ms_to_Mc_eta
import sys
from .IMRPhenomD import get_Amp0
from .utils_tidal import *

# D: just below Eq. (24)
NRTidalv2_coeffs = jnp.array([
    2.4375, # c_Newt
    -12.615214237993088, # n_1
    19.0537346970349, # n_3over2
    -21.166863146081035, # n_2
    90.55082156324926, # n_5over2
    -60.25357801943598, # n_3
    -15.111207827736678, # d_1
    22.195327350624694, # d_3over2
    8.064109635305156, # d_2
])


def get_tidal_phase(x: Array, theta: Array, kappa: float) -> Array:
    
    # Mass variables
    m1, m2, _, _, _, _ = theta 
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # Compute ratios
    X1 = m1_s / M_s
    X2 = m2_s / M_s

    # Compute powers
    x_2 = x ** (2.)
    x_3 = x ** (3.)
    x_3over2 = x ** (3.0/2.0)
    x_5over2 = x ** (5.0/2.0)

    c_Newt   =   2.4375
    n_1      = -12.615214237993088
    n_3over2 =  19.0537346970349
    n_2      = -21.166863146081035
    n_5over2 =  90.55082156324926
    n_3      = -60.25357801943598
    d_1      = -15.111207827736678
    d_3over2 =  22.195327350624694
    d_2      =   8.064109635305156

    # Pade approximant
    num = 1.0 + (n_1 * x) + (n_3over2 * x_3over2) + (n_2 * x_2) + (n_5over2 * x_5over2) + (n_3 * x_3)
    den = 1.0 + (d_1 * x) + (d_3over2 * x_3over2) + (d_2 * x_2)
    ratio = num / den
    
    psi_T  = - kappa * c_Newt / (X1 * X2) * x_5over2
    psi_T *= ratio
    
    return psi_T



def get_spin_phase_correction(x: Array, theta: Array) -> Array:
    
    m1, m2, chi1, chi2, lambda1, lambda2 = theta

    # Convert the mass variables
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # Compute the auxiliary variables
    X1 = m1_s / M_s
    X2 = m2_s / M_s

    X1sq = X1 * X1
    X2sq = X2 * X2
    chi1_sq = chi1 * chi1
    chi2_sq = chi2 * chi2

    # Compute quadparam1
    quadparam1, octparam1 = get_quadparam_octparam(lambda1)
    quadparam2, octparam2 = get_quadparam_octparam(lambda2)
    
    # Remember to remove 1 from quadrupole and octupole, for the BBH baseline
    
    SS_2 = - 50. * ((quadparam1 - 1) * chi1_sq * X1sq + (quadparam2 - 1) * chi2_sq * X2sq)
    SS_3 = 5.0/84.0 * (9407.0 + 8218.0 * X1 - 2016.0 * X1sq) * (quadparam1 - 1) * X1sq * chi1_sq \
         + 5.0/84.0 * (9407.0 + 8218.0 * X2 - 2016.0 * X2sq) * (quadparam2 - 1) * X2sq * chi2_sq
    
    # Following is taken from LAL source code
    SS_3p5 = - 400. * PI * (quadparam1 - 1) * chi1_sq * X1sq \
             - 400. * PI * (quadparam2 - 1) * chi2_sq * X2sq
    # Just add SSS_3p5 to SS_3p5 for simplicity
    SSS_3p5 = 10. * ((X1sq + 308./3. * X1) * chi1 + (X2sq - 89./3. * X2) * chi2) * (quadparam1 - 1) * X1sq * chi1_sq \
            + 10. * ((X2sq + 308./3. * X2) * chi2 + (X1sq - 89./3. * X1) * chi1) * (quadparam2 - 1) * X2sq * chi1_sq \
                - 440. * (octparam1 - 1) * X1 * X1sq * chi1_sq * chi1 \
                - 440. * (octparam2 - 1) * X2 * X2sq * chi2_sq * chi2

    prefac = (3. / (128. * eta))
    psi_SS = prefac * (SS_2 * x ** (-1./2.) + SS_3 * x ** (1./2.) + (SS_3p5 + SSS_3p5) * x)

    return psi_SS

def get_tidal_amplitude(x: Array, theta: Array, kappa: float, distance: float =1):
    """_summary_

    Args:
        x (Array): _description_
        theta (Array): _description_
        kappa (float): _description_
        distance (float, optional): Distance to source in megaparsecs. Defaults to 1.

    Returns:
        _type_: _description_
    """
    
    # Mass variables
    m1, m2, _, _, _, _ = theta 
    M = m1 + m2
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)
    
    distance *= m_per_Mpc
    
    # Pade approximant
    n1   = 4.157407407407407
    n289 = 2519.111111111111
    d    = 13477.8073677
    num = 1.0 + n1 * x + n289 * x ** 2.89
    den = 1.0 + d * x ** 4.
    poly = num / den
    
    # Prefactor from lal
    prefac = - 9.0 * kappa
    ampT = prefac * x ** (13. / 4.) * poly
    
    # Now get the FULL tidal amplitude - have two extra prefactors to take into account
    amp0 = get_amp0_lal(M, distance)
    ampT *= amp0 * 2 * jnp.sqrt(PI / 5)
    
    return ampT 


def _get_merger_frequency(theta, kappa=None):
    
    # Already rescaled below in global function
    m1, m2, chi1, chi2, lambda1, lambda2 = theta 
    # Convert the mass variables
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    M = m1 + m2

    q = m1_s / m2_s

    X1 = m1_s / M_s
    X2 = m2_s / M_s
    
    # If kappa was not given, compute it
    if kappa is None:
        kappa = get_kappa(theta)
        
    a_0 = 0.3586
    n_1 = 3.35411203e-2
    n_2 = 4.31460284e-5
    d_1 = 7.54224145e-2
    d_2 = 2.23626859e-4
    
    kappa_2 = kappa * kappa

    num = 1.0 + n_1 * kappa + n_2 * kappa_2
    den = 1.0 + d_1 * kappa + d_2 * kappa_2
    Q_0 = a_0 * (q) ** (-1./2.)

    # Dimensionless angular frequency of merger
    Momega_merger = Q_0 * (num / den)

    # convert from angular frequency to frequency (divide by 2*pi) and then convert from dimensionless frequency to Hz (divide by mtot * LAL_MTSUN_SI)
    fHz_merger = Momega_merger / (M * gt) / (TWO_PI)

    return fHz_merger




def _gen_NRTidalv2(f: Array, theta_intrinsic: Array, theta_extrinsic: Array, h0_bbh: Array):

    m1, m2, chi1, chi2, lambda1, lambda2 = theta_intrinsic
    M = m1 + m2
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # Compute auxiliary quantities like kappa
    kappa = get_kappa(theta=theta_intrinsic)

    # Compute x: see NRTidalv2 paper
    x = (PI * M_s * f) ** (2.0/3.0)

    # Get BBH amplitude and phase
    A_bbh = jnp.abs(h0_bbh)
    psi_bbh = jnp.log(h0_bbh / A_bbh) * 1.j
    
    # Get tidal amplitude and Planck taper
    A_T = get_tidal_amplitude(x, theta_intrinsic, kappa, distance=theta_extrinsic[0])
    f_merger = _get_merger_frequency(theta_intrinsic, kappa)
    A_P = jnp.ones_like(f) - get_planck_taper(f, f_merger)

    # Get tidal phase and spin corrections for BNS
    psi_T = get_tidal_phase(x, theta_intrinsic, kappa)
    
    psi_SS = get_spin_phase_correction(x, theta_intrinsic)
    h0 = A_P * (A_bbh + A_T) * jnp.exp(1.j * -(psi_bbh + psi_T + psi_SS))

    return h0

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
    
    IMRphenom: string selecting the underlying BBH approximant

    Returns:
    --------
      h0 (array): Strain
    """
    
    # Get component masses
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    chi1, chi2 = params[2], params[3]
    lambda1, lambda2 = params[4], params[5]
    
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * gt
    theta_extrinsic = params[6:]

    # Get the parameters that are passed to the BBH waveform, all except lambdas
    bbh_params = jnp.concatenate((jnp.array([params[0], params[1], params[2], params[3]]), theta_extrinsic))

    # TODO - make compatible with other waveforms as well
    if IMRphenom == "IMRPhenomD":
        from ripple.waveforms.IMRPhenomD import (
            gen_IMRPhenomD as bbh_waveform_generator,
        )
    else:
        print("IMRPhenom string not recognized")
        return jnp.zeros_like(f)
    
    # Generate BBH waveform strain and get its amplitude and phase
    h0_bbh = bbh_waveform_generator(f, bbh_params, f_ref)

    # Use BBH waveform and add tidal corrections
    return _gen_NRTidalv2(f, theta_intrinsic, theta_extrinsic, h0_bbh)


def gen_NRTidalv2_hphc(f: Array, params: Array, f_ref: float, IMRphenom: str="IMRPhenomD"):
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
