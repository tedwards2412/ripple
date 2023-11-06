"""This file implements the NRTidalv3 corrections, see http://arxiv.org/abs/1905.06011 and paper yet unreleased"""

# FIXME make sure that jax differentiable is OK

import jax
import jax.numpy as jnp

from ..constants import EulerGamma, gt, m_per_Mpc, C, PI, TWO_PI, MSUN, MRSUN
from ..typing import Array
from ripple import Mc_eta_to_ms, ms_to_Mc_eta
import sys
from .IMRPhenomD import get_Amp0
from .utils_tidal import *


# TODO get coeffs
# NRTidalv3_coeffs = jnp.array([
# ])

def get_tidal_phase(f: Array, theta: Array, kappa: float) -> Array:
    
    # Mass variables
    m1, m2, _, _, lambda1, lambda2 = theta 
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    # eta = m1_s * m2_s / (M_s**2.0)

    # Compute ratios
    X1 = m1_s / M_s
    X2 = m2_s / M_s
    q = X1 / X2

    # Compute powers
    M_omega = (PI * M_s * f)
    x = M_omega ** (2.0/3.0)
    
    x_2      = x ** (2.)
    x_3      = x ** (3.)
    x_3over2 = x ** (3.0/2.0)
    x_5over2 = x ** (5.0/2.0)

    # Initialize coefficients to be used in this function
    s10 = 2.73000423e-01
    s11 = 3.64169971e-03
    s12 = 1.76144380e-03

    s20 =  2.68793291e+01
    s21 =  1.18175396e-02
    s22 = -5.39996790e-03

    s30 =  1.42449682e-01
    s31 = -1.70505852e-05
    s32 =  3.38040594e-05

    alpha = -8.08155404e-03
    beta  = -1.13695919e+00

    n_5over20 = -9.40654388e+02
    n_5over21 =  6.26517157e+02
    n_5over22 =  5.53629706e+02
    n_5over23 =  8.84823087e+01

    n_30 =  4.05483848e+02
    n_31 = -4.25525054e+02
    n_32 = -1.92004957e+02
    n_33 = -5.10967553e+01

    d_10 =  3.80343306e+00
    d_11 = -2.52026996e+01
    d_12 = -3.08054443e+00
    
    # Get all auxiliary quantities
    s1 = 1 + s10 + s11 * kappa + s12 * q * kappa
    s2 = 1 + s20 + s21 * kappa + s22 * q * kappa
    s3 =     s30 + s31 * kappa + s32 * q * kappa
    
    dynk2bar = 1.0 + (s1 - 1.0) / (1.0 + jnp.exp(- s2 * (2.0 * M_omega - s3))) - (s1 - 1.0) / (1.0 + jnp.exp(s2 * s3)) - 2.0 * M_omega * (s1 - 1) * s2 / ((1.0 + jnp.exp(s2 * s3)) ** 2.0)

    kappaA = 3.0 * X2 * (X1 ** 4.0) * lambda1
    kappaB = 3.0 * X1 * (X2 ** 4.0) * lambda2
    
    dynkappaA = kappaA * dynk2bar
    dynkappaB = kappaB * dynk2bar
    
    # Get powers
    kappaA_alpha = (kappaA + 1) ** alpha
    kappaB_alpha = (kappaB + 1) ** alpha

    Xa_beta = X1 ** beta
    Xb_beta = X2 ** beta

    # Pade coefficients
    n_5over2A = n_5over20 + n_5over21 * X1 + n_5over22 * kappaA_alpha + n_5over23 * Xa_beta
    n_5over2B = n_5over20 + n_5over21 * X2 + n_5over22 * kappaB_alpha + n_5over23 * Xb_beta
    
    n_3A      = n_30 + n_31 * X1 + n_32 * kappaA_alpha + n_33 * Xa_beta
    n_3B      = n_30 + n_31 * X2 + n_32 * kappaB_alpha + n_33 * Xb_beta
    
    d_1A      = d_10 + d_11 * X1 + d_12 * Xa_beta
    d_1B      = d_10 + d_11 * X2 + d_12 * Xb_beta
    
    # 7.5PN coefficients
    c_NewtA   = (3.0 * (X1 + X2) ** 2.0 * (12.0 - 11.0 * X1)) / (16 * X1 * (X2) ** 2.0)
    c_NewtB   = (3.0 * (X2 + X1) ** 2.0 * (12.0 - 11.0 * X2)) / (16 * X2 * (X1) ** 2.0)
    
    c_1A      = -5.0 * (260.0 * (X1) ** 3.0 - 2286.0 * (X1) ** 2.0 - 919.0 * X1 + 3179.0) / (672.0 * (11.0 * X1 - 12.0))
    c_1B      = -5.0 * (260.0 * (X2) ** 3.0 - 2286.0 * (X2) ** 2.0 - 919.0 * X2 + 3179.0) / (672.0 * (11.0 * X2 - 12.0))
    
    c_3over2A = -3.14159265358979323846
    c_3over2B = -3.14159265358979323846
    
    c_2A      = (5.0 * (4572288.0 * (X1) ** 5.0 - 20427120.0 * (X1) ** 4.0 + 158378220.0 * (X1) ** 3.0 + 174965616.0 * (X1) ** 2.0 + 43246839.0 * X1 - 387973870.0)) / (27433728.0 * (11.0 * X1 - 12.0))
    c_2B      = (5.0 * (4572288.0 * (X2) ** 5.0 - 20427120.0 * (X2) ** 4.0 + 158378220.0 * (X2) ** 3.0 + 174965616.0 * (X2) ** 2.0 + 43246839.0 * X2 - 387973870.0)) / (27433728.0 * (11.0 * X2 - 12.0))
    
    c_5over2A = -3.14159265358979323846 * (10520.0 * (X1) ** 3.0 - 7598.0 * (X1) ** 2.0 + 22415.0 * X1 - 27719.0) / (672.0 * (11.0 * X1 - 12.0))
    c_5over2B = -3.14159265358979323846 * (10520.0 * (X2) ** 3.0 - 7598.0 * (X2) ** 2.0 + 22415.0 * X2 - 27719.0) / (672.0 * (11.0 * X2 - 12.0))
    
    # Pade Coefficients constrained with PN
    n_1A      = c_1A + d_1A
    n_1B      = c_1B + d_1B
    
    n_3over2A = (c_1A * c_3over2A - c_5over2A - c_3over2A * d_1A + n_5over2A) / c_1A
    n_3over2B = (c_1B * c_3over2B - c_5over2B - c_3over2B * d_1B + n_5over2B) / c_1B
    
    n_2A      = c_2A + c_1A * d_1A # + d_2A
    n_2B      = c_2B + c_1B * d_1B # + d_2B
    
    d_3over2A = - (c_5over2A + c_3over2A * d_1A - n_5over2A) / c_1A
    d_3over2B = - (c_5over2B + c_3over2B * d_1B - n_5over2B) / c_1B
    
    # Build Pade approximants
    factorA = - c_NewtA * x_5over2 * dynkappaA
    factorB = - c_NewtB * x_5over2 * dynkappaB

    numA = 1.0 + (n_1A * x) + (n_3over2A * x_3over2) + (n_2A * x_2) + (n_5over2A * x_5over2) + (n_3A * x_3)
    numB = 1.0 + (n_1B * x) + (n_3over2B * x_3over2) + (n_2B * x_2) + (n_5over2B * x_5over2) + (n_3B * x_3)
    
    denA = 1.0 + (d_1A * x) + (d_3over2A * x_3over2) # + (d_2A*PN_x_2)
    denB = 1.0 + (d_1B * x) + (d_3over2B * x_3over2) # + (d_2B*PN_x_2)
    
    ratioA = numA / denA
    ratioB = numB / denB
    
    tidal_phaseA = factorA * ratioA
    tidal_phaseB = factorB * ratioB
    
    tidal_phase = tidal_phaseA + tidal_phaseB
    
    return tidal_phase


# TODO - add this new phase to overall phase?
def get_tidal_phase_PN(x: Array, theta: Array) -> Array:
    
    # Mass variables
    m1, m2, _, _, lambda1, lambda2 = theta 
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    # eta = m1_s * m2_s / (M_s**2.0)

    # Compute ratios
    X1 = m1_s / M_s
    X2 = m2_s / M_s
    q = X1 / X2

    # Compute powers
    x_2 = x ** (2.)
    x_3 = x ** (3.)
    x_3over2 = x ** (3.0/2.0)
    x_5over2 = x ** (5.0/2.0)

    kappaA = 3.0 * X2 * (X1 ** 4.0) * lambda1
    kappaB = 3.0 * X1 * (X2 ** 4.0) * lambda2

    c_NewtA   = (3.0 * (X1 + X2) ** 2.0 * (12.0 -11.0 * X1)) / (16 * X1 * X2 ** 2.0)
    c_1A      = -5.0 * (260.0 * (X1) ** 3.0 - 2286.0 * (X1) ** 2.0 - 919.0 * X1 + 3179.0) / (672.0 * (11.0 * X1-12.0))
    c_3over2A = -3.14159265358979323846
    c_2A      = (5.0 * (4572288.0 * (X1) ** 5.0 - 20427120.0 * (X1) ** 4.0 + 158378220.0 * (X1) ** 3.0 + 174965616.0 * (X1) ** 2.0 + 43246839.0 * X1 - 387973870.0)) / (27433728.0 * (11.0 * X1 - 12.0))
    c_5over2A = -3.14159265358979323846 * (10520.0 * (X1) ** 3.0 - 7598.0 * (X1) ** 2.0 + 22415.0 * X1 - 27719.0) / (672.0 * (11.0 * X1 - 12.0))

    c_NewtB   = (3.0 * (X2 + X1) ** 2.0) * (12.0 -11.0 * X2) / (16 * X2 * (X1) ** 2.0)
    c_1B      = -5.0 * (260.0 * (X2) ** 3.0 - 2286.0 * (X2) ** 2.0 - 919.0 * X2 + 3179.0) / (672.0 * (11.0 * X2 - 12.0))
    c_3over2B = -3.14159265358979323846
    c_2B      = (5.0 * (4572288.0 * (X2) ** 5.0 - 20427120.0 * (X2) ** 4.0 + 158378220.0 * (X2) ** 3.0 + 174965616.0 * (X2) ** 2.0 + 43246839.0 * X2 - 387973870.0)) / (27433728.0 * (11.0 * X2 - 12.0))
    c_5over2B = -3.14159265358979323846 * (10520.0 * (X2) ** 3.0 - 7598.0 * (X2) ** 2.0 + 22415.0 * X2 - 27719.0) / (672.0 * (11.0 * X2 - 12.0))

    factorA = -c_NewtA * x_5over2 * kappaA
    factorB = -c_NewtB * x_5over2 * kappaB

    tidal_phasePNA = factorA * (1.0 + (c_1A * x) + (c_3over2A * x_3over2) + (c_2A * x_2) + (c_5over2A * x_5over2))
    tidal_phasePNB = factorB * (1.0 + (c_1B * x) + (c_3over2B * x_3over2) + (c_2B * x_2) + (c_5over2B * x_5over2))

    tidal_phasePN = tidal_phasePNA + tidal_phasePNB

    return tidal_phasePN


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

def _get_merger_frequency(theta, kappa=None):
    """
    Uses a new fit from Gonzalez, et. al (2022); Eq. (23) of https://arxiv.org/abs/2210.16366.

    Args:
        theta (_type_): _description_
        kappa (_type_, optional): _description_. Defaults to None. Not used, but kept to have same function signature as v2.

    Returns:
        _type_: _description_
    """
    # TODO this has to be changed to the v3 implementation, if different?
    
    # Already rescaled below in global function
    m1, m2, chi1, chi2, lambda1, lambda2 = theta 
    # Convert the mass variables
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    M = m1 + m2
    # q = m1_s / m2_s
    X1 = m1_s / M_s
    X2 = m2_s / M_s
    nu = X1 * X2
    
    # TODO this must be aligned spin component, change so that it works for other BBH waveforms as well
    chi1_AS = chi1
    chi2_AS = chi2

    kappa2eff = 3.0 * nu * (X1 ** (3.0) * lambda1 + X2 ** (3.0) * lambda2)

    # Define coefficients
    a_0 = 0.22

    a_1M = 0.80
    a_1S = 0.25
    b_1S = -1.99

    a_1T = 0.0485
    a_2T = 0.00000586
    a_3T = 0.10
    a_4T = 0.000186

    b_1T = 1.80
    b_2T = 599.99
    b_3T = 7.80
    b_4T = 84.76

    Xval = 1.0 - 4.0 * nu

    p_1S = a_1S * (1.0 + b_1S * Xval)
    p_1T = a_1T * (1.0 + b_1T * Xval)
    p_2T = a_2T * (1.0 + b_2T * Xval)
    p_3T = a_3T * (1.0 + b_3T * Xval)
    p_4T = a_4T * (1.0 + b_4T * Xval)

    kappa2eff2 = kappa2eff * kappa2eff

    Sval = X1 ** 2.0 * chi1_AS + X2 ** 2.0 * chi2_AS

    QM = 1.0 + a_1M*Xval
    QS = 1.0 + p_1S*Sval

    num = 1.0 + p_1T*kappa2eff + p_2T*kappa2eff2
    den = 1.0 + p_3T*kappa2eff + p_4T*kappa2eff2
    QT = num / den

    # dimensionless angular frequency of merger
    Qfit = a_0 * QM * QS * QT

    Momega_merger = nu * Qfit * (2 * PI)

    # TODO check this line against v2 implementation
    # convert from angular frequency to frequency (divide by 2*pi) and then convert from dimensionless frequency to Hz (divide by mtot * LAL_MTSUN_SI)
    fHz_merger = Momega_merger / (2* PI) / ((m1 + m2) * gt)

    return fHz_merger

def _gen_NRTidalv3(f: Array, theta_intrinsic: Array, theta_extrinsic: Array, h0_bbh: Array):

    # Convert masses
    m1, m2, _, _, _, _ = theta_intrinsic
    M = m1 + m2
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    # Compute auxiliary quantities like kappa
    kappa = get_kappa(theta=theta_intrinsic)

    x = (PI * M_s * f) ** (2.0/3.0)

    # Get BBH amplitude and phase
    A_bbh = jnp.abs(h0_bbh)
    psi_bbh = jnp.log(h0_bbh / A_bbh) * 1.j
    
    # Get tidal amplitude and Planck taper
    A_T = get_tidal_amplitude(x, theta_intrinsic, kappa, distance=theta_extrinsic[0])
    f_merger = _get_merger_frequency(theta_intrinsic, kappa)
    A_P = jnp.ones_like(f) - get_planck_taper(f, f_merger)

    # Get tidal phase and PN phase
    psi_tidal = get_tidal_phase(f, theta_intrinsic, kappa)
    psi_PN    = get_tidal_phase_PN(x, theta_intrinsic)
    
    # We employ here the smooth connection between NRTidal and PN post-merger
    taper_phase = planck_taper(f, 1.15 * f_merger, 1.35 * f_merger)
    psi_T = psi_tidal * (jnp.ones_like(f) - taper_phase) + psi_PN * taper_phase
    # psi_T = psi_tidal * taper_phase + psi_PN * (jnp.ones_like(f) - taper_phase)
    
    # Get spin corrections
    psi_SS = get_spin_phase_correction(x, theta_intrinsic)
    h0 = A_P * (A_bbh + A_T) * jnp.exp(1.j * -(psi_bbh + psi_T + psi_SS))

    return h0

def gen_NRTidalv3(f: Array, params: Array, f_ref: float, IMRphenom: str) -> Array:
    """
    Generate NRTidalv3 frequency domain waveform following 1508.07253.
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
    return _gen_NRTidalv3(f, theta_intrinsic, theta_extrinsic, h0_bbh)


def gen_NRTidalv3_hphc(f: Array, params: Array, f_ref: float, IMRphenom: str="IMRPhenomD"):
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
    h0 = gen_NRTidalv3(f, params[:-1], f_ref, IMRphenom=IMRphenom)
    
    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
