"""Small utility script for shared functions between tidal waveforms, especially for NRTidalv2 and NRTidalv3"""

import jax
import jax.numpy as jnp
from ..typing import Array
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI, TWO_PI, MSUN, MRSUN

def universal_relation(coeffs, x):
    return coeffs[0] + coeffs[1] * x + coeffs[2] * (x ** 2) + coeffs[3] * (x ** 3) + coeffs[4] * (x ** 4)

def get_quadparam_octparam(lambda_: float) -> tuple[float, float]:
    
    # Check if lambda is low or not
    is_low_lambda = lambda_ < 1
    
    return jax.lax.cond(is_low_lambda, _get_quadparam_octparam_low, _get_quadparam_octparam_high, lambda_)

def get_kappa(theta):
    m1, m2, _, _, lambda1, lambda2 = theta
    M = m1 + m2

    # Compute X
    X1 = m1 / M
    X2 = m2 / M

    term1 = (1.0 + 12.0 * X2 / X1) * (X1 ** 5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2 ** 5.0) * lambda2
    
    kappa = (3./13.) * (term1 + term2)
    
    return kappa 

def get_amp0_lal(M, distance):
    """amp0 as defined by LAL in LALSimIMRPhenomD, line 331. 
    
    Args:
        distance: Distance to source in meters

    Returns:
        float: amp0 defined by LALSimIMRPhenomD, line 331. 
    """
    
    amp0 = 2. * jnp.sqrt(5. / (64. * PI)) * M * MRSUN * M * gt / distance
    
    return amp0

def _get_quadparam_octparam_low(lambda_: float) -> tuple[float, float]:
    """
    Computes quadparameter, see eq (28) of NRTidalv2 paper and also LALSimUniversalRelations.c of lalsuite
    
    Version for lambdas smaller than 1.
    
    LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
    Args:
        lambda_: tidal deformability

    Returns:
        quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
        octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
    """
    
    # Coefficients of universal relation
    quad_coeffs = [0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4]
    oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]
    
    # Extension of the fit in the range lambda2 = [0,1.] so that the BH limit is enforced, lambda2bar->0 gives quadparam->1. and the junction with the universal relation is smooth, of class C2
    quadparam = 1. + lambda_ * (0.427688866723244 + lambda_ * (-0.324336526985068 + lambda_ * 0.1107439432180572))
    log_quadparam = jnp.log(quadparam)
        
    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)

    # Get rid of log and remove 1 for BBH baseline
    # quadparam = jnp.exp(log_quadparam) - 1
    octparam = jnp.exp(log_octparam)

    return quadparam, octparam

def _get_quadparam_octparam_high(lambda_: float) -> tuple[float, float]:
    """
    Computes quadparameter, see eq (28) of NRTidalv2 paper and also LALSimUniversalRelations.c of lalsuite
    
    Version for lambdas greater than 1.
    
    LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
    Args:
        lambda_: tidal deformability

    Returns:
        quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
        octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
    """
    
    # Coefficients of universal relation
    quad_coeffs = [0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4]
    oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]
        
    # High lambda (above 1): use universal relation
    log_lambda = jnp.log(lambda_)
    log_quadparam = universal_relation(quad_coeffs, log_lambda)
    
    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)

    quadparam = jnp.exp(log_quadparam)
    octparam = jnp.exp(log_octparam)

    return quadparam, octparam

def planck_taper(t: Array, t1: float, t2: float) -> Array:
    """
    As taken from Lalsuite
    Args:
        t:
        t1:
        t2:

    Returns:
        Planck taper
    """

    # Middle part: transition formula for Planck taper
    middle = 1. / (jnp.exp((t2 - t1)/(t - t1) + (t2 - t1)/(t - t2)) + 1.)

    taper = jnp.heaviside(t1 - t, 1) * jnp.zeros_like(t) \
            + jnp.heaviside(t - t1, 1) * jnp.heaviside(t2 - t, 1) * middle \
            + jnp.heaviside(t - t2, 1) * jnp.ones_like(t)

    return taper

def get_planck_taper(f: Array, f_merger: float):
    
    f_start = f_merger
    f_end   = 1.2 * f_merger

    A_P = planck_taper(f, f_start, f_end)

    return A_P

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