"""Small utility script for shared functions between all tidal waveforms"""

import jax
import jax.numpy as jnp

def universal_relation(coeffs, x):
    return coeffs[0] + coeffs[1] * x + coeffs[2] * (x ** 2) + coeffs[3] * (x ** 3) + coeffs[4] * (x ** 4)

def get_quadparam_octparam(lambda_: float) -> tuple[float, float]:
    
    # Check if lambda is low or not
    is_low_lambda = lambda_ < 1
    
    return jax.lax.cond(is_low_lambda, _get_quadparam_octparam_low, _get_quadparam_octparam_high, lambda_)


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
    octparam = jnp.exp(log_octparam) - 1

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

    # Get rid of log and remove 1 for BBH baseline
    quadparam = jnp.exp(log_quadparam) # - 1 ### correct?
    octparam = jnp.exp(log_octparam) - 1

    return quadparam, octparam
