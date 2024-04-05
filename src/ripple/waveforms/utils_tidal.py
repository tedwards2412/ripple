"""Small utility script for shared functions between tidal waveforms, especially for NRTidalv2"""

import jax
import jax.numpy as jnp
from ..typing import Array
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI, TWO_PI, MSUN, MRSUN

def universal_relation(coeffs: Array, x: float):
    """Applies the general formula of a universal relationship, which is a quartic polynomial.

    Args:
        coeffs (Array): Array of coefficients for the quartic polynomial, starting from the constant term and going to the fourth order.
        x (float): Variable of quartic polynomial

    Returns:
        float: Result of universal relation
    """
    return coeffs[0] + coeffs[1] * x + coeffs[2] * (x ** 2) + coeffs[3] * (x ** 3) + coeffs[4] * (x ** 4)

def get_quadparam_octparam(lambda_: float) -> tuple[float, float]:
    """Compute the quadrupole and octupole parameter by checking the value of lambda and choosing the right subroutine.
    If lambda is smaller than 1, we make use of the fit formula as given by the LAL source code. Otherwise, we rely on the equations of
    the NRTidalv2 paper to get these parameters.

    Args:
        lambda_ (float): Tidal deformability of object.

    Returns:
        tuple[float, float]: Quadrupole and octupole parameters.
    """
    
    # Check if lambda is low or not, and choose right subroutine
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
    oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]
    
    # Extension of the fit in the range lambda2 = [0,1.] so that the BH limit is enforced, lambda2bar->0 gives quadparam->1. and the junction with the universal relation is smooth, of class C2
    quadparam = 1. + lambda_ * (0.427688866723244 + lambda_ * (-0.324336526985068 + lambda_ * 0.1107439432180572))
    log_quadparam = jnp.log(quadparam)
        
    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)
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

def get_kappa(theta: Array) -> float:
    """Computes the tidal deformability parameter kappa according to equation (8) of the NRTidalv2 paper.

    Args:
        theta (Array): Intrinsic parameters m1, m2, chi1, chi2, lambda1, lambda2

    Returns:
        float: kappa_eff^T from equation (8) of NRTidalv2 paper.
    """
    
    # Auxiliary variables
    m1, m2, _, _, lambda1, lambda2 = theta
    M = m1 + m2
    X1 = m1 / M
    X2 = m2 / M

    # Get kappa
    term1 = (1.0 + 12.0 * X2 / X1) * (X1 ** 5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2 ** 5.0) * lambda2
    kappa = (3./13.) * (term1 + term2)
    
    return kappa 

def get_amp0_lal(M: float, distance: float):
    """Get the amp0 prefactor as defined in LAL in LALSimIMRPhenomD, line 331. 

    Args:
        M (float): Total mass in solar masses
        distance (float): Distance to the source in Mpc.

    Returns:
        float: amp0 from LAL.
    """
    amp0 = 2. * jnp.sqrt(5. / (64. * PI)) * M * MRSUN * M * gt / distance
    return amp0


# def planck_taper(t: Array, t1: float, t2: float) -> Array:
#     """Function to compute the Planck taper window between t1 and t2.

#     Args:
#         t (Array): Times at which the Planck taper has to be computed.
#         t1 (float): Start of Planck taper.
#         t2 (float): End of Planck taper.

#     Returns:
#         Array: Planck taper A_P
#     """

#     # Planck taper consists of three parts:
#     begin = jnp.zeros_like(t)
#     end = jnp.ones_like(t)
#     middle = 1. / (jnp.exp((t2 - t1)/(t - t1) + (t2 - t1)/(t - t2)) + 1.)

#     # Build the taper from the three parts with step functions
#     taper = jnp.heaviside(t1 - t, 1) * begin \
#             + jnp.heaviside(t - t1, 1) * jnp.heaviside(t2 - t, 1) * middle \
#             + jnp.heaviside(t - t2, 1) * end

#     return taper

# def get_planck_taper(f: Array, f_merger: float) -> Array:
#     """Get the Planck taper for the purpose of NRTidalv2, namely by applying it in the window [f_merger, 1.2f_merger]

#     Args:
#         f (Array): Frequency grid at which the GW is being computed.
#         f_merger (float): Merger frequency in Hz.

#     Returns:
#         Array: Planck taper for NRTidalv2
#     """
#     return 1.0 - planck_taper(f, f_merger, 1.2 * f_merger)

### The code below to compute the Planck taper is obtained from gwfast (https://github.com/CosmoStatGW/gwfast/blob/ccde00e644682639aa8c9cbae323e42718fd61ca/gwfast/waveforms.py#L1332)
@jax.custom_jvp
def get_planck_taper(x, y):
    # Terminate the waveform at 1.2 times the merger frequency
    a=1.2
    yp = a*y
    planck_taper = jnp.where(x < y, 1., jnp.where(x > yp, 0., 1. - 1./(jnp.exp((yp - y)/(x - y) + (yp - y)/(x - yp)) + 1.)))

    return planck_taper

def get_planck_taper_der(x,y):
    # Terminate the waveform at 1.2 times the merger frequency
    a = 1.2
    yp = a*y
    tangent_out = jnp.where(x < y, 0., jnp.where(x > yp, 0., jnp.exp((yp - y)/(x - y) + (yp - y)/(x - yp))*((-1.+a)/(x-y) + (-1.+a)/(x-yp) + (-y+yp)/((x-y)**2) + 1.2*(-y+yp)/((x-yp)**2))/((jnp.exp((yp - y)/(x - y) + (yp - y)/(x - yp)) + 1.)**2)))
    tangent_out = jnp.nan_to_num(tangent_out)
    return tangent_out
get_planck_taper.defjvps(None, lambda y_dot, primal_out, x, y: get_planck_taper_der(x,y) * y_dot)
        

def get_tidal_amplitude(x: Array, theta: Array, kappa: float, distance: float =1):
    """Get the tidal amplitude corrections as given in equation (24) of the NRTidal paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)
        kappa (float): Tidal parameter kappa
        distance (float, optional): Distance to the source in Mpc.

    Returns:
        Array: Tidal amplitude corrections A_T from NRTidalv2 paper.
    """
    
    # Mass variables
    m1, m2, _, _, _, _ = theta 
    M = m1 + m2
    m1_s = m1 * gt
    m2_s = m2 * gt
    
    # Convert distance to meters
    distance *= m_per_Mpc
    
    # Pade approximant
    n1   = 4.157407407407407
    n289 = 2519.111111111111
    d    = 13477.8073677
    num = 1.0 + n1 * x + n289 * x ** 2.89
    den = 1.0 + d * x ** 4.
    poly = num / den
    
    # Prefactors are taken from lal source code
    prefac = - 9.0 * kappa
    ampT = prefac * x ** (13. / 4.) * poly
    amp0 = get_amp0_lal(M, distance)
    ampT *= amp0 * 2 * jnp.sqrt(PI / 5)
    
    return ampT 

def _get_spin_induced_quadrupole_phase_coeff(lambda_: float, mass: float) -> float:
    """Compute the quantity from equation (11) from http://arxiv.org/abs/1503.05405

    Args:
        lambda_ (float): Tidal deformability of object
        mass (float): Mass of object in solar masses

    Returns:
        float: a(m) 
    """
    is_low_lambda = lambda_ < 1
    return jax.lax.cond(is_low_lambda, _get_spin_induced_quadrupole_phase_coeff_low, _get_spin_induced_quadrupole_phase_coeff_high, lambda_)
        

def _get_spin_induced_quadrupole_phase_coeff_low(lambda_: float) -> float:
    """Compute the quantity from equation (11) from http://arxiv.org/abs/1503.05405

    Args:
        lambda_ (float): Tidal deformability of object
        mass (float): Mass of object in solar masses

    Returns:
        float: a(m) as defined in https://git.ligo.org/thibeau.wouters/bonz_marlinde/-/blob/main/UniversalRelation/FitUniversalRelation.ipynb?ref_type=heads (fit)
    """
    coeffs = jnp.array([1.0, 0.32812816173650255, -0.16209486695933736, 0.05219418106960124, -0.006406318945489099])
    a = universal_relation(coeffs, lambda_)
    return a

def _get_spin_induced_quadrupole_phase_coeff_high(lambda_: float) -> float:
    """Compute the quantity from equation (11) from http://arxiv.org/abs/1503.05405

    Args:
        lambda_ (float): Tidal deformability of object
        mass (float): Mass of object in solar masses

    Returns:
        float: a(m) as defined in equation (11) of http://arxiv.org/abs/1503.05405
    """
    
    # Auxiliary parameter:
    # TODO what if lambda is zero or negative?
    x = jnp.log(lambda_)
    coeffs = jnp.array([0.194, 0.0936, 0.0474, -4.21e-3, 1.23e-4])
    
    ln_a = universal_relation(coeffs, x)
    a = jnp.exp(ln_a)
        
    return a

def get_spin_induced_quadrupole_phase(v: Array, theta: Array) -> Array:
    """Computes the contribution to the phase from the spin-induced quadrupole moment

    Args:
        x (Array): Array of velocities, that is, v = (pi M f)^(1/3)
        theta (Array): Array of parameters (m1, m2, chi1, chi2, lambda1, lambda2)   

    Returns:
        Array: Phase contribution from spin-induced quadrupole moment
    """
    # See http://arxiv.org/abs/1503.05405, around eq (11)
    
    # Get parameters and auxiliary variables
    m1, m2, chi1, chi2, lambda1, lambda2 = theta 
    M  = m1 + m2
    X1 = m1 / M
    X2 = m2 / M
    eta = m1 * m2 / (M ** 2.0)
    
    # Compute the spin-induced quadrupole phase (NOTE this is assuming aligned spin)
    a1 = _get_spin_induced_quadrupole_phase_coeff(lambda1, m1)
    a2 = _get_spin_induced_quadrupole_phase_coeff(lambda2, m2)
    sigma_qm_1 = 5 * a1 * (X1 ** 2) * chi1 ** 2
    sigma_qm_2 = 5 * a2 * (X2 ** 2) * chi2 ** 2
    
    sigma_qm = sigma_qm_1 + sigma_qm_2
    psi_qm = - (30 / (128 * eta)) * sigma_qm / v
    
    return psi_qm