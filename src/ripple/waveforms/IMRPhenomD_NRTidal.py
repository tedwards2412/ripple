import jax
import jax.numpy as jnp

# from .IMRPhenomD_utils import (
#     get_coeffs,
#     get_delta0,
#     get_delta1,
#     get_delta2,
#     get_delta3,
#     get_delta4,
#     get_transition_frequencies,
# )

# from .IMRPhenomD_QNMdata import fM_CUT
from ..constants import EulerGamma, gt, m_per_Mpc, C, PI
from ..typing import Array
from ripple import Mc_eta_to_ms

# Import the original IMRPhenomD

def get_inspiral_phase(fM_s: Array, theta: Array, coeffs: Array) -> Array:
    """
    Generate the inspiral phase for the IMRPhenomD_NRTidal waveform. 
    
    FIXME: currently under development. As a starting point, will return simply the OG phase, 
    i.e. without adding contribution from tidal effects

    Args:
        fM_s (Array): Frequencies
        theta (Array): Arguments (see order below)
        coeffs (Array): Coefficients

    Returns:
        Array: Phase of the waveform.
    """
    
    # NEW - lambdas are the tidal deformabilities
    
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    
    # Compute auxiliary mass variables
    
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    
    