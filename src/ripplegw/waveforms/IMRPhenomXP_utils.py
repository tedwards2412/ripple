import jax
import jax.numpy as jnp
from ripplegw import Mc_eta_to_ms

from typing import Tuple
from ..constants import gt, MSUN, G, C
import numpy as np
from .IMRPhenomD import Phase as PhDPhase
from .IMRPhenomD import Amp as PhDAmp
from .IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
    EradRational0815,
    FinalSpin0815_s,
)
from ..typing import Array
from .IMRPhenomD_QNMdata import QNMData_a, QNMData_fRD, QNMData_fdamp
from .spherical_harmonics import *
from ..gsl_ellint import ellipfinc, gsl_sf_elljac_e


# helper functions for LALtoPhenomP:
def IMRPhenomX_rotate_y(v, theta):
    """Rotate vector(s) v about the y-axis by angle theta (radians)."""
    R_y = jnp.array([
        [ jnp.cos(theta), 0.0, jnp.sin(theta)],
        [ 0.0,            1.0, 0.0           ],
        [-jnp.sin(theta), 0.0, jnp.cos(theta)]
    ])
    return R_y @ v

def IMRPhenomX_rotate_z(v, theta):
    """Rotate vector(s) v about the z-axis by angle theta (radians)."""
    R_z = jnp.array([
        [ jnp.cos(theta), -jnp.sin(theta), 0.0],
        [ jnp.sin(theta),  jnp.cos(theta), 0.0],
        [ 0.0,             0.0,            1.0]
    ])
    return R_z @ v

###

def IMRPhenomX_Return_phi_zeta_costhetaL_MSA(
    v,  ## velocity
    pWF,  ## IMRPhenomX waveform struct
    pPrec  ## IMRPhenomX precession struct
    ):  ## has to output a jnp array
    
    L_norm = pWF['eta'] / v
    
    J_norm = IMRPhenomX_JNorm_MSA(L_norm,pPrec)  
    
    ## J_norm = jax.lax.cond(pPrec["useMSA"], )
    
    L_norm3PN = IMRPhenomX_L_norm_3PN_of_v(v, v*v, L_norm, pPrec)   ## for 223
    
    J_norm3PN = IMRPhenomX_JNorm_MSA(L_norm3PN,pPrec)
    
    vRoots    = IMRPhenomX_Return_Roots_MSA(L_norm,J_norm,pPrec)  ## return jnp.array
    
    pPrec["S32"]  = vRoots[0]
    pPrec["Smi2"] = vRoots[1]
    pPrec["Spl2"] = vRoots[2]
    
    pPrec["Spl2mSmi2"]   = pPrec["Spl2"] - pPrec["Smi2"]
    pPrec["Spl2pSmi2"]   = pPrec["Spl2"] + pPrec["Smi2"]
    pPrec["Spl"]         = jnp.sqrt(pPrec["Spl2"])
    pPrec["Smi"]           = jnp.sqrt(pPrec["Smi2"])
    
    SNorm = IMRPhenomX_Return_SNorm_MSA(v,pPrec)  
    pPrec["S_norm"]      = SNorm
    pPrec["S_norm_2"]    = SNorm * SNorm
    
    ''' Get phiz_0_MSA and zeta_0_MSA '''
    condition = jnp.atleast_1d(jnp.fabs(pPrec["Smi2"] - pPrec["Spl2"]) > 1.e-5)
    ## this check is needed because of broadcasting if condition is N-dimensional 
    # (N,) cannot broadcast to (N,3)
    condition = condition[..., None] if condition.ndim == 1 else condition  
    
    vMSA = jnp.where(condition, 
                        IMRPhenomX_Return_MSA_Corrections_MSA(v, L_norm, J_norm, pPrec),  ## return v.shape[0]x3 dimensional jnp.array
                        jnp.zeros((jnp.atleast_1d(v).shape[0], 3)),
                    )   
        
    phiz_MSA     = vMSA[:,0]
    zeta_MSA     = vMSA[:,1]

    phiz         = jnp.atleast_1d(IMRPhenomX_Return_phiz_MSA(v,J_norm,pPrec))  
    zeta         = jnp.atleast_1d(IMRPhenomX_Return_zeta_MSA(v,pPrec))
    cos_theta_L      = jnp.atleast_1d(IMRPhenomX_costhetaLJ(L_norm3PN,J_norm3PN,SNorm)) 

    vout = jnp.stack([
        phiz + phiz_MSA,
        zeta + zeta_MSA,
        cos_theta_L
    ], axis=-1)
    
    return vout

def WignerdCoefficients_cosbeta(
    cos_beta        ## cos(beta) 
    ):
    ''' Note that the results here are indeed always non-negative '''
    cos_beta_half = + jnp.sqrt( jnp.fabs(1.0 + cos_beta) / 2.0 )
    sin_beta_half = + jnp.sqrt( jnp.fabs(1.0 - cos_beta) / 2.0 )
 
    return cos_beta_half, sin_beta_half

def WignerdCoefficients(
  v,        # Cubic root of (Pi * Frequency (geometric))
  pWF,     # IMRPhenomX waveform struct   
  pPrec   # IMRPhenomX precession struct 
):
    # This implementation is different respect to the one in Pv2_utils
    
    # Orbital angular momentum L
    L = XLALSimIMRPhenomXLPNAnsatz(
        v,
        pWF['eta'] / v,
        pPrec['L0'], pPrec['L1'], pPrec['L2'], pPrec['L3'], pPrec['L4'],
        pPrec['L5'], pPrec['L6'], pPrec['L7'], pPrec['L8'], pPrec['L8L']
    )

    # We ignore the sign of L + SL below:
    # s = S_perp / (L + SL)
    denom = L + pPrec['SL']
    s = pPrec['Sperp'] / denom
    s2 = s * s

    cos_beta = jnp.sign(denom) / jnp.sqrt(1.0 + s2)

    # cos(beta/2) and sin(beta/2) 
    cos_beta_half = jnp.sqrt(jnp.abs(1.0 + cos_beta) / 2.0)
    sin_beta_half = jnp.sqrt(jnp.abs(1.0 - cos_beta) / 2.0)

    return cos_beta_half, sin_beta_half

def IMRPhenomX_costhetaLJ(
    L_norm: float, 
    J_norm: float, 
    S_norm: float
    ) -> float:
    costhetaLJ = 0.5 * (J_norm**2 + L_norm**2 - S_norm**2) / L_norm * J_norm

    # Clamp the value to the interval [-1.0, 1.0]
    costhetaLJ = jnp.clip(costhetaLJ, -1.0, 1.0)

    return costhetaLJ

def IMRPhenomX_Return_zeta_MSA(
    v: float, 
    pPrec
    ) -> float:
    invv = 1.0 / v
    invv2 = invv * invv
    invv3 = invv * invv2
    v2 = v * v
    logv = jnp.log(v)

    # Compute zeta using precession coefficients
    zeta_out = pPrec["eta"] * (
        pPrec["Omegazeta0_coeff"] * invv3 +
        pPrec["Omegazeta1_coeff"] * invv2 +
        pPrec["Omegazeta2_coeff"] * invv +
        pPrec["Omegazeta3_coeff"] * logv +
        pPrec["Omegazeta4_coeff"] * v +
        pPrec["Omegazeta5_coeff"] * v2
    ) + pPrec["zeta_0"]

    # Replace NaNs with 0 using jnp.nan_to_num
    zeta_out = jnp.nan_to_num(zeta_out, nan=0.0)

    return zeta_out


def IMRPhenomX_Return_phiz_MSA(
    v: float, 
    JNorm: float, 
    pPrec
    ) -> float:
    
    invv = 1.0 / v
    invv2 = invv * invv
    LNewt = pPrec["eta"] / v

    c1 = pPrec["c1"]
    c12 = c1 * c1

    SAv2 = pPrec["SAv2"]
    SAv = pPrec["SAv"]
    invSAv = pPrec["invSAv"]
    invSAv2 = pPrec["invSAv2"]

    # These are log functions defined in Eq. D27 and D28 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    log1 = jnp.log(jnp.abs(c1 + JNorm * pPrec["eta"] + pPrec["eta"] * LNewt))
    log2 = jnp.log(jnp.abs(c1 + JNorm * SAv * v + SAv2 * v))

    # Eq. D22-D27 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    phiz_0_coeff = (JNorm * pPrec["inveta"]**4) * (
        0.5 * c12 - (c1 * pPrec["eta2"] * invv) / 6.0 - (SAv2 * pPrec["eta2"]) / 3.0 - (pPrec["eta4"] * invv2) / 3.0
    ) - (0.5 * c1 * pPrec["inveta"]) * (
        c12 * pPrec["inveta"]**4 - SAv2 * pPrec["inveta"]**2
    ) * log1

    phiz_1_coeff = (
        -0.5 * JNorm * pPrec["inveta"]**2 * (c1 + pPrec["eta"] * LNewt)
        + 0.5 * pPrec["inveta"]**3 * (c12 - pPrec["eta2"] * SAv2) * log1
    )

    phiz_2_coeff = -JNorm + SAv * log2 - c1 * log1 * pPrec["inveta"]

    phiz_3_coeff = JNorm * v - pPrec["eta"] * log1 + c1 * log2 * invSAv

    phiz_4_coeff = (
        0.5 * JNorm * invSAv2 * v * (c1 + v * SAv2)
        - 0.5 * invSAv2 * invSAv * (c12 - pPrec["eta2"] * SAv2) * log2
    )

    phiz_5_coeff = (
        -JNorm * v * (
            0.5 * c12 * invSAv2 * invSAv2
            - c1 * v * invSAv2 / 6.0
            - v * v / 3.0
            - pPrec["eta2"] * invSAv2 / 3.0
        )
        + 0.5 * c1 * invSAv2 * invSAv2 * invSAv * (c12 - pPrec["eta2"] * SAv2) * log2
    )

    # Eq. 66 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
 
    # \phi_{z,-1} = \sum^5_{n=0} <\Omega_z>^(n) \phi_z^(n) + \phi_{z,-1}^0
 
    # Note that the <\Omega_z>^(n) are given by pPrec->Omegazn_coeff's as in Eqs. D15-D20
    phiz_out = (
        phiz_0_coeff * pPrec["Omegaz0_coeff"]
        + phiz_1_coeff * pPrec["Omegaz1_coeff"]
        + phiz_2_coeff * pPrec["Omegaz2_coeff"]
        + phiz_3_coeff * pPrec["Omegaz3_coeff"]
        + phiz_4_coeff * pPrec["Omegaz4_coeff"]
        + phiz_5_coeff * pPrec["Omegaz5_coeff"]
        + pPrec["phiz_0"]
    )

    # Ensure no NaN (replace with 0.0 if NaN)
    phiz_out = jnp.nan_to_num(phiz_out, nan=0.0)

    return phiz_out


def IMRPhenomX_Return_MSA_Corrections_MSA(
    v, 
    LNorm, 
    JNorm, 
    pPrec
    ):
    
    v2 = v * v

    # Sets c0, c2 and c4 in pPrec as per Eq. B6-B8 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    c_vec = IMRPhenomX_Return_Constants_c_MSA(v, JNorm, pPrec)
    # Sets d0, d2 and d4 in pPrec as per Eq. B9-B11 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    d_vec = IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, pPrec)  

    c0, c2, c4 = c_vec
    d0, d2, d4 = d_vec

    two_d0 = 2.0 * d0
    
    # Eq. B20 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    sd = jnp.sqrt(jnp.abs(d2 * d2 - 4.0 * d0 * d4))

    # Eq. F20-21 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    A_theta_L = 0.5 * ((JNorm / LNorm) + (LNorm / JNorm) - (pPrec["Spl2"] / (JNorm * LNorm)))
    B_theta_L = 0.5 * pPrec["Spl2mSmi2"] / (JNorm * LNorm)

    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd

    nc = nc_num / nc_denom
    nd = nc_denom / two_d0

    sqrt_nc = jnp.sqrt(jnp.abs(nc))
    sqrt_nd = jnp.sqrt(jnp.abs(nd))

    psi = IMRPhenomX_Return_Psi_MSA(v, v2, pPrec) + pPrec["psi0"]  
    psi_dot = IMRPhenomX_Return_Psi_dot_MSA(v, pPrec) 

    tan_psi = jnp.tan(psi)
    atan_psi = jnp.arctan(tan_psi)

    C1 = -0.5 * (c0 / d0 - 2.0 * (c0 + c2 + c4) / nc_num)
    C2num = (c0 * (-2.0 * d0 * d4 + d2 * d2 + d2 * d4) -
             c2 * d0 * (d2 + 2.0 * d4) +
             c4 * d0 * (two_d0 + d2))
    C2den = 2.0 * d0 * sd * (d0 + d2 + d4)
    C2 = C2num / C2den

    Cphi = C1 + C2
    Dphi = C1 - C2

    def compute_Cphi_term():
        
        return jnp.abs((
            (c4 * d0 * ((2 * d0 + d2) + sd) -
                c2 * d0 * ((d2 + 2.0 * d4) - sd) -
                c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - 
                sd))) / C2den) * (sqrt_nc / (nc - 1.0)) * (atan_psi - jnp.arctan(sqrt_nc * tan_psi))) / psi_dot
        
    def compute_Dphi_term():
            return jnp.abs((
                (-c4 * d0 * ((2 * d0 + d2) - sd) +
                 c2 * d0 * ((d2 + 2.0 * d4) + sd) -
                 c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))) / C2den
            ) * (sqrt_nd / (nd - 1.0)) * (atan_psi - jnp.arctan(sqrt_nd * tan_psi))) / psi_dot

    phiz_0_MSA_Cphi_term = jnp.where(nc == 1.0, 0.0, compute_Cphi_term())
    phiz_0_MSA_Dphi_term = jnp.where(nd == 1.0, 0.0, compute_Dphi_term())

    vMSA_x = phiz_0_MSA_Cphi_term + phiz_0_MSA_Dphi_term

    #####  restart from here
    vMSA_y = A_theta_L * vMSA_x + 2.0 * B_theta_L * d0 * (
                phiz_0_MSA_Cphi_term / (sd - d2) - phiz_0_MSA_Dphi_term / (sd + d2))

    vMSA_x = jnp.where(jnp.isnan(vMSA_x), 0.0, vMSA_x)
    vMSA_y = jnp.where(jnp.isnan(vMSA_y), 0.0, vMSA_y)

    return jnp.stack([vMSA_x, vMSA_y, jnp.zeros_like(vMSA_x)], axis=-1)

def IMRPhenomX_JNorm_MSA(LNorm, pPrec):
    JNorm2 = (LNorm * LNorm + 2.0 * LNorm * pPrec['c1_over_eta'] + pPrec['SAv2'])
    return jnp.sqrt(JNorm2)

def IMRPhenomX_Return_SNorm_MSA(v, pPrec):

    v2 = v * v

    cancel_condition = jnp.abs(pPrec['Smi2'] - pPrec['Spl2']) < 1e-5

    def sn_zero():
        sn = jnp.array(0.0)
        return sn

    def sn_jacobi():
        # Equation 25 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
        m = (pPrec['Smi2'] - pPrec['Spl2']) / (pPrec['S32'] - pPrec['Spl2'])

        psi = IMRPhenomX_psiofv(
            v, v2,
            pPrec['psi0'], pPrec['psi1'], pPrec['psi2'],
            pPrec
        )

        # Jacobi elliptic functions
        sn, cn, dn = gsl_sf_elljac_e(psi, m)
        return sn

    sn = jnp.where(cancel_condition, sn_zero(), sn_jacobi())

    # Equation 23 of Chatziioannou et al, PRD 95, 104004, (2017), arXiv:1703.03967
    SNorm2 = pPrec['Spl2'] + (pPrec['Smi2'] - pPrec['Spl2']) * sn * sn

    return jnp.sqrt(SNorm2)
    
def IMRPhenomX_L_norm_3PN_of_v(v: float, v2: float, L_norm: float, pPrec) -> float:
    cL = pPrec['constants_L']  # shorthand
    return L_norm * 1.0 + v2 * (
        cL[0] + v * cL[1] + v2 * (
            cL[2] + v * cL[3] + v2 * cL[4]
        )
    )
    
def IMRPhenomX_Return_Roots_MSA(LNorm, JNorm, pPrec):
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec)  
    B, C, D = vBCD[0], vBCD[1], vBCD[2]

    B2 = B * B
    B3 = B2 * B
    BC = B * C

    p = C - B2 / 3.0
    qc = (2.0 / 27.0) * B3 - BC / 3.0 + D

    sqrtarg = jnp.sqrt(-p / 3.0)
    acosarg = 1.5 * qc / (p * sqrtarg)
    acosarg = jnp.clip(acosarg, -1.0, 1.0)

    theta = jnp.arccos(acosarg) / 3.0
    cos_theta = jnp.cos(theta)
    
    vector_condition = jnp.logical_or(jnp.isnan(theta),
                                                   (jnp.isnan(sqrtarg)))
    scalar_condition = jnp.logical_or.reduce(jnp.array([(pPrec['dotS1Ln'] == 1.0),
                                                   (pPrec['dotS2Ln'] == 1.0),
                                                   (pPrec['dotS1Ln'] == -1.0),
                                                   (pPrec['dotS2Ln'] == -1.0),
                                                   (pPrec['S1_norm_2'] == 0.0),
                                                   (pPrec['S2_norm_2'] == 0.0)]))
    invalid_case = jnp.logical_or(vector_condition, scalar_condition)

    def roots_when_valid():
        tmp1 = 2.0 * sqrtarg * jnp.cos(theta - 4.0 * jnp.pi / 3.0) - B / 3.0
        tmp2 = 2.0 * sqrtarg * jnp.cos(theta - 2.0 * jnp.pi / 3.0) - B / 3.0
        tmp3 = 2.0 * sqrtarg * cos_theta - B / 3.0

        tmp4 = jnp.maximum(jnp.maximum(tmp1, tmp2), tmp3)
        tmp5 = jnp.minimum(jnp.minimum(tmp1, tmp2), tmp3)

        tmp6 = jnp.where(
            (tmp4 - tmp3 > 0.0) & (tmp5 - tmp3 < 0.0),
            tmp3,
            jnp.where((tmp4 - tmp1 > 0.0) & (tmp5 - tmp1 < 0.0), tmp1, tmp2)
        )

        S32 = tmp5
        Smi2 = jnp.abs(tmp6)
        Spl2 = jnp.abs(tmp4)
        return jnp.array([S32, Smi2, Spl2])

    def roots_when_invalid():
        Smi2 = pPrec['S_0_norm']**2 * jnp.ones_like(LNorm)
        Spl2 = Smi2 + 1e-9
        S32 = jnp.zeros_like(LNorm)
        return jnp.array([S32, Smi2, Spl2])

    roots_array = jnp.where(
        jnp.atleast_1d(invalid_case),
        roots_when_invalid(),
        roots_when_valid()
    )

    return roots_array

def IMRPhenomX_Return_Constants_c_MSA(v, JNorm, pPrec):
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v6 = v3 * v3
    JNorm2 = JNorm * JNorm
    Seff = pPrec['Seff']


    x = JNorm * (
        0.75 * (1.0 - Seff * v) * v2 * (
            pPrec['eta3']
            + 4.0 * pPrec['eta3'] * Seff * v
            - 2.0 * pPrec['eta'] * (
                JNorm2 - pPrec['Spl2'] + 2.0 * (pPrec['S1_norm_2'] - pPrec['S2_norm_2']) * pPrec['delta_qq']
            ) * v2
            - 4.0 * pPrec['eta'] * Seff * (JNorm2 - pPrec['Spl2']) * v3
            + (JNorm2 - pPrec['Spl2']) ** 2 * v4 * pPrec['inveta']
        )
    )

    y = JNorm * (
        -1.5 * pPrec['eta'] * (pPrec['Spl2'] - pPrec['Smi2'])
        * (1.0 + 2.0 * Seff * v - (JNorm2 - pPrec['Spl2']) * v2 * pPrec['inveta']**2)
        * (1.0 - Seff * v) * v4
    )

    z = JNorm * (
        0.75 * pPrec['inveta'] * (pPrec['Spl2'] - pPrec['Smi2']) ** 2
        * (1.0 - Seff * v) * v6
    )

    return jnp.array([x, y, z])

def IMRPhenomX_Return_Constants_d_MSA(LNorm, JNorm, pPrec):
    LNorm2 = LNorm * LNorm
    JNorm2 = JNorm * JNorm

    x = - (JNorm2 - (LNorm + pPrec['Spl'])) ** 2 * (JNorm2 - (LNorm - pPrec['Spl'])) ** 2

    y = -2.0 * (pPrec['Spl2'] - pPrec['Smi2']) * (JNorm2 + LNorm2 - pPrec['Spl2'])

    z = -(pPrec['Spl2'] - pPrec['Smi2']) ** 2

    return jnp.array([x, y, z])

def IMRPhenomX_Return_Psi_MSA(v, v2, pPrec):
    return -0.75 * pPrec['g0'] * pPrec['delta_qq'] * (1.0 + pPrec['psi1'] * v + pPrec['psi2'] * v2) / (v2 * v)

def IMRPhenomX_Return_Psi_dot_MSA(v, pPrec):
    v2 = v * v

    A_coeff = -1.5 * v2 * v2 * v2 * (1.0 - v * pPrec['Seff']) * jnp.sqrt(pPrec['inveta'])
    psi_dot = 0.5 * A_coeff * jnp.sqrt(pPrec['Spl2'] - pPrec['S32'])

    return psi_dot

def IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(LNorm, JNorm, pPrec):
    JNorm2 = JNorm * JNorm
    LNorm2 = LNorm * LNorm

    S1Norm2 = pPrec['S1_norm_2']
    S2Norm2 = pPrec['S2_norm_2']
    q       = pPrec['qq']
    eta     = pPrec['eta']
    delta   = pPrec['delta_qq']
    deltaSq = delta * delta
    Seff    = pPrec['Seff']

    J2mL2   = JNorm2 - LNorm2
    J2mL2Sq = J2mL2 * J2mL2

    # B coefficient (Eq. B2)
    B_coeff = ((LNorm2 + S1Norm2) * q +
               2.0 * LNorm * Seff -
               2.0 * JNorm2 -
               S1Norm2 - S2Norm2 +
               (LNorm2 + S2Norm2) / q)

    # C coefficient (Eq. B3)
    C_coeff = (J2mL2Sq -
               2.0 * LNorm * Seff * J2mL2 -
               2.0 * ((1.0 - q) / q) * LNorm2 * (S1Norm2 - q * S2Norm2) +
               4.0 * eta * LNorm2 * Seff * Seff -
               2.0 * delta * (S1Norm2 - S2Norm2) * Seff * LNorm +
               2.0 * ((1.0 - q) / q) * (q * S1Norm2 - S2Norm2) * JNorm2)

    # D coefficient (Eq. B4)
    D_coeff = (((1.0 - q) / q) * (S2Norm2 - q * S1Norm2) * J2mL2Sq +
               deltaSq * (S1Norm2 - S2Norm2)**2 * LNorm2 / eta +
               2.0 * delta * LNorm * Seff * (S1Norm2 - S2Norm2) * J2mL2)

    return jnp.array([B_coeff, C_coeff, D_coeff])

def IMRPhenomXGetAndSetPrecessionVariables(pWF, params):
    
    m1_SI = pWF['m1']*gt
    m2_SI = pWF['m2']*gt
    
    chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = pWF['chi1x'], pWF['chi1y'], pWF['chi1z'], pWF['chi2x'], pWF['chi2y'], pWF['chi2z']
    
    pPrec = {}
    
    ## Here we're only considering the default setting, where the expansion order for MSA correction is 5
    ## pPrec['ExpansionOrder'] = XLALSimInspiralWaveformParamsLookupPhenomXPExpansionOrder(params)  ## (TODO)
    pPrec['ExpansionOrder'] = 5
    
    ## allow for conditional disabling of precession multibanding given mass ratio and opening angle  ## (TODO) -> line 137 of x_precession.C
    pPrec['conditionalPrecMBand'] = 0
    ## default value for PrecThresholdMband according to arXiv:2004.06503v2, Table VI
    pPrec['PrecThresholdMband'] = 1e-3
    
    Mtot_SI = m1_SI + m2_SI  
    # Normalize masses
    m1 = m1_SI / Mtot_SI
    m2 = m2_SI / Mtot_SI
    M = m1 + m2

    # Mass ratio and symmetric mass ratio
    q   = m1 / m2
    eta = pWF['eta']
    pPrec['eta'] = eta
    pPrec['eta2'] = eta**2
    pPrec['eta3'] = eta**3
    pPrec['eta4'] = eta**4
    pPrec['inveta'] = 1 / eta
    
    m1_2 = m1**2
    m2_2 = m2**2
    
    ### pWF needs to be a dict??
    ## TODO: check how is delta stored in pWF
    delta     = pWF['delta']
    ## TODO: compute chieff?
    
    pPrec['piGM'] = jnp.pi * (m1_SI+m2_SI) * G / C / C / C

    # Spin inputs
    for i, (x, y, z) in enumerate([(chi1x, chi1y, chi1z), (chi2x, chi2y, chi2z)], start=1):
        chi_norm = jnp.sqrt(x*x + y*y + z*z)
        pPrec[f'chi{i}x'] = x
        pPrec[f'chi{i}y'] = y
        pPrec[f'chi{i}z'] = z
        pPrec[f'chi{i}_norm'] = chi_norm

    # Dimensionful spins
    pPrec['S1x'] = chi1x * m1_2
    pPrec['S1y'] = chi1y * m1_2
    pPrec['S1z'] = chi1z * m1_2
    pPrec['S1_norm'] = jnp.abs(pPrec['chi1_norm']) * m1_2
    pPrec['S2x'] = chi2x * m2_2
    pPrec['S2y'] = chi2y * m2_2
    pPrec['S2z'] = chi2z * m2_2
    pPrec['S2_norm'] = jnp.abs(pPrec['chi2_norm']) * m2_2
    
    pPrec['S1_norm_2'] = pPrec['S1_norm']**2
    pPrec['S2_norm_2'] = pPrec['S2_norm']**2

    # In-plane magnitudes
    pPrec['chi1_perp'] = jnp.sqrt(chi1x*chi1x + chi1y*chi1y)
    pPrec['chi2_perp'] = jnp.sqrt(chi2x*chi2x + chi2y*chi2y)
    pPrec['S1_perp'] = m1_2 * pPrec['chi1_perp']
    pPrec['S2_perp'] = m2_2 * pPrec['chi2_perp']
    STot_x = pPrec['S1x'] + pPrec['S2x']
    STot_y = pPrec['S1y'] + pPrec['S2y']
    pPrec['STot_perp'] = jnp.sqrt(STot_x**2 + STot_y**2)
    pPrec['chiTot_perp'] = pPrec['STot_perp'] * (M**2) / m1_2
    # pWF['chiTot_perp'] = pPrec['chiTot_perp']  

    ## disable tuned PNR angles, tuned coprec and mode asymmetries in low in-plane spin limit (TODO)
    
    pPrec['A1'] = 2.0 + (3.0 * m2) / (2.0 * m1)
    pPrec['A2'] = 2.0 + (3.0 * m1) / (2.0 * m2)

    pPrec['ASp1'] = pPrec['A1'] * pPrec['S1_perp']
    pPrec['ASp2'] = pPrec['A2'] * pPrec['S2_perp']

    # S_p = max(A1 S1_perp, A2 S2_perp)
    num = jnp.where(pPrec['ASp2'] > pPrec['ASp1'], pPrec['ASp2'], pPrec['ASp1'])

    den = jnp.where(m2 > m1, pPrec['A2'] * m2_2, pPrec['A1'] * m1_2)

    chip = num / den
    chi1L = chi1z
    chi2L = chi2z

    pPrec['chi_p'] = chip
    pWF['chi_p']   = pPrec['chi_p']   # propagate to waveform struct
    pPrec['phi0_aligned'] = pWF['phi0']

    # Effective (dimensionful) aligned spin
    pPrec['SL'] = chi1L * m1_2 + chi2L * m2_2

    # Effective (dimensionful) in-plane spin
    # (code assumes m1 > m2)
    pPrec['Sperp'] = chip * m1_2

    # initialize error flag
    pPrec['MSA_ERROR'] = 0

    pPrec['pWF22AS'] = None

    pPrec['precessing_tag'] = 2

    '''''
    The following code initializes variables needed for PNR. Since I am implementing only XP->223 for the moment
    I assume this is not needed (to be checked further)
    
    # Initialize PNR variables
    pPrec['chi_singleSpin'] = 0.0
    pPrec['costheta_singleSpin'] = 0.0
    pPrec['costheta_final_singleSpin'] = 0.0
    pPrec['chi_singleSpin_antisymmetric'] = 0.0
    pPrec['theta_antisymmetric'] = 0.0
    pPrec['PNR_HM_Mflow'] = 0.0
    pPrec['PNR_HM_Mfhigh'] = 0.0

    pPrec['PNR_q_window_lower'] = 0.0
    pPrec['PNR_q_window_upper'] = 0.0
    pPrec['PNR_chi_window_lower'] = 0.0
    pPrec['PNR_chi_window_upper'] = 0.0
    # pPrec['PNRInspiralScaling'] = 0  # if needed

    status = IMRPhenomX_PNR_GetAndSetPNRVariables(pWF, pPrec)  ## (TODO)

    # XLAL_CHECK equivalent (trigger MSA_ERROR if failed)
    pPrec['MSA_ERROR'] = status

    # Reset alpha, beta, gamma
    pPrec['alphaPNR'] = 0.0
    pPrec['betaPNR'] = 0.0
    pPrec['gammaPNR'] = 0.0

    # Get and/or store CoPrec params
    status = IMRPhenomX_PNR_GetAndSetCoPrecParams(pWF, pPrec, params)  ## (TODO)

    pPrec['MSA_ERROR'] = status
    '''''
    
    pPrec = IMRPhenomX_Initialize_MSA_System(pWF,pPrec,pPrec['ExpansionOrder']) 
    
    IMRPhenomX_SetPrecessingRemnantParams(pWF,pPrec, params) ## (TODO)
    
    chip2 = chip * chip

    chi1L2 = chi1L * chi1L
    chi2L2 = chi2L * chi2L
    
    pPrec['L0']  = 1.0
    pPrec['L1']  = 0.0
    pPrec['L2']  = 1.5 + eta / 6.0
    pPrec['L3']  = (-7.0 * (chi1L + chi2L + chi1L * delta - chi2L * delta) +
                    5.0 * (chi1L + chi2L) * eta) / 6.0
    pPrec['L4']  = (81.0 + (-57.0 + eta) * eta) / 24.0
    pPrec['L5']  = (-1650.0 * (chi1L + chi2L + chi1L * delta - chi2L * delta) +
                    1336.0 * (chi1L + chi2L) * eta +
                    511.0 * (chi1L - chi2L) * delta * eta +
                    28.0 * (chi1L + chi2L) * eta**2) / 600.0
    pPrec['L6']  = (10935.0 + eta * (-62001.0 + 1674.0 * eta + 7.0 * eta**2 +
                    2214.0 * jnp.pi**2)) / 1296.0
    pPrec['L7']  = 0.0
    pPrec['L8']  = 0.0

    pPrec['L8L'] = 0.0
    
    pPrec['LRef'] = (
        M * M * XLALSimIMRPhenomXLPNAnsatz(
            pWF['v_ref'],  ## (TODO) check if variables in pWF are stored correctly
            pWF['eta'] / pWF['v_ref'],
            pPrec['L0'], pPrec['L1'], pPrec['L2'], pPrec['L3'], pPrec['L4'],
            pPrec['L5'], pPrec['L6'], pPrec['L7'], pPrec['L8'], pPrec['L8L']
        )
    )

    # Source frame J = L + S1 + S2
    pPrec['J0x_Sf'] = m1_2 * chi1x + m2_2 * chi2x
    pPrec['J0y_Sf'] = m1_2 * chi1y + m2_2 * chi2y
    pPrec['J0z_Sf'] = m1_2 * chi1z + m2_2 * chi2z + pPrec['LRef']

    pPrec['J0'] = jnp.sqrt(
        pPrec['J0x_Sf'] * pPrec['J0x_Sf'] +
        pPrec['J0y_Sf'] * pPrec['J0y_Sf'] +
        pPrec['J0z_Sf'] * pPrec['J0z_Sf']
    )

    # Angle between J0 and LN (z-direction)
    def small_J0_case(_):
        return 0.0

    def general_case(pPrec):
        return jnp.arccos(pPrec['J0z_Sf'] / pPrec['J0'])

    pPrec['thetaJ_Sf'] = jnp.where(pPrec['J0'] < 1e-10,
                                   small_J0_case(pPrec),
                                   general_case(pPrec))
    
    phiRef = pWF['phiRef_In'] ## (TODO) check if variables in pWF are stored correctly
    
    convention = 1
    
    max_tol_condition = jnp.logical_and(
        jnp.abs(pPrec['J0x_Sf']) < 1e-15,
        jnp.abs(pPrec['J0y_Sf']) < 1e-15
    )

    pPrec['phiJ_Sf'] = jnp.where(
        max_tol_condition,
        0.0,
        jnp.arctan2(pPrec['J0y_Sf'], pPrec['J0x_Sf'])
    )

    pPrec['phi0_aligned'] = -pPrec['phiJ_Sf']
    pWF['phi0'] = 0  ## (TODO) check if variables in pWF are stored correctly

    # Now rotate from SF to J frame to compute alpha0, the azimuthal angle of LN, as well as 
    # thetaJ, the angle between J and N.
    
    pPrec['Nx_Sf'] = jnp.sin(pWF['inclination']) * jnp.cos((jnp.pi / 2.0) - pWF['phiRef_In'])
    pPrec['Ny_Sf'] = jnp.sin(pWF['inclination']) * jnp.sin((jnp.pi / 2.0) - pWF['phiRef_In'])
    pPrec['Nz_Sf'] = jnp.cos(pWF['inclination'])

    # Temporary vector copy
    tmp_v = jnp.array([pPrec['Nx_Sf'], pPrec['Ny_Sf'], pPrec['Nz_Sf']])

    # Rotate around z, then y
    tmp_v = IMRPhenomX_rotate_z(tmp_v, -pPrec['phiJ_Sf'])
    tmp_v = IMRPhenomX_rotate_y(tmp_v, -pPrec['thetaJ_Sf'])

    ## Difference in overall - sign w.r.t PhenomPv2 code 
    pPrec['kappa'] = jnp.where((jnp.abs(tmp_v[0]) < 1e-15) & (jnp.abs(tmp_v[1]) < 1e-15), 0.0, jnp.arctan2(tmp_v[1], tmp_v[0]))
    
    ## Now determine alpha0 by rotating LN. In the source frame, LN = {0,0,1}
    tmp_v = jnp.array([0,0,1])
    tmp_v = IMRPhenomX_rotate_z(tmp_v, -pPrec['phiJ_Sf'])
    tmp_v = IMRPhenomX_rotate_y(tmp_v, -pPrec['thetaJ_Sf'])
    tmp_v = IMRPhenomX_rotate_z(tmp_v, -pPrec['kappa'])
    
    pPrec['alpha0'] = jnp.pi - pPrec['kappa']
    
    J0dotN = (
        pPrec['J0x_Sf'] * pPrec['Nx_Sf']
        + pPrec['J0y_Sf'] * pPrec['Ny_Sf']
        + pPrec['J0z_Sf'] * pPrec['Nz_Sf']
    )

    pPrec['thetaJN'] = jnp.arccos(J0dotN / pPrec['J0'])

    pPrec['Nz_Jf'] = jnp.cos(pPrec['thetaJN'])
    pPrec['Nx_Jf'] = jnp.sin(pPrec['thetaJN'])
    
    """
    Define the polarizations used. This follows the conventions adopted for IMRPhenomPv2.
 
    The IMRPhenomP polarizations are defined following the conventions in Arun et al (arXiv:0810.5336),
        i.e. projecting the metric onto the P, Q, N triad defining where: P = (N x J) / |N x J|.
 
    However, the triad X,Y,N used in LAL (the "waveframe") follows the definition in the
        NR Injection Infrastructure (Schmidt et al, arXiv:1703.01076).
 
    The triads differ from each other by a rotation around N by an angle zeta. We therefore need to rotate
        the polarizations by an angle 2 zeta.
    """
    
    pPrec['Xx_Sf'] = -jnp.cos(pWF['inclination']) * jnp.sin(phiRef)
    pPrec['Xy_Sf'] = -jnp.cos(pWF['inclination']) * jnp.cos(phiRef)
    pPrec['Xz_Sf'] =  jnp.sin(pWF['inclination'])
    
    tmp_v = jnp.array([pPrec['Xx_Sf'],pPrec['Xy_Sf'],pPrec['Xz_Sf']])
    tmp_v = IMRPhenomX_rotate_z(tmp_v, -pPrec['phiJ_Sf'])
    tmp_v = IMRPhenomX_rotate_y(tmp_v, -pPrec['thetaJ_Sf'])
    tmp_v = IMRPhenomX_rotate_z(tmp_v, -pPrec['kappa'])
    
    pPrec['PArun_Jf'] = jnp.array([pPrec['Nz_Jf'],0,-pPrec['Nx_Jf']])

    # Q = (N x P) by construction
    pPrec['QArun_Jf'] = jnp.array([0,1,0])
    
    pPrec['XdotPArun'] = jnp.dot(tmp_v, pPrec['PArun_Jf'])
    pPrec['XdotQArun'] = jnp.dot(tmp_v, pPrec['QArun_Jf'])
    
    pPrec['zeta_polarization'] = jnp.arctan2(pPrec['XdotQArun'], pPrec['XdotPArun'])

    pPrec['alpha1']    = 0.0
    pPrec['alpha2']    = 0.0
    pPrec['alpha3']    = 0.0
    pPrec['alpha4L']   = 0.0
    pPrec['alpha5']    = 0.0
    pPrec['epsilon1']  = 0.0
    pPrec['epsilon2']  = 0.0
    pPrec['epsilon3']  = 0.0
    pPrec['epsilon4L'] = 0.0
    pPrec['epsilon5']  = 0.0
    
    pPrec['epsilon0'] = pPrec['phiJ_Sf'] - jnp.pi
    
    alpha_offset, epsilon_offset = Get_alphaepsilon_atfref(2, pPrec, pWF)  
    pPrec['alpha_offset']     = alpha_offset
    pPrec['epsilon_offset']   = epsilon_offset
    pPrec['alpha_offset_1']   = alpha_offset
    pPrec['epsilon_offset_1'] = epsilon_offset
    pPrec['alpha_offset_3']   = alpha_offset
    pPrec['epsilon_offset_3'] = epsilon_offset
    pPrec['alpha_offset_4']   = alpha_offset
    pPrec['epsilon_offset_4'] = epsilon_offset
    
    pPrec['cexp_i_alpha']   = 0.0
    pPrec['cexp_i_epsilon'] = 0.0
    pPrec['cexp_i_betah']   = 0.0
    
    """
    Check whether maximum opening angle becomes larger than pi/2 or pi/4.
 
    If (L + S_L) < 0, then Wigner-d Coefficients will not track the angle between J and L, meaning
        that the model may become pathological as one moves away from the aligned-spin limit.
 
    If this does not happen, then max_beta will be the actual maximum opening angle.
 
    This function uses a 2PN non-spinning approximation to the orbital angular momentum L, as
        the roots can be analytically derived.
    """
    
    # lalParams key, is set to zero if max_beta > Pi/2, q>7 and conditionalPrecMBand = 1,
    # thus disabling multibanding
    PrecThresholdMband = IMRPhenomXPCheckMaxOpeningAngle(pWF, pPrec)
    
    ## TODO: conditions to switch on/ off multibanding
    
    ytheta  = pPrec['thetaJN']
    yphi    = 0.0
    
    pPrec['Y2m2']         = compute_sminus2_l2(ytheta, -2)
    pPrec['Y2m1']         = compute_sminus2_l2(ytheta, -1)
    pPrec['Y20']          = compute_sminus2_l2(ytheta,  0)
    pPrec['Y21']          = compute_sminus2_l2(ytheta,  1)
    pPrec['Y22']          = compute_sminus2_l2(ytheta,  2)
    pPrec['Y3m3']         = compute_sminus2_l3(ytheta, -3)
    pPrec['Y3m2']         = compute_sminus2_l3(ytheta, -2)
    pPrec['Y3m1']         = compute_sminus2_l3(ytheta, -1)
    pPrec['Y30']          = compute_sminus2_l3(ytheta,  0)
    pPrec['Y31']          = compute_sminus2_l3(ytheta,  1)
    pPrec['Y32']          = compute_sminus2_l3(ytheta,  2)
    pPrec['Y33']          = compute_sminus2_l3(ytheta,  3)
    pPrec['Y4m4']         = compute_sminus2_l4(ytheta, -4)
    pPrec['Y4m3']         = compute_sminus2_l4(ytheta, -3)
    pPrec['Y4m2']         = compute_sminus2_l4(ytheta, -2)
    pPrec['Y4m1']         = compute_sminus2_l4(ytheta, -1)
    pPrec['Y40']          = compute_sminus2_l4(ytheta,  0)
    pPrec['Y41']          = compute_sminus2_l4(ytheta,  1)
    pPrec['Y42']          = compute_sminus2_l4(ytheta,  2)
    pPrec['Y43']          = compute_sminus2_l4(ytheta,  3)
    pPrec['Y44']          = compute_sminus2_l4(ytheta,  4)
    
    
    return pPrec

def IMRPhenomX_Initialize_MSA_System(pWF, pPrec, ExpansionOrder):
    eta  = pWF['eta']
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta

    m1 = pWF['m1']
    m2 = pWF['m2']

    domegadt_constants_NS = jnp.array([
        96. / 5., -1486. / 35., -264. / 5., 384. * jnp.pi / 5., 34103. / 945., 
        13661. / 105., 944. / 15., jnp.pi * (-4159. / 35.), jnp.pi * (-2268. / 5.),
        (16447322263. / 7276500. + jnp.pi**2 * 512. / 5. - jnp.log(2.) * 109568. / 175. - jnp.euler_gamma * 54784. / 175.),
        (-56198689. / 11340. + jnp.pi**2 * 902. / 5.),
        1623. / 140., -1121. / 27., -54784. / 525., -jnp.pi * 883. / 42.,
        jnp.pi * 71735. / 63., jnp.pi * 73196. / 63.
    ])

    domegadt_constants_SO = jnp.array([
        -904. / 5., -120., -62638. / 105., 4636. / 5., -6472. / 35., 3372. / 5.,
        -jnp.pi * 720., -jnp.pi * 2416. / 5., -208520. / 63., 796069. / 105.,
        -100019. / 45., -1195759. / 945., 514046. / 105., -8709. / 5.,
        -jnp.pi * 307708. / 105., jnp.pi * 44011. / 7., -jnp.pi * 7992. / 7.,
        jnp.pi * 151449. / 35.
    ])

    domegadt_constants_SS = jnp.array([
        -494. / 5., -1442. / 5., -233. / 5., -719. / 5.
    ])

    L_csts_nonspin = jnp.array([
        3. / 2., 1. / 6., 27. / 8., -19. / 8., 1. / 24., 135. / 16.,
        -6889. / 144. + 41. / 24. * jnp.pi**2, 31. / 24., 7. / 1296.
    ])

    L_csts_spinorbit = jnp.array([
        -14. / 6., -3. / 2., -11. / 2., 133. / 72., -33. / 8., 7. / 4.
    ])

    # Flip q convention: Chatziioannou uses q < 1 (m1 > m2), IMRPhenomX uses q > 1
    q = m2 / m1           # q < 1, m1 > m2
    invq = 1.0 / q

    pPrec['qq'] = q
    pPrec['invqq'] = invq

    # Reduced mass
    mu = (m1 * m2) / (m1 + m2)
    
    pPrec['delta_qq']  = (1.0 - pPrec['qq']) / (1.0 + pPrec['qq'])
    pPrec['delta2_qq'] = pPrec['delta_qq'] ** 2
    pPrec['delta3_qq'] = pPrec['delta_qq'] * pPrec['delta2_qq']
    pPrec['delta4_qq'] = pPrec['delta_qq'] * pPrec['delta3_qq']

    # Initialize vectors
    S1v = jnp.array([0.0, 0.0, 0.0])
    S2v = jnp.array([0.0, 0.0, 0.0])
    Lhat = jnp.array([0.0, 0.0, 1.0])

    # Set fixed Lhat variables
    pPrec['Lhat_cos_theta'] = 1.0
    pPrec['Lhat_phi'] = 0.0
    pPrec['Lhat_theta'] = 0.0

    # Dimensionful spin vectors (eta = m1 * m2, q = m2 / m1)
    S1v.at[0].set(pPrec['chi1x'] * eta / q)
    S1v.at[1].set(pPrec['chi1y'] * eta / q)
    S1v.at[2].set(pPrec['chi1z'] * eta / q)

    S2v.at[0].set(pPrec['chi2x'] * eta * q)
    S2v.at[1].set(pPrec['chi2y'] * eta * q)
    S2v.at[2].set(pPrec['chi2z'] * eta * q)

    # Norms of spin vectors
    S1_0_norm = jnp.linalg.norm(S1v)
    S2_0_norm = jnp.linalg.norm(S2v)

    # Store initial spin vectors
    pPrec['S1_0'] = S1v
    pPrec['S2_0'] = S2v

    # Reference velocity v and v^2
    pPrec['v_0'] = jnp.cbrt(pPrec['piGM'] * pWF['fRef'])
    pPrec['v_0_2'] = pPrec['v_0'] ** 2

    # Reference orbital angular momentum
    L_0 = Lhat * eta / pPrec['v_0']
    pPrec['L_0'] = L_0
    
    dotS1L = jnp.dot(S1v, Lhat)
    dotS2L = jnp.dot(S2v, Lhat)
    dotS1S2 = jnp.dot(S1v, S2v)
    dotS1Ln = dotS1L / S1_0_norm
    dotS2Ln = dotS2L / S2_0_norm

    # Store results in the precession structure
    pPrec['dotS1L'] = dotS1L
    pPrec['dotS2L'] = dotS2L
    pPrec['dotS1S2'] = dotS1S2
    pPrec['dotS1Ln'] = dotS1Ln
    pPrec['dotS2Ln'] = dotS2Ln
    
    # --- PN Coefficients for orbital angular momentum ---
    pPrec['constants_L'] = jnp.zeros(5)
    pPrec['constants_L'] = pPrec['constants_L'].at[0].set(
        L_csts_nonspin[0] + eta * L_csts_nonspin[1]
    )

    pPrec['constants_L'] = pPrec['constants_L'].at[1].set(
        IMRPhenomX_Get_PN_beta(L_csts_spinorbit[0], L_csts_spinorbit[1], pPrec)  ## (TODO)
    )

    pPrec['constants_L'] = pPrec['constants_L'].at[2].set(
        L_csts_nonspin[2]
        + eta * L_csts_nonspin[3]
        + eta ** 2 * L_csts_nonspin[4]
    )

    pPrec['constants_L'] = pPrec['constants_L'].at[3].set(
        IMRPhenomX_Get_PN_beta(
            L_csts_spinorbit[2] + L_csts_spinorbit[3] * eta,
            L_csts_spinorbit[4] + L_csts_spinorbit[5] * eta,
            pPrec
        )
    )

    pPrec['constants_L'] = pPrec['constants_L'].at[4].set(
        L_csts_nonspin[5]
        + L_csts_nonspin[6] * eta
        + L_csts_nonspin[7] * eta ** 2
        + L_csts_nonspin[8] * eta ** 3
    )

    # Effective total spin 
    Seff = (1.0 + q) * pPrec['dotS1L'] + (1.0 + 1.0 / q) * pPrec['dotS2L']
    pPrec['Seff'] = Seff

    # Initial total spin, S0 = S1 + S2
    S_0 = S1v + S2v  
    pPrec['S_0'] = S_0

    # Initial total angular momentum J0 = L + S1 + S2 
    pPrec['J_0'] = L_0 + S_0 

    pPrec['S_0_norm'] = jnp.linalg.norm(S_0)

    pPrec['L_0_norm'] = jnp.linalg.norm(pPrec['L_0'])
    pPrec['J_0_norm'] = jnp.linalg.norm(pPrec['J_0'])
    
    vBCD = IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA(pPrec['L_0_norm'], pPrec['J_0_norm'], pPrec)
    
    vRoots = IMRPhenomX_Return_Roots_MSA(pPrec['L_0_norm'], pPrec['J_0_norm'], pPrec)
    
    pPrec['Spl2'] = vRoots[2]
    pPrec['Smi2'] = vRoots[1]
    pPrec['S32']  = vRoots[0]

    pPrec['Spl2pSmi2'] = pPrec['Spl2'] + pPrec['Smi2']
    pPrec['Spl2mSmi2'] = pPrec['Spl2'] - pPrec['Smi2']

    pPrec['Spl'] = jnp.sqrt(pPrec['Spl2'])
    pPrec['Smi'] = jnp.sqrt(pPrec['Smi2'])

    # Eq. 45 of PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['SAv2'] = 0.5 * pPrec['Spl2pSmi2']
    pPrec['SAv']  = jnp.sqrt(pPrec['SAv2'])
    pPrec['invSAv2'] = 1.0 / pPrec['SAv2']
    pPrec['invSAv']  = 1.0 / pPrec['SAv']

    # Eq. 41 of PRD, 95, 104004, (2017), arXiv:1703.03967
    c_1 = 0.5 * (pPrec['J_0_norm']**2 - pPrec['L_0_norm']**2 - pPrec['SAv2']) / (pPrec['L_0_norm'] * eta)
    pPrec['c1'] = c_1
    pPrec['c12'] = c_1 ** 2
    pPrec['c1_over_eta'] = c_1 / eta

    # Average spin couplings over one precession cycle: A9 - A14 of arXiv:1703.03967
    omqsq = (1.0 - q)**2 + 1e-16
    omq2  = (1.0 - q**2) + 1e-16

    pPrec['S1L_pav'] = (c_1 * (1.0 + q) - q * eta * Seff) / (eta * omq2)
    pPrec['S2L_pav'] = -q * (c_1 * (1.0 + q) - eta * Seff) / (eta * omq2)
    pPrec['S1S2_pav'] = 0.5 * pPrec['SAv2'] - 0.5 * (pPrec['S1_norm_2'] + pPrec['S2_norm_2'])
    pPrec['S1Lsq_pav'] = pPrec['S1L_pav'] ** 2 + (pPrec['Spl2mSmi2'] ** 2 * pPrec['v_0_2']) / (32.0 * eta2 * omqsq)
    pPrec['S2Lsq_pav'] = pPrec['S2L_pav'] ** 2 + (q**2 * (pPrec['Spl2mSmi2'] ** 2) * pPrec['v_0_2']) / (32.0 * eta2 * omqsq)
    pPrec['S1LS2L_pav'] = pPrec['S1L_pav'] * pPrec['S2L_pav'] - q * pPrec['Spl2mSmi2'] * pPrec['v_0_2'] / (32.0 * eta2 * omqsq)

    # beta coefficients
    pPrec['beta3'] = ((113.0/12.0) + (25.0/4.0)*(m2/m1)) * pPrec['S1L_pav'] + ((113.0/12.0) 
                        + (25.0/4.0)*(m1/m2)) * pPrec['S2L_pav']

    pPrec['beta5'] = (((31319.0/1008.0) - (1159.0/24.0)*eta) + (m2/m1)*((809.0/84.0) 
                        - (281.0/8.0)*eta)) * pPrec['S1L_pav'] + (((31319.0/1008.0) - (1159.0/24.0)*eta) + (m1/m2)*((809.0/84.0) - (281.0/8.0)*eta)) * pPrec['S2L_pav']

    pPrec['beta6'] = jnp.pi * (
        ((75.0/2.0) + (151.0/6.0)*(m2/m1)) * pPrec['S1L_pav'] +
        ((75.0/2.0) + (151.0/6.0)*(m1/m2)) * pPrec['S2L_pav']
    )

    beta7_common = (130325.0/756.0) - (796069.0/2016.0)*eta + (100019.0/864.0)*eta2
    beta7_S = (1195759.0/18144.0 - 257023.0/1008.0 * eta + 2903.0/32.0 * eta2) 

    pPrec['beta7'] = beta7_common + (m2/m1) * beta7_S * pPrec['S1L_pav'] + beta7_common + (m1/m2) * beta7_S * pPrec['S2L_pav']

    pPrec['sigma4'] = (1.0 / mu) * ((247.0/48.0) * pPrec['S1S2_pav'] - (721.0/48.0) * pPrec['S1L_pav'] * pPrec['S2L_pav']) + (1.0 / (m1**2)) * ((233.0/96.0) * pPrec['S1_norm_2'] 
                        - (719.0/96.0) * pPrec['S1Lsq_pav']) + (1.0 / (m2**2)) * ((233.0/96.0) * pPrec['S2_norm_2'] - (719.0/96.0) * pPrec['S2Lsq_pav'])

    # PN coefficients
    pPrec['a0'] = 96.0 * eta / 5.0
    '''
    pPrec['a2'] = (-(743.0/336.0) - (11.0/4.0)*eta) * pPrec['a0']
    pPrec['a3'] = (4.0 * jnp.pi - pPrec['beta3']) * pPrec['a0']
    pPrec['a4'] = ((34103.0/18144.0) + (13661.0/2016.0)*eta + (59.0/18.0)*eta2 - pPrec['sigma4']) * pPrec['a0']
    pPrec['a5'] = (-(4159.0/672.0)*jnp.pi - (189.0/8.0)*jnp.pi*eta - pPrec['beta5']) * pPrec['a0']
    '''
    pPrec['a6'] = ((16447322263.0/139708800.0) + (16.0/3.0)*jnp.pi**2 - (856.0/105.0)*jnp.log(16.0) - (1712.0/105.0)*jnp.euler_gamma 
            - pPrec['beta6'] + eta*(451.0/48.0)*jnp.pi**2 - (56198689.0/217728.0) + eta2*(541.0/896.0) - eta3*(5605.0/2592.0)) * pPrec['a0']
    pPrec['a7'] = (-(4415.0/4032.0)*jnp.pi + (358675.0/6048.0)*jnp.pi*eta + (91495.0/1512.0)*jnp.pi*eta2 - pPrec['beta7']) * pPrec['a0']
    
    # Default behaviour of IMRPhenomXP (223: MSA with fallback to NNLO)
    pPrec['a0'] = eta * domegadt_constants_NS[0]
    pPrec['a2'] = eta * (domegadt_constants_NS[1] + eta * domegadt_constants_NS[2])
    pPrec['a3'] = eta * (domegadt_constants_NS[3] +
        IMRPhenomX_Get_PN_beta(domegadt_constants_SO[0],domegadt_constants_SO[1],pPrec))
    pPrec['a4'] = eta * (domegadt_constants_NS[4] +eta * (domegadt_constants_NS[5] + eta * domegadt_constants_NS[6]) +
        IMRPhenomX_Get_PN_sigma(domegadt_constants_SS[0],domegadt_constants_SS[1],pPrec) +
        IMRPhenomX_Get_PN_tau(domegadt_constants_SS[2],domegadt_constants_SS[3],pPrec))
    pPrec['a5'] = eta * (domegadt_constants_NS[7] +eta * domegadt_constants_NS[8] +
        IMRPhenomX_Get_PN_beta(
            domegadt_constants_SO[2] + eta * domegadt_constants_SO[3],
            domegadt_constants_SO[4] + eta * domegadt_constants_SO[5],
            pPrec
        )
    )
    
    pPrec['a0_2'] = pPrec['a0'] * pPrec['a0']
    pPrec['a0_3'] = pPrec['a0_2'] * pPrec['a0']
    pPrec['a2_2'] = pPrec['a2'] * pPrec['a2']

    # g-coefficients from Appendix A of Chatziioannou et al, PRD, 95, 104004, (2017), arXiv:1703.03967.
    pPrec['g0'] = 1.0 / pPrec['a0']
    pPrec['g2'] = -pPrec['a2'] / pPrec['a0_2']
    pPrec['g3'] = -pPrec['a3'] / pPrec['a0_2']
    pPrec['g4'] = -(pPrec['a4'] * pPrec['a0'] - pPrec['a2_2']) / pPrec['a0_3']
    pPrec['g5'] = -(pPrec['a5'] * pPrec['a0'] - 2.0 * pPrec['a3'] * pPrec['a2']) / pPrec['a0_3']

    delta = pPrec['delta_qq']
    delta2 = delta * delta
    delta3 = delta * delta2
    delta4 = delta * delta3

    # Phase coefficients: Eq. 51 and Appendix C of arXiv:1703.03967
    pPrec['psi0'] = 0.0
    pPrec['psi1'] = 3.0 * (2.0 * eta2 * Seff - c_1) / (eta * delta2)
    pPrec['psi2'] = 0.0

    # Precompute useful quantities
    c_1_over_nu = pPrec['c1_over_eta']
    c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
    one_p_q_sq = (1.0 + q)**2
    Seff_2 = Seff * Seff
    q_2 = q * q
    one_m_q_sq = (1.0 - q)**2
    one_m_q2_2 = (1.0 - q_2)**2
    one_m_q_4 = one_m_q_sq * one_m_q_sq
    
    Del1 = 4.0 * c_1_over_nu_2 * one_p_q_sq
    Del2 = 8.0 * c_1_over_nu * q * (1.0 + q) * Seff
    Del3 = 4.0 * (one_m_q2_2 * pPrec['S1_norm_2'] - q_2 * Seff_2)
    Del4 = 4.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
    Del5 = 8.0 * c_1_over_nu * q_2 * (1.0 + q) * Seff
    Del6 = 4.0 * (one_m_q2_2 * pPrec['S2_norm_2'] - q_2 * Seff_2)

    pPrec['Delta'] = jnp.sqrt(jnp.abs((Del1 - Del2 - Del3) * (Del4 - Del5 - Del6)))
    
    u1 = 3.0 * pPrec['g2'] / pPrec['g0']
    ### TODO: when m1=m2, one_p_q_sq = 0, division by zero occurs. Check what to do about it
    u2 = 0.75 * one_p_q_sq / one_m_q_4
    u3 = -20.0 * c_1_over_nu_2 * q_2 * one_p_q_sq
    u4 = 2.0 * one_m_q2_2 * (q * (2.0 + q) * pPrec['S1_norm_2']
        + (1.0 + 2.0 * q) * pPrec['S2_norm_2'] - 2.0 * q * pPrec['SAv2'])
    u5 = 4.0 * q_2 * (7.0 + 6.0 * q + 7.0 * q_2) * c_1_over_nu * Seff
    u6 = 2.0 * q_2 * (3.0 + 4.0 * q + 3.0 * q_2) * Seff_2
    u7 = q * pPrec['Delta']

    # (Eq. C2 of 1703.03967)
    pPrec['psi2'] = u1 + u2 * (u3 + u4 + u5 - u6 + u7)
    
    # Eq. D1 - D5  of 1703.03967
    Rm = pPrec['Spl2'] - pPrec['Smi2']
    Rm_2 = Rm * Rm
    cp = pPrec['Spl2'] * eta2 - pPrec['c12']
    cm = pPrec['Smi2'] * eta2 - pPrec['c12']
    cpcm = jnp.abs(cp * cm)
    sqrt_cpcm = jnp.sqrt(cpcm)
    a1dD = 0.5 + 0.75 / eta
    a2dD = -0.75 * Seff / eta

    # Eq. E3- E4 of 1703.03967
    D2RmSq = (cp - sqrt_cpcm) / eta2
    D4RmSq = -0.5 * Rm * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)

    S0m = pPrec['S1_norm_2'] - pPrec['S2_norm_2']

    aw = -3.0 * (1. + q) / q * (2. * (1. + q) * eta2 * Seff * c_1 - (1. + q) * pPrec['c12'] + (1. - q) * eta2 * S0m)
    cw = 3.0 / 32.0 / eta * Rm_2
    dw = 4.0 * cp - 4.0 * D2RmSq * eta2
    hw = -2.0 * (2.0 * D2RmSq - Rm) * c_1
    fw = Rm * D2RmSq - D4RmSq - 0.25 * Rm_2

    adD = aw / dw
    hdD = hw / dw
    cdD = cw / dw
    fdD = fw / dw

    gw = 3. / 16. / eta2 / eta * Rm_2 * (c_1 - eta2 * Seff)
    gdD = gw / dw

    # Powers of coefficients
    hdD_2 = hdD * hdD
    adDfdD = adD * fdD
    adDfdDhdD = adDfdD * hdD
    adDhdD_2 = adD * hdD_2

    # Eq. D10-D15 in PRD, 95, 104004, (2017), arXiv:1703.03967
    pPrec['Omegaz0'] = a1dD + adD    
    pPrec['Omegaz1'] = a2dD - adD*Seff - adD*hdD
    pPrec['Omegaz2'] = adD*hdD*Seff + cdD - adD*fdD + adD*hdD_2
    pPrec['Omegaz3'] = (adDfdD - cdD - adDhdD_2)*(Seff + hdD) + adDfdDhdD
    pPrec['Omegaz4'] = (cdD + adDhdD_2 - 2.0*adDfdD)*(hdD*Seff + hdD_2 - fdD) - adD*fdD*fdD
    pPrec['Omegaz5'] = (cdD - adDfdD + adDhdD_2) * fdD * (Seff + 2.0*hdD) - (cdD + adDhdD_2 - 2.0*adDfdD) * hdD_2 * (Seff + hdD) - adDfdD*fdD*hdD
    
    # Condition for MSA fallback to NNLO
    condition = jnp.abs(pPrec['Omegaz5']) > 1000.0
    pPrec['MSA_ERROR'] = jnp.where(condition, 1, 0)
    
    g0 = pPrec['g0']

    # Eq. 65 coefficients (D16 - D21 of PRD, 95, 104004, (2017), arXiv:1703.03967)
    pPrec['Omegaz0_coeff'] = 3.0 * g0 * pPrec['Omegaz0']
    pPrec['Omegaz1_coeff'] = 3.0 * g0 * pPrec['Omegaz1']
    pPrec['Omegaz2_coeff'] = 3.0 * (g0 * pPrec['Omegaz2'] + pPrec['g2'] * pPrec['Omegaz0'])
    pPrec['Omegaz3_coeff'] = 3.0 * (g0 * pPrec['Omegaz3'] + pPrec['g2'] * pPrec['Omegaz1'] + pPrec['g3'] * pPrec['Omegaz0'])
    pPrec['Omegaz4_coeff'] = 3.0 * (g0 * pPrec['Omegaz4'] + pPrec['g2'] * pPrec['Omegaz2'] + pPrec['g3'] * pPrec['Omegaz1'] + pPrec['g4'] * pPrec['Omegaz0'])
    pPrec['Omegaz5_coeff'] = 3.0 * (g0 * pPrec['Omegaz5'] + pPrec['g2'] * pPrec['Omegaz3'] + pPrec['g3'] * pPrec['Omegaz2'] + pPrec['g4'] * pPrec['Omegaz1'] + pPrec['g5'] * pPrec['Omegaz0'])

    # zeta coefficients (Appendix E of PRD, 95, 104004, (2017), arXiv:1703.03967)
    c1oveta2 = c_1 / eta2
    pPrec['Omegazeta0'] = pPrec['Omegaz0']
    pPrec['Omegazeta1'] = pPrec['Omegaz1'] + pPrec['Omegaz0'] * c1oveta2
    pPrec['Omegazeta2'] = pPrec['Omegaz2'] + pPrec['Omegaz1'] * c1oveta2
    pPrec['Omegazeta3'] = pPrec['Omegaz3'] + pPrec['Omegaz2'] * c1oveta2 + gdD
    pPrec['Omegazeta4'] = pPrec['Omegaz4'] + pPrec['Omegaz3'] * c1oveta2 - gdD * Seff - gdD * hdD
    pPrec['Omegazeta5'] = pPrec['Omegaz5'] + pPrec['Omegaz4'] * c1oveta2 + gdD * hdD * Seff + gdD * (hdD_2 - fdD)

    pPrec['Omegazeta0_coeff'] = -pPrec['g0'] * pPrec['Omegazeta0']
    pPrec['Omegazeta1_coeff'] = -1.5 * pPrec['g0'] * pPrec['Omegazeta1']
    pPrec['Omegazeta2_coeff'] = -3.0 * (pPrec['g0'] * pPrec['Omegazeta2'] + pPrec['g2'] * pPrec['Omegazeta0'])
    pPrec['Omegazeta3_coeff'] = 3.0 * (pPrec['g0'] * pPrec['Omegazeta3'] + pPrec['g2'] * pPrec['Omegazeta1'] + pPrec['g3'] * pPrec['Omegazeta0'])
    pPrec['Omegazeta4_coeff'] = 3.0 * (pPrec['g0'] * pPrec['Omegazeta4'] + pPrec['g2'] * pPrec['Omegazeta2'] + pPrec['g3'] * pPrec['Omegazeta1'] + pPrec['g4'] * pPrec['Omegazeta0'])
    pPrec['Omegazeta5_coeff'] = 1.5 * (pPrec['g0'] * pPrec['Omegazeta5'] + pPrec['g2'] * pPrec['Omegazeta3'] + pPrec['g3'] * pPrec['Omegazeta2'] + pPrec['g4'] * pPrec['Omegazeta1'] + pPrec['g5'] * pPrec['Omegazeta0'])
    
    ## Here we're only considering the default setting, where the expansion order for MSA correction is 5
    ## switch to choose expansion order not yet implemented (TODO)
    pPrec['Omegaz5_coeff']    = 0.0
    pPrec['Omegazeta5_coeff'] = 0.0
    

    condition_equal = jnp.abs(pPrec['Smi2'] - pPrec['Spl2']) < 1.0e-5

    def branch_equal(pPrec):
        return 0.0

    def branch_not_equal(pPrec):
        mm_val = jnp.sqrt((pPrec['Smi2'] - pPrec['Spl2']) / (pPrec['S32'] - pPrec['Spl2']))
        tmpB_val = ((pPrec['S_0_norm'] * pPrec['S_0_norm']) - pPrec['Spl2']) / (pPrec['Smi2'] - pPrec['Spl2'])

        vol_elem = jnp.dot(jnp.cross(L_0, S1v),S2v)
        vol_sign_val = jnp.sign(vol_elem)

        psi_v0_val = IMRPhenomX_psiofv(pPrec['v_0'], pPrec['v_0_2'], 0.0,
                                       pPrec['psi1'], pPrec['psi2'], pPrec)

        # Clamp tmpB in conditions
        cond_case1 = jnp.logical_and(tmpB_val > 1.0,
                                     (tmpB_val - 1.0) < 1.0e-5)
        cond_case2 = jnp.logical_and(tmpB_val < 0.0,
                                     tmpB_val > -1.0e-5)

        def case1():
            return ellipfinc( 
                jnp.arcsin(vol_sign_val * jnp.sqrt(1.0)),
                mm_val
            ) - psi_v0_val

        def case2():
            return ellipfinc(
                jnp.arcsin(vol_sign_val * jnp.sqrt(0.0)),
                mm_val
            ) - psi_v0_val

        def case3():
            return ellipfinc(
                jnp.arcsin(vol_sign_val * jnp.sqrt(tmpB_val)),
                mm_val
            ) - psi_v0_val

        psi0_val = jnp.select(
            [cond_case1, cond_case2, jnp.logical_not(
                jnp.logical_or(tmpB_val > 1.0, tmpB_val < 0.0))],
            [case1(), case2(), case3()]
        )

        return psi0_val

    pPrec['psi0'] = jnp.where(condition_equal,
                     branch_equal(pPrec),
                     branch_not_equal(pPrec))
    
    vMSA = jnp.where(condition_equal, jnp.array([0.,0.,0.]), 
                     IMRPhenomX_Return_MSA_Corrections_MSA(pPrec['v_0'],pPrec['L_0_norm'],pPrec['J_0_norm'],pPrec))
    
    pPrec['phiz_0'] = 0.
    phiz_0        = IMRPhenomX_Return_phiz_MSA(pPrec['v_0'],pPrec['J_0_norm'],pPrec)
    
    pPrec['zeta_0'] = 0.
    zeta_0        = IMRPhenomX_Return_zeta_MSA(pPrec['v_0'],pPrec)
    
    pPrec['phiz_0']    = - phiz_0 - vMSA[0]
    pPrec['zeta_0']    = - zeta_0 - vMSA[1]
    
    return pPrec

def IMRPhenomXPCheckMaxOpeningAngle(
    pWF,
    pPrec
):
    eta = pWF['eta']

    # For now, use the 2PN non-spinning maximum opening angle
    inner_sqrt = 1539.0 - 1008.0 * eta + 19.0 * eta * eta
    numerator = -9.0 - eta + jnp.sqrt(inner_sqrt)
    denominator = 81.0 - 57.0 * eta + eta * eta
    v_at_max_beta = jnp.sqrt(2.0 / 3.0) * jnp.sqrt(numerator / denominator)

    cBetah, sBetah = WignerdCoefficients(v_at_max_beta, pWF, pPrec)

    L_min = XLALSimIMRPhenomXL2PNNS(v_at_max_beta, eta)  
    max_beta = 2.0 * jnp.arccos(cBetah)
    
    def positive_case(vals):
        q, conditionalPrecMBand = vals[0], vals[1]
        jax.debug.print('The maximum opening angle exceeds Pi/2. The model may be pathological in this regime.')
        return jax.lax.cond(jnp.logical_and(q > 7.0, conditionalPrecMBand == 1), lambda x: x*0, lambda x: x, vals[3])
    
    def negative_case(vals):
        max_beta = vals[2]
        PrecThresholdMband = vals[3]
        return PrecThresholdMband
    
    L_plus_SL = L_min + pPrec['SL']
    
    ### missing a warning message if negatice case and max_beta > PI/4 (TODO)
    
    return jax.lax.cond(
        (L_plus_SL < 0.0) & (pPrec['chi_p'] > 0.0),
        positive_case,
        negative_case,
        (pWF['q'], pPrec['conditionalPrecMBand'], max_beta, pPrec['PrecThresholdMband'])
    )

def IMRPhenomX_SetPrecessingRemnantParams(pWF, pPrec, params):
    
    # TODO: Not yet implemented toggles:
    ## Toggle for PNR coprecessing tuning. PNRUseTunedCoprec == 0 for the moment
    ## Toggle for enforced use of non-precessing spin as is required during tuning of PNR's coprecessing model
    ## High-level toggle for whether to apply deviations. APPLY_PNR_DEVIATIONS == 0 for the moment
    
    af_parallel = XLALSimIMRPhenomXFinalSpin2017(pWF['eta'], pPrec['chi1z'], pPrec['chi2z'])  
    M = pWF['M']
    Lfinal = M * M * af_parallel - pWF['m1_2'] * pPrec['chi1z'] - pWF['m2_2'] * pPrec['chi2z']

    # Just implementing FinalSpinMod fsflag == 3, since it is default for MSA approximation
    # see arXiv:2004.06503v2
    # Just implementing branch for MSA_ERROR == 0. No fallback to NNLO for the moment

    method_flag = XLALSimInspiralWaveformParamsLookupPhenomXPTransPrecessionMethod(  ## (TODO)
        params
    )
    sign = jax.lax.cond(
        method_flag == 1,
        lambda _: jnp.copysign(1.0, af_parallel),
        lambda _: 1.0,
        operand=None,
    )
    pWF['afinal_prec'] = sign * jnp.sqrt(pPrec['SAv2'] + Lfinal**2 + 2.0 * Lfinal * (pPrec['S1L_pav'] + pPrec['S2L_pav'])
        ) / (M**2)
    
    pWF['afinal'] = pWF['afinal_prec']
    
    def clip_afinal_prec(_):
        afinal_prec = jnp.copysign(1.0, pWF['afinal_prec'])
        afinal = jnp.copysign(1.0, pWF['afinal'])
        return {**pWF, 'afinal_prec': afinal_prec, 'afinal': afinal}

    def no_clip(_):
        return pWF

    pWF = jax.lax.cond(
        jnp.abs(pWF['afinal_prec']) > 1.0,
        clip_afinal_prec,
        no_clip,
        operand=None,
    )

    # Update ringdown and damping frequency: no precession; to be used for PNR tuned deviations
    pWF['fRING'] = evaluate_QNMfit_fring22(pWF['afinal']) / pWF['Mfinal']  
    pWF['fDAMP'] = evaluate_QNMfit_fdamp22(pWF['afinal']) / pWF['Mfinal']  
    
    return pWF


def IMRPhenomX_Get_PN_beta(a, b, pPrec):
    return (pPrec['dotS1L'] * (a + b * pPrec['qq']) +
            pPrec['dotS2L'] * (a + b / pPrec['qq']))

def IMRPhenomX_Get_PN_sigma(a, b, pPrec):
    return pPrec['inveta'] * (a * pPrec['dotS1S2'] -
                              b * pPrec['dotS1L'] * pPrec['dotS2L'])

def IMRPhenomX_Get_PN_tau(a, b, pPrec):
    return (
        pPrec['qq'] * (pPrec['S1_norm_2'] * a - b * pPrec['dotS1L'] * pPrec['dotS1L']) +
        (a * pPrec['S2_norm_2'] - b * pPrec['dotS2L'] * pPrec['dotS2L']) / pPrec['qq']
    ) / pPrec['eta']
    
def IMRPhenomX_psiofv(v, v2, psi0, psi1, psi2, pPrec):
    # Equation 51 in arXiv:1703.03967
    return psi0 - 0.75 * pPrec['g0'] * pPrec['delta_qq'] * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)


def XLALSimIMRPhenomXFinalSpin2017(eta, chi1L, chi2L):

    delta  = jnp.sqrt(1.0 - 4.0 * eta)
    m1     = 0.5 * (1.0 + delta)
    m2     = 0.5 * (1.0 - delta)
    m1Sq   = m1 * m1
    m2Sq   = m2 * m2

    eta2   = eta * eta
    eta3   = eta2 * eta

    S = XLALSimIMRPhenomXSTotR(eta, chi1L, chi2L)
    S2 = S * S
    S3 = S2 * S

    dchi  = chi1L - chi2L
    dchi2 = dchi * dchi

    noSpin = (3.4641016151377544 * eta
              + 20.0830030082033 * eta2
              - 12.333573402277912 * eta3) / (1.0 + 7.2388440419467335 * eta)

    eqSpin = ((m1Sq + m2Sq) * S
              + ((-0.8561951310209386 * eta
                  - 0.09939065676370885 * eta2
                  + 1.668810429851045 * eta3) * S
                 + (0.5881660363307388 * eta
                    - 2.149269067519131 * eta2
                    + 3.4768263932898678 * eta3) * S2
                 + (0.142443244743048 * eta
                    - 0.9598353840147513 * eta2
                    + 1.9595643107593743 * eta3) * S3)
              / (1.0 + (-0.9142232693081653
                        + 2.3191363426522633 * eta
                        - 9.710576749140989 * eta3) * S))

    uneqSpin = (0.3223660562764661 * dchi * delta * (1 + 9.332575956437443 * eta) * eta2
                - 0.059808322561702126 * dchi2 * eta3
                + 2.3170397514509933 * dchi * delta * (1 - 3.2624649875884852 * eta) * eta3 * S)

    return noSpin + eqSpin + uneqSpin

def XLALSimIMRPhenomXSTotR(eta, chi1L, chi2L):
    """
    Total spin normalized to [-1, 1]
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)

    m1s = m1 * m1
    m2s = m2 * m2

    # Spin combination
    return (m1s * chi1L + m2s * chi2L) / (m1s + m2s)

def evaluate_QNMfit_fring22(a):
    """
    This implementation is the same as the in from IMRPhenomX_utils -> get_cutoff_fMs
    Taken from https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_t_h_m__fits_8c_source.html
    Evaluate the ringdown frequency
    """
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a
    a7 = a6 * a

    return (
        (
            0.05947169566573468
            - 0.14989771215394762 * a
            + 0.09535606290986028 * a2
            + 0.02260924869042963 * a3
            - 0.02501704155363241 * a4
            - 0.005852438240997211 * a5
            + 0.0027489038393367993 * a6
            + 0.0005821983163192694 * a7
        )
        / (
            1
            - 2.8570126619966296 * a
            + 2.373335413978394 * a2
            - 0.6036964688511505 * a4
            + 0.0873798215084077 * a6
        )
    )
    
def evaluate_QNMfit_fdamp22(a):
    """
    This implementation is the same as the in from IMRPhenomX_utils -> get_cutoff_fMs
    Taken from https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_t_h_m__fits_8c_source.html
    Evaluate the ringdown frequency
    """
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a
    
    return (
        (
            0.014158792290965177
            - 0.036989395871554566 * a
            + 0.026822526296575368 * a2
            + 0.0008490933750566702 * a3
            - 0.004843996907020524 * a4
            - 0.00014745235759327472 * a5
            + 0.0001504546201236794 * a6
        )
        / (
            1
            - 2.5900842798681376 * a
            + 1.8952576220623967 * a2
            - 0.31416610693042507 * a4
            + 0.009002719412204133 * a6
        )
    )
    
def XLALSimIMRPhenomXLPNAnsatz(v, LNorm, L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L):
    """
    Computes the PN orbital angular momentum expansion.
    v : Input velocity.
    LNorm : Orbital angular momentum normalization (e.g. η / sqrt(x)).
    L0–L8, L8L : PN coefficients.
    
    Returns
    L : Post-Newtonian angular momentum.
    """

    x = v * v
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    sqx = jnp.sqrt(x)

    L = (
        L0
        + L1 * sqx
        + L2 * x
        + L3 * (x * sqx)
        + L4 * x2
        + L5 * (x2 * sqx)
        + L6 * x3
        + L7 * (x3 * sqx)
        + L8 * x4
        + L8L * x4 * jnp.log(x)
    )
    return LNorm * L

def XLALSimIMRPhenomXL2PNNS(v, eta):
    
    eta2 = eta * eta
    x = v * v
    x2 = x * x
    sqx = v

    return (eta / sqx) * (
        1.0
        + x  * (1.5 + eta / 6.0)
        + x2 * (27.0 / 8.0 - (19.0 * eta) / 8.0 + eta2 / 24.0)
    )

def Get_alphaepsilon_atfref(mprime, pPrec, pWF):

    omega_ref = pWF['piM'] * pWF['fRef'] * (2.0 / mprime)

    v = jnp.cbrt(omega_ref)

    vangles = IMRPhenomX_Return_phi_zeta_costhetaL_MSA(v, pWF, pPrec)

    alpha_offset   = vangles[:,0] - pPrec['alpha0']
    epsilon_offset = vangles[:,1] - pPrec['epsilon0']
    
    return alpha_offset, epsilon_offset


##########################
## NOT YET IMPLEMENTED ###
##########################

def XLALSimInspiralWaveformParamsLookupPhenomXPTransPrecessionMethod(dict):
    """
    TODO: Not yet implemented
    """
    return 1