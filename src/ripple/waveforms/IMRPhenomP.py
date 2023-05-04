import math
import jax
import jax.numpy as jnp
from math import acos, atan2, sqrt, sin, cos, pi, log
from typing import Tuple
from scipy.special import factorial
import numpy as np
from .IMRPhenomD import Amp as D_Amp

#print(jnp.abs(-3))

LAL_MSUN_SI = 1.9885e30  # Solar mass in kg
LAL_MTSUN_SI = LAL_MSUN_SI * 4.925491025543575903411922162094833998e-6  # Solar mass times G over c^3 in seconds


#helper functions for LALtoPhenomP:
def ROTATEZ(angle, x, y, z):
    tmp_x = x * cos(angle) - y * sin(angle)
    tmp_y = x * sin(angle) + y * cos(angle)
    return tmp_x, tmp_y, z

def ROTATEY(angle, x, y, z):
    tmp_x = x * cos(angle) + z * sin(angle)
    tmp_z = -x * sin(angle) + z * cos(angle)
    return tmp_x, y, tmp_z

def atan2tol(y, x, tol):
    if abs(x) < tol and abs(y) < tol:
        return 0.0
    else:
        return atan2(y, x)


def LALtoPhenomP(
    m1_SI: float, m2_SI: float, f_ref: float, phiRef: float, incl: float, 
    s1x: float, s1y: float, s1z: float, s2x: float, s2y: float, s2z: float
) -> Tuple[float, float, float, float, float, float, float]:

    MAX_TOL_ATAN = 1e-10
    
    # Check arguments for sanity
    if f_ref <= 0:
        raise ValueError("Reference frequency must be positive.")
    if m1_SI <= 0:
        raise ValueError("m1 must be positive.")
    if m2_SI <= 0:
        raise ValueError("m2 must be positive.")
    if abs(s1x**2 + s1y**2 + s1z**2) > 1.0:
        raise ValueError("|S1/m1^2| must be <= 1.")
    if abs(s2x**2 + s2y**2 + s2z**2) > 1.0:
        raise ValueError("|S2/m2^2| must be <= 1.")

    m1 = m1_SI / LAL_MSUN_SI  # Masses in solar masses
    m2 = m2_SI / LAL_MSUN_SI
    M = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    eta = m1 * m2 / (M * M)  # Symmetric mass-ratio

    # From the components in the source frame, we can easily determine
    # chi1_l, chi2_l, chip and phi_aligned, which we need to return.
    # We also compute the spherical angles of J,
    # which we need to transform to the J frame

    # Aligned spins
    chi1_l = s1z  # Dimensionless aligned spin on BH 1
    chi2_l = s2z  # Dimensionless aligned spin on BH 2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * sqrt(s1x**2 + s1y**2)
    S2_perp = m2_2 * sqrt(s2x**2 + s2y**2)

    # In-plane spin components
    deltaPhi = atan2(s2y, s2x) - atan2(s1y, s1x)  # Difference in azimuthal angles
    chip = (S1_perp * cos(deltaPhi) + S2_perp) / (M * M)  # Effective precession spin

    # Orbital angular momentum
    L_x = -eta * M * sqrt(1.0 - chip * chip) * sin(incl)
    L_y = eta * M * sqrt(1.0 - chip * chip) * cos(incl)
    L_z = (m1 * chi1_l + m2 * chi2_l) / (M * M)  # This is L.N/M^2

    # Total angular momentum
    J_x = L_x + m1_2 * s1x + m2_2 * s2x
    J_y = L_y + m1_2 * s1y + m2_2 * s2y
    J_z = L_z + m1_2 * s1z + m2_2 * s2z

    # Compute spherical coordinates of J
    J = sqrt(J_x**2 + J_y**2 + J_z**2)
    if J != 0:
        thetaJN = acos(J_z / J)
    else:
        thetaJN = 0

    # Compute the azimuthal angle of J
    phiJL = atan2tol(J_y, J_x, MAX_TOL_ATAN)

    # Rotate the spins to the J frame
    s1x, s1y, s1z = ROTATEZ(-phiJL, s1x, s1y, s1z)
    s1x, s1y, s1z = ROTATEY(-thetaJN, s1x, s1y, s1z)
    s2x, s2y, s2z = ROTATEZ(-phiJL, s2x, s2y, s2z)
    s2x, s2y, s2z = ROTATEY(-thetaJN, s2x, s2y, s2z)

    return m1_SI, m2_SI, chi1_l, chi2_l, chip, thetaJN, phiJL


#helper functions for spin-weighted spherical harmonics:
def comb(a,b):
    temp = factorial(a)/(factorial(b) * factorial(a-b))
    return temp

def SpinWeightedY(theta, phi, s, l, m):
    'copied from SphericalHarmonics.c in LAL'
    if s == -2:
        if l == 2:
            if m == -2:
                fac = jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1.0 - jnp.cos(theta)) * (1.0 - jnp.cos(theta))
            elif m == -1:
                fac = jnp.sqrt(5.0 / (16.0 * jnp.pi)) * jnp.sin(theta) * (1.0 -
                        jnp.cos(theta))
            elif m == 0:
                fac = jnp.sqrt(15.0 / (32.0 * jnp.pi)) * jnp.sin(theta) * jnp.sin(theta)
            elif m == 1:
                fac = jnp.sqrt(5.0 / (16.0 * jnp.pi)) * jnp.sin(theta) * (1.0 + jnp.cos(theta))
            elif m == 2:
                fac = jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1.0 + jnp.cos(theta)) * (1.0 + jnp.cos(theta))
            else:
                raise ValueError(f"Invalid mode s={s}, l={l}, m={m} - require |m| <= l")
    return fac * np.exp(1j * m * phi)
    #summation = 0
    #for r in range(l-s+1):
    #    summation += (-1)**r * comb(l-s, r) * comb(l+s, r+s-m) / (np.tan(theta/2.0))**(2*r+s-m)
    #outtemp = (-1)**(l+m-s) * np.sqrt( factorial(l+m)* factorial(l-m)* (2*l+1)/(4* np.pi* factorial(l+s)* factorial#(l-s)))
    #out = outtemp * (np.sin(theta/2))**(2*l) * summation * np.exp(1j * m * phi)
    #outreal = out * np.cos(m * phi)
    #outim = out * (-np.sin(m* phi))
    #return out




def PhenomPCoreTwistUp(
    fHz, hPhenom, eta, chi1_l, chi2_l, chip, M, angcoeffs, Y2m, alphaoffset, epsilonoffset, IMRPhenomP_version):
    
    assert angcoeffs is not None
    assert Y2m is not None

    f = fHz * LAL_MTSUN_SI * M  # Frequency in geometric units

    q = (1.0 + sqrt(1.0 - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
    m1 = 1.0 / (1.0 + q)  # Mass of the smaller BH for unit total mass M=1.
    m2 = q / (1.0 + q)  # Mass of the larger BH for unit total mass M=1.
    Sperp = chip * (m2 * m2)  # Dimensionfull spin component in the orbital plane. S_perp = S_2_perp
    chi_eff = (m1 * chi1_l + m2 * chi2_l)  # effective spin for M=1

    if IMRPhenomP_version == 'IMRPhenomPv1_V':
        SL = chi_eff * m2  # Dimensionfull aligned spin of the largest BH. SL = m2^2 chil = m2 * M * chi_eff
    elif IMRPhenomP_version == 'IMRPhenomPv2_V' or IMRPhenomP_version == 'IMRPhenomPv2NRTidal_V':
        SL = chi1_l * m1 * m1 + chi2_l * m2 * m2  # Dimensionfull aligned spin.
    else:
        raise ValueError("Unknown IMRPhenomP version! At present only v1 and v2 and tidal are available.")

    omega = pi * f
    logomega = log(omega)
    omega_cbrt = (omega)**(1/3)
    omega_cbrt2 = omega_cbrt * omega_cbrt

    alpha = (angcoeffs['alphacoeff1'] / omega
             + angcoeffs['alphacoeff2'] / omega_cbrt2
             + angcoeffs['alphacoeff3'] / omega_cbrt
             + angcoeffs['alphacoeff4'] * logomega
             + angcoeffs['alphacoeff5'] * omega_cbrt) - alphaoffset

    epsilon = (angcoeffs['epsiloncoeff1'] / omega
               + angcoeffs['epsiloncoeff2'] / omega_cbrt2
               + angcoeffs['epsiloncoeff3'] / omega_cbrt
               + angcoeffs['epsiloncoeff4'] * logomega
               + angcoeffs['epsiloncoeff5'] * omega_cbrt) - epsilonoffset

    if IMRPhenomP_version == 'IMRPhenomPv1_V':
        pass
    elif IMRPhenomP_version == 'IMRPhenomPv2_V' or IMRPhenomP_version == 'IMRPhenomPv2NRTidal_V':
        cBetah, sBetah = WignerdCoefficients(omega_cbrt, SL, eta, Sperp)
    else:
        raise ValueError("Unknown IMRPhenomP version! At present only v1 and v2 and tidal are available.")

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah2 * cBetah
    cBetah4 = cBetah3 * cBetah
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah2 * sBetah
    sBetah4 = sBetah3 * sBetah

    d2 = [sBetah4, 2 * cBetah * sBetah3, sqrt(6) * sBetah2 * cBetah2, 2 * cBetah3 * sBetah, cBetah4]
    dm2 = [d2[4], -d2[3], d2[2], -d2[1], d2[0]]


    Y2mA = Y2m # this change means you need to pass Y2m in a 5-component list
    hp_sum = 0
    hc_sum = 0

    cexp_i_alpha = np.exp(1j * alpha)
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    cexp_im_alpha = [cexp_m2i_alpha, cexp_mi_alpha, 1.0, cexp_i_alpha, cexp_2i_alpha]

    for m in range(-2, 3):
        T2m = cexp_im_alpha[-m + 2] * dm2[m + 2] * Y2mA[m + 2]
        Tm2m = cexp_im_alpha[m + 2] * d2[m + 2] * np.conj(Y2mA[m + 2])
        hp_sum += T2m + Tm2m
        #print("m=", m)
        #print(T2m, Tm2m)
        hc_sum += 1j * (T2m - Tm2m)
        #print(hc_sum)
        #print("end")

    eps_phase_hP = np.exp(-2j * epsilon) * hPhenom / 2.0
    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return hp, hc

def L2PNR(v: float, eta: float) -> float:
    eta2 = eta**2
    x = v**2
    x2 = x**2
    return (eta*(1.0 + (1.5 + eta/6.0)*x + (3.375 - (19.0*eta)/8. - eta2/24.0)*x2)) / x**0.5

def WignerdCoefficients(v: float, SL: float, eta: float, Sp: float):
    
    # We define the shorthand s := Sp / (L + SL)
    L = L2PNR(v, eta)
    s = Sp / (L + SL)
    s2 = s**2
    cos_beta = 1.0 / (1.0 + s2)**0.5
    cos_beta_half = (1.0 + cos_beta)**0.5 / 2.0  # cos(beta/2)
    sin_beta_half = (1.0 - cos_beta)**0.5 / 2.0  # sin(beta/2)
    
    return cos_beta_half, sin_beta_half

def ComputeNNLOanglecoeffs(q, chil, chip):
    m2 = q/(1. + q)
    m1 = 1./(1. + q)
    dm = m1 - m2
    mtot = 1.
    eta = m1*m2  # mtot = 1
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    mtot2 = mtot*mtot
    mtot4 = mtot2*mtot2
    mtot6 = mtot4*mtot2
    mtot8 = mtot6*mtot2
    chil2 = chil*chil
    chip2 = chip*chip
    chip4 = chip2*chip2
    dm2 = dm*dm
    dm3 = dm2*dm
    m2_2 = m2*m2
    m2_3 = m2_2*m2
    m2_4 = m2_3*m2
    m2_5 = m2_4*m2
    m2_6 = m2_5*m2
    m2_7 = m2_6*m2
    m2_8 = m2_7*m2

    angcoeffs = {}
    angcoeffs['alphacoeff1'] = (-0.18229166666666666 - (5*dm)/(64.*m2))

    angcoeffs['alphacoeff2'] = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta))

    angcoeffs['alphacoeff3'] = (-1.7952473958333333 - (4555*dm)/(7168.*m2) -
         (15*chip2*dm*m2_3)/(128.*mtot4*eta2) -
         (35*chip2*m2_4)/(128.*mtot4*eta2) - (515*eta)/384. - (15*dm2*eta)/(256.*m2_2) -
         (175*dm*eta)/(256.*m2))

    angcoeffs['alphacoeff4'] = - (35*pi)/48. - (5*dm*pi)/(16.*m2) + \
      (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) + \
      (2545*m2_2*chil)/(1152.*mtot2) - \
      (5*chip2*dm*m2_5*chil)/(128.*mtot6*eta3) - \
      (35*chip2*m2_6*chil)/(384.*mtot6*eta3) + (2035*dm*m2*chil)/(21504.*mtot2*eta) + \
      (2995*m2_2*chil)/(9216.*mtot2*eta)

    angcoeffs['alphacoeff5'] = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) -
        (15*chip4*dm*m2_7)/(512.*mtot8*eta4) -
        (35*chip4*m2_8)/(512.*mtot8*eta4) -
        (485*chip2*dm*m2_3)/(14336.*mtot4*eta2) + (475*chip2*m2_4)/(6144.*mtot4*eta2) + \
        (15*chip2*dm2*m2_2)/(256.*mtot4*eta) + (145*chip2*dm*m2_3)/(512.*mtot4*eta) + \
        (575*chip2*m2_4)/(1536.*mtot4*eta) + (39695*eta)/86016. + (1615*dm2*eta)/(28672.*m2_2) - \
        (265*dm*eta)/(14336.*m2) + (955*eta2)/576. + (15*dm3*eta2)/(1024.*m2_3) + \
        (35*dm2*eta2)/(256.*m2_2) + (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*pi*chil)/(16.*mtot2*eta) - \
        (35*m2_2*pi*chil)/(16.*mtot2*eta) + (15*chip2*dm*m2_7*chil2)/(128.*mtot8*eta4) + \
        (35*chip2*m2_8*chil2)/(128.*mtot8*eta4) + \
        (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) + \
        (1645*m2_4*chil2)/(192.*mtot4*eta))
    
    angcoeffs['epsiloncoeff1'] = (-0.18229166666666666 - (5*dm)/(64.*m2))
    angcoeffs['epsiloncoeff2'] = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta))
    angcoeffs['epsiloncoeff3'] = (-1.7952473958333333 - (4555*dm)/(7168.*m2) - (515*eta)/384. -
         (15*dm2*eta)/(256.*m2_2) - (175*dm*eta)/(256.*m2))
    angcoeffs['epsiloncoeff4'] =  - (35*pi)/48. - (5*dm*pi)/(16.*m2) + \
      (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) + \
      (2545*m2_2*chil)/(1152.*mtot2) + (2035*dm*m2*chil)/(21504.*mtot2*eta) + \
      (2995*m2_2*chil)/(9216.*mtot2*eta)
    angcoeffs['epsiloncoeff5'] = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) + (39695*eta)/86016. +
         (1615*dm2*eta)/(28672.*m2_2) - (265*dm*eta)/(14336.*m2) + (955*eta2)/576. +
         (15*dm3*eta2)/(1024.*m2_3) + (35*dm2*eta2)/(256.*m2_2) +
         (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*pi*chil)/(16.*mtot2*eta) - (35*m2_2*pi*chil)/(16.*mtot2*eta) +
         (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) +
         (1645*m2_4*chil2)/(192.*mtot4*eta))
    return angcoeffs

def PhenomPOneFrequency(f, eta, distance, M, phic, PCparams ):
    #PhD.Amp(f, (m1, m2, chi1, chi2), )
    pass

def PhenomPcore(m1_SI: float, m2_SI: float, f_ref: float, phiRef: float, incl: float, s1x: float, s1y: float, s1z: float, s2x: float, s2y: float, s2z: float):
    m1_SI, m2_SI, chi1_l, chi2_l, chip, thetaJN, phiJL = LALtoPhenomP(m1_SI, m2_SI, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y,s2z)
    m1 = m1_SI / LAL_MSUN_SI
    m2 = m2_SI / LAL_MSUN_SI
    q = m2 / m1 # q>=1 
    M = m1 + m2
    chi_eff = (m1*chi1_l + m2*chi2_l) / M
    chil = (1.0+q)/q * chi_eff
    eta = m1 * m2 / (M*M)
    m_sec = M * LAL_MTSUN_SI
    piM = np.pi * m_sec


    omega_ref = piM * f_ref
    logomega_ref = math.log(omega_ref)
    omega_ref_cbrt = (piM * f_ref)**(1/3)  # == v0
    omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt


    
    angcoeffs = ComputeNNLOanglecoeffs(q, chil, chip)

    alphaNNLOoffset = (angcoeffs["alphacoeff1"] / omega_ref
                   + angcoeffs["alphacoeff2"] / omega_ref_cbrt2
                   + angcoeffs["alphacoeff3"] / omega_ref_cbrt
                   + angcoeffs["alphacoeff4"] * logomega_ref
                   + angcoeffs["alphacoeff5"] * omega_ref_cbrt)

    epsilonNNLOoffset = (angcoeffs["epsiloncoeff1"] / omega_ref
                     + angcoeffs["epsiloncoeff2"] / omega_ref_cbrt2
                     + angcoeffs["epsiloncoeff3"] / omega_ref_cbrt
                     + angcoeffs["epsiloncoeff4"] * logomega_ref
                     + angcoeffs["epsiloncoeff5"] * omega_ref_cbrt)
    
    Y2m2 = SpinWeightedY(thetaJN, 0 , -2, 2, -2)
    Y2m1 = SpinWeightedY(thetaJN, 0 , -2, 2, -1)
    Y20 = SpinWeightedY(thetaJN, 0 , -2, 2, -0)
    Y21 = SpinWeightedY(thetaJN, 0 , -2, 2, -1)
    Y22 = SpinWeightedY(thetaJN, 0 , -2, 2, -2)
    Y2 = [Y2m2, Y2m1, Y20, Y21, Y22]

    fCut = 0.0
    finspin = 0.0
    f_final = 0.0

    #finspin = FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH(m1, m2, chi1_l, chi2_l, chip)
    #ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi2_l, chi1_l, finspin)
    #ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi2_l, chi1_l, finspin, extraParams)


    # Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
    # (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
    # was not available when PhenomD was tuned.
    #pn->v[6] -= (Subtract3PNSS(m1, m2, M, eta, chi1_l, chi2_l) * pn->v[0]);
    
    #ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors, 1.0, 1.0);
    # This should be the same as the ending frequency in PhenomD
    #fCut = f_CUT / m_sec
    #f_final = pAmp->fRD / m_sec
    
    hp, hc = PhenomPCoreTwistUp(
    900, 1+1.5j, eta, chi1_l, chi2_l, chip, M, angcoeffs, Y2, 0, 0, "IMRPhenomPv2_V")
    print(hp, hc)


#For test purposes
if __name__ == "__main__":
   pass