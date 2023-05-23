from math import pi
from tqdm import tqdm
import jax.numpy as jnp
import time
import jax
from jax import grad, vmap

from ripple import get_eff_pads, get_match_arr
from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
from ripple.waveforms import IMRPhenomP
import matplotlib.pyplot as plt
from ripple.constants import gt, MSUN

# plt.style.use("../plot_style.mplstyle")
import numpy as np
import cProfile
import lalsimulation as lalsim
from ripple import ms_to_Mc_eta
import lal


def lal_phenomD_phenomP_test():
    dist_mpc = 1
    m1_test = 4 * 10**30
    m2_test = 7 * 10**30
    m1_msun = m1_test / MSUN
    m2_msun = m2_test / MSUN
    f_ref = 32
    phi_ref = 0.0
    incl = 0.13
    s1x = -0.0
    s1y = -0.0
    s1z = 0.0
    s2x = 0.0
    s2y = 0.0
    s2z = 0.0
    M = m1_test + m2_test
    f_l = f_ref
    f_u = 1024
    df = 0.0125
    fs = np.arange(f_ref, f_u, df)
    Mc, eta = ms_to_Mc_eta(jnp.array([m2_msun, m1_msun]))
    #theta_ripple = np.array([Mc, eta, s2z, s1z, dist_mpc, 0, phi_ref, incl])
    #hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple, f_ref)
    #hp_ripple, hc_ripple = IMRPhenomP.PhenomPcore(
    #fs, m1_test, m2_test, f_ref, phi_ref, dist_mpc, incl, s1x, s1y, s1z, s2x, s2y, s2z
    #)
    #phi_ripple_p = np.unwrap(np.angle(hp_ripple))

    # print(hps)
    distance = dist_mpc * 1e6 * lal.PC_SI

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")
    hpP, hcP = lalsim.SimInspiralChooseFDWaveform(
        m1_test,
        m2_test,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        distance,
        incl,
        phi_ref,
        0.0,
        0.0,
        0.0,
        df,
        f_ref,
        f_u,
        f_ref,
        None,
        approximant,
    )


    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")
    hpD, hcD = lalsim.SimInspiralChooseFDWaveform(
        m2_test,
        m1_test,
        s2x,
        s2y,
        s2z,
        s1x,
        s1y,
        s1z,
        distance,
        incl,
        phi_ref,
        0.0,
        0.0,
        0.0,
        df,
        f_ref,
        f_u,
        f_ref,
        None,
        approximant,
    )
    #m1 = m1_test / MSUN
    #m2 = m2_test / MSUN
    #q = m2 / m1  # q>=1
    #M = m1 + m2
    # chi_eff = (m1*chi1_l + m2*chi2_l) / M
    # chil = (1.0+q)/q * chi_eff

    freq = np.arange(len(hpD.data.data)) * df
    f_mask = (freq >= f_l) & (freq < f_u)
    # print(freq[f_mask])

    # plt.plot(fs, hcs, "--", label="ripple_hcs")
    # plt.plot(fs, hcsD, "--", label="ripple_hcsD")
    #plt.plot(fs, hp_ripple, "--", label="hp_ripple")
    # print(hpsD)
    # plt.plot(fs, hps/hpsD, "--", label="ripple_hpsD",alpha=0.5)
    # print(hps[-1])
    hpAmp = np.abs(hpD.data.data)
    hppahse = np.unwrap(np.angle(hpD.data.data))
    hpnewD = hpAmp*np.exp(1j* (hppahse + 9.19e-4*freq + 3.497))
    plt.plot(freq[f_mask], np.imag(hpnewD[f_mask]), label="lal_hpD", alpha=0.6)
    plt.plot(freq[f_mask], hpP.data.data[f_mask],"--", label="lal_hpP", alpha=0.6)
    phi_lal_D = np.unwrap(np.angle(hpnewD))[f_mask]
    phi_lal_P = np.unwrap(np.angle(hpP.data.data))[f_mask]
    plt.legend()
    plt.figure()
    plt.plot(fs, np.gradient(phi_lal_D - phi_lal_P, fs), label="diff angle gradient")
    # plt.plot(fs, phi_lal-phi_ripple_p, label = "diff angle")
    # print(phi_lal-phi_python)
    # print(hp.data.data[f_mask])
    plt.legend()
    plt.figure()
    plt.plot(fs, phi_lal_D - phi_lal_P, label="diff angle")
    plt.show()

def my_phenomP_test():
    dist_mpc = 1
    m1_test = 4 * 10**30
    m2_test = 4 * 10**30
    m1_msun = m1_test / MSUN
    m2_msun = m2_test / MSUN
    f_ref = 32
    phi_ref = 0.0
    incl = 0.0
    s1x = -0.0
    s1y = -0.0
    s1z = 0.0
    s2x = 0.0
    s2y = 0.0
    s2z = 0.0
    M = m1_test + m2_test
    f_l = f_ref
    f_u = 1024
    df = 0.0125
    fs = np.arange(f_ref, f_u, df)
    hp_ripple, hc_ripple = IMRPhenomP.PhenomPcore(
    fs, m1_test, m2_test, f_ref, phi_ref, dist_mpc, incl, s1x, s1y, s1z, s2x, s2y, s2z
    )
    phi_ripple_p = np.unwrap(np.angle(hp_ripple))

    # print(hps)
    distance = dist_mpc * 1e6 * lal.PC_SI

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")
    hpP_lal, hcP_lal = lalsim.SimInspiralChooseFDWaveform(
        m1_test,
        m2_test,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        distance,
        incl,
        phi_ref,
        0.0,
        0.0,
        0.0,
        df,
        f_ref,
        f_u,
        f_ref,
        None,
        approximant,
    )

    freq = np.arange(len(hpP_lal.data.data)) * df
    f_mask = (freq >= f_l) & (freq < f_u)
    # print(freq[f_mask])

    # plt.plot(fs, hcs, "--", label="ripple_hcs")
    # plt.plot(fs, hcsD, "--", label="ripple_hcsD")
    plt.plot(fs, hp_ripple, "--", label="hp_ripple")
    plt.plot(freq[f_mask], hpP_lal.data.data[f_mask],"--", label="lal_hpP", alpha=0.6)
    phi_lal_P = np.unwrap(np.angle(hpP_lal.data.data))[f_mask]
    plt.legend()
    plt.figure()
    plt.plot(fs, np.gradient(phi_ripple_p - phi_lal_P, fs), label="diff angle gradient")
    # plt.plot(fs, phi_lal-phi_ripple_p, label = "diff angle")
    # print(phi_lal-phi_python)
    # print(hp.data.data[f_mask])
    plt.legend()
    plt.figure()
    plt.plot(fs, phi_ripple_p - phi_lal_P, label="diff angle")
    plt.show()

# m1 = m1_test / MSUN
# m2 = m2_test / MSUN
# q = m2 / m1  # q>=1
# M = m1 + m2
# chi_eff = (m1 * chi1_l + m2 * chi2_l) / M
# chil = (1.0 + q) / q * chi_eff
# eta = m1 * m2 / (M * M)
#dist_mpc = 1
#m1_test = 4 * 10**30
#m2_test = 7 * 10**30
#m1_msun = m1_test / MSUN
#m2_msun = m2_test / MSUN
#f_ref = 32
#phi_ref = 0.0
#incl = 0.0
#s1x = -0.0
#s1y = -0.0
#s1z = 0.0
#s2x = 0.0
#s2y = 0.0
#s2z = 0.0
#M = m1_test + m2_test
#f_l = f_ref
#f_u = 1024
#df = 0.0125
#fs = np.arange(f_ref, f_u, df)

#(
#    chi1_l,
#    chi2_l,
#    chip,
#    thetaJN,
#    alpha0,
#   phi_aligned,
#    zeta_polariz,
#) = IMRPhenomP.LALtoPhenomP(
#    m1_test, m2_test, f_ref, phi_ref, incl, s1x, s1y, s1z, s2x, s2y, s2z
#)
# print(chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz)
# print(phenom, phasing)
#hpsP, hcsP = IMRPhenomP.PhenomPcore(
#    fs, m1_test, m2_test, f_ref, phi_ref, dist_mpc, incl, s1x, s1y, s1z, s2x, s2y, s2z
#)
# hpsD,_ = IMRPhenomP.PhenomPOneFrequency(fs, m2_msun, m1_msun, s2z, s1z, phi_ref, (m1_test+m2_test)/MSUN, dist_mpc)


#lal_phenomD_phenomP_test()
my_phenomP_test()
# thetaJN = 0.58763
# Y2m2 = IMRPhenomP.SpinWeightedY(thetaJN, 0 , -2, 2, -2)
# Y2m1 = IMRPhenomP.SpinWeightedY(thetaJN, 0 , -2, 2, -1)
# Y20 = IMRPhenomP.SpinWeightedY(thetaJN, 0 , -2, 2, -0)
# Y21 = IMRPhenomP.SpinWeightedY(thetaJN, 0 , -2, 2, 1)
# Y22 = IMRPhenomP.SpinWeightedY(thetaJN, 0 , -2, 2, 2)
# Y2 = [Y2m2, Y2m1, Y20, Y21, Y22]
# print("thetaJN, ", thetaJN)
# print(Y2)
# A_test = jnp.array([1+2j, 4+5j, 3-0.3j])

# angcoeffs = IMRPhenomP.ComputeNNLOanglecoeffs(q, chil, chip)
# hp,hc = IMRPhenomP.PhenomPCoreTwistUp(100, 1, eta, chi1_l, chi2_l, chip, M, angcoeffs, Y2, 0.01-alpha0, 0.01)
# print(hp, hc)
# A_trans = jnp.array(A_trans)
# print(A_trans/A_test)
