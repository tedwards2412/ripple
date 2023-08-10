from math import pi
from tqdm import tqdm
import jax.numpy as jnp
import time
import jax
from jax import grad, vmap

from ripple import get_eff_pads, get_match_arr
from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
from ripple.waveforms import PPE_IMRPhenomP, IMRPhenomP
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
    m1_test = 10 * 10**30
    m2_test = 10 * 10**30
    m1_msun = m1_test / MSUN
    m2_msun = m2_test / MSUN
    f_ref = 32
    phi_ref = 0.4
    incl = 0.0
    s1x = -0.0
    s1y = -0.0
    s1z = 0.5
    s2x = 0.0
    s2y = 0.0
    s2z = 0.5
    M = m1_test + m2_test
    f_l = f_ref
    f_u = 1024
    df = 0.0025
    fs = np.arange(f_ref, f_u, df)
    Mc, eta = ms_to_Mc_eta(jnp.array([m2_msun, m1_msun]))
    # theta_ripple = np.array([Mc, eta, s2z, s1z, dist_mpc, 0, phi_ref, incl])
    # hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple, f_ref)
    # hp_ripple, hc_ripple = IMRPhenomP.PhenomPcore(
    # fs, m1_test, m2_test, f_ref, phi_ref, dist_mpc, incl, s1x, s1y, s1z, s2x, s2y, s2z
    # )
    # phi_ripple_p = np.unwrap(np.angle(hp_ripple))

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
    # m1 = m1_test / MSUN
    # m2 = m2_test / MSUN
    # q = m2 / m1  # q>=1
    # M = m1 + m2
    # chi_eff = (m1*chi1_l + m2*chi2_l) / M
    # chil = (1.0+q)/q * chi_eff

    freq = np.arange(len(hpD.data.data)) * df
    f_mask = (freq >= f_l) & (freq < f_u)
    # print(freq[f_mask])

    # plt.plot(fs, hcs, "--", label="ripple_hcs")
    # plt.plot(fs, hcsD, "--", label="ripple_hcsD")
    # plt.plot(fs, hp_ripple, "--", label="hp_ripple")
    # print(hpsD)
    # plt.plot(fs, hps/hpsD, "--", label="ripple_hpsD",alpha=0.5)
    # print(hps[-1])
    hpAmp = np.abs(hpD.data.data)
    hppahse = np.unwrap(np.angle(hpD.data.data))
    hpnewD = hpAmp * np.exp(1j * (hppahse + 9.19e-4 * freq + 3.497))
    plt.plot(freq[f_mask], np.imag(hpnewD[f_mask]), label="lal_hpD", alpha=0.6)
    plt.plot(freq[f_mask], hpP.data.data[f_mask], "--", label="lal_hpP", alpha=0.6)
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


def my_phenomP_test(phi_ref=0, s1z=0, s2z=0, incl=0):
    dist_mpc = 600
    m1_test = 18.0
    m2_test = 4.0
    m1_SI = m1_test * MSUN
    m2_SI = m2_test * MSUN
    f_ref = 20
    phi_ref = 2.
    incl = 2.
    s1x = 0.1
    s1y = 0.4
    s1z = 0.5
    s2x = 0.3
    s2y = -0.7
    s2z = -0.3
    M = m1_test + m2_test
    f_l = f_ref
    f_u = 2000
    df = 0.005
    fs = np.arange(f_ref, f_u, df)
    #fs = np.array([213.876])
    theta = [
        m1_test,
        m2_test,
        f_ref,
        phi_ref,
        dist_mpc,
        incl,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
    ]
    ppes = np.zeros(15)
    hp_ripple, hc_ripple = IMRPhenomP.PhenomPcore(fs, theta)
    phi_ripple_p = np.unwrap(np.angle(hp_ripple))

    # print(hps)
    distance = dist_mpc * 1e6 * lal.PC_SI

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")
    hpP_lal, hcP_lal = lalsim.SimInspiralChooseFDWaveform(
        m1_SI,
        m2_SI,
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
    plt.plot(fs, hp_ripple,  label="hp_ripple", linewidth = 2, alpha=0.5)
    plt.plot(freq[f_mask], hpP_lal.data.data[f_mask], "--", label="lal_hpP", alpha=0.6)
    phi_lal_P = np.unwrap(np.angle(hpP_lal.data.data))[f_mask]
    # plt.legend()
    plt.figure()
    plt.plot(fs, np.gradient(phi_ripple_p - phi_lal_P, fs), label="diff angle gradient")
    # plt.plot(fs, phi_lal-phi_ripple_p, label = "diff angle")
    # print(phi_lal-phi_python)
    # print(hp.data.data[f_mask])
    # plt.legend()
    plt.figure()
    phase_diff = phi_ripple_p - phi_lal_P
    fit_params = np.polyfit(fs, phase_diff, 1)
    plt.plot(fs, phase_diff, label="diff angle")
    #print(fit_params)
    plt.plot(fs, fit_params[0] * fs + fit_params[1], "--")
    plt.show()
    return fit_params[1]


def random_match_waveforms(n=1000):
    # Get a frequency domain waveform
    thetas = []
    matches = []
    f_ASD, ASD = np.loadtxt("O3Livingston.txt", unpack=True)

    for i in tqdm(range(n)):
        m1 = np.random.uniform(1.0, 100.0)
        m2 = np.random.uniform(1.0, 100.0)
        s1 = np.random.uniform(-1.0, 1.0)
        s2 = np.random.uniform(-1.0, 1.0)
        s1_amp = np.random.uniform(0, 1.0)
        s1_theta = np.random.uniform(0, np.pi)
        s1_phi = np.random.uniform(0, 2 * np.pi)
        s2_amp = np.random.uniform(0, 1.0)
        s2_theta = np.random.uniform(0, np.pi)
        s2_phi = np.random.uniform(0, 2 * np.pi)
        #print(s2_theta, s2_phi)

        # translate that into cartesian
        s1x = s1_amp * np.sin(s1_theta) * np.cos(s1_phi)
        s1y = s1_amp * np.sin(s1_theta) * np.sin(s1_phi)
        s1z = s1_amp * np.cos(s1_theta)

        s2x = s2_amp * np.sin(s2_theta) * np.cos(s2_phi)
        s2y = s2_amp * np.sin(s2_theta) * np.sin(s2_phi)
        s2z = s2_amp * np.cos(s2_theta)
        M = m1 + m2
        tc = 0.0
        phic = 0.0
        dist_mpc = 440
        inclination = np.pi / 2.0
        phi_ref = 0

        ################################
        # Need to normalise the frequency
        # grid according to the total mass
        ################################
        # xi_l = 0.004
        # xi_u = 0.2
        # dxi = 0.000005
        # M_s = M * gt

        # f_l = xi_l / M_s
        # f_u = xi_u / M_s
        # del_f = dxi / M_s
        #################################

        # f_l = 30.0
        # f_u = 1000.0
        # del_f = 0.0125

        f_l = 32.0
        f_u = 2048.0
        del_f = 0.0125

        f_l_idx = round(f_l / del_f)
        f_u_idx = f_u // del_f
        f_l = f_l_idx * del_f
        f_u = f_u_idx * del_f
        fs = np.arange(f_l_idx, f_u_idx) * del_f
        f_ref = f_l
        m1_kg = m1 * lal.MSUN_SI
        m2_kg = m2* lal.MSUN_SI
        distance = dist_mpc * 1e6 * lal.PC_SI

        if m1 < m2:
            theta = np.array(
                [
                    m2,
                    m1,
                    phi_ref,
                    dist_mpc,
                    inclination,
                    s2x,
                    s2y,
                    s2z,
                    s1x,
                    s1y,
                    s1z,
                ]
            )
        elif m1 > m2:
            theta = np.array(
                [
                    m1,
                    m2,
                    phi_ref,
                    dist_mpc,
                    inclination,
                    s1x,
                    s1y,
                    s1z,
                    s2x,
                    s2y,
                    s2z,
                ]
            )
        else:
            raise ValueError("Something went wrong with the parameters")
        approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")

        # coeffs = IMRPhenomD_utils.get_coeffs(theta)
        # _, _, f3, f4, _, _ = IMRPhenomD_utils.get_transition_frequencies(
        #     theta, coeffs[5], coeffs[6]
        # )

        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            theta[0]*lal.MSUN_SI,
            theta[1]*lal.MSUN_SI,
            theta[5],
            theta[6],
            theta[7],
            theta[8],
            theta[9],
            theta[10],
            distance,
            inclination,
            phi_ref,
            0,
            0.0,
            0.0,
            del_f,
            f_l,
            f_u,
            f_ref,
            None,
            approximant,
        )
        freqs = np.arange(len(hp.data.data)) * del_f

        # fs = np.arange(f_l, f_u, del_f)
        # Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

        # theta_ripple = np.array([Mc, eta, s1, s2, dist_mpc, tc, phic])
        ppes = np.zeros(15)
        hp_ripple, hc_ripple = IMRPhenomP.PhenomPcore(fs, theta, f_ref)
        h0_ripple = 2.0 * hp_ripple
        # hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple, f_ref)
        pad_low, pad_high = get_eff_pads(fs)
        PSD_vals = np.interp(fs, f_ASD, ASD) ** 2

        #try:
        mask_lal = (freqs >= f_l) & (freqs < f_u)
        h0_lalsuite = 2.0 * hp.data.data[mask_lal]
        matches.append(
            get_match_arr(
                pad_low,
                pad_high,
                # np.ones_like(fs) * 1.0e-42,
                PSD_vals,
                h0_ripple,
                h0_lalsuite,
            )
        )
        thetas.append(theta)
        #except:
        #    print("Arrays are wrong")

        # else:
        #     mask_lal = (freqs >= f_l) & (freqs <= f_u)
        #     matches.append(
        #         get_match_arr(
        #             pad_low,
        #             pad_high,
        #             np.ones_like(fs) * 1.0e-42,
        #             hp_ripple,
        #             hp.data.data[mask_lal],
        #         )
        #     )
        #     thetas.append(theta)

    thetas = np.array(thetas)
    print(thetas.shape)
    matches = np.array(matches)

    # np.savetxt(thetas)
    np.savetxt("ripple_phenomPv2_matches.txt", np.c_[thetas, matches])

    plt.figure(figsize=(7, 5))
    cm = plt.cm.get_cmap("inferno")
    q = thetas[:, 1] / thetas[:, 0]
    mask = matches == 1.0
    Mtot = thetas[:, 0] + thetas[:, 1]
        # Magnitude of the spin projections in the orbital plane
    S1_perp = thetas[:,0]**2 * jnp.sqrt(thetas[:,5]**2 + thetas[:,6]**2)
    S2_perp = thetas[:,1]**2 * jnp.sqrt(thetas[:,8]**2 + thetas[:,9]**2)

    # print("perps: ", S1_perp, S2_perp)
    A1 = 2 + (3 * thetas[:,1]) / (2 * thetas[:,0])
    A2 = 2 + (3 * thetas[:,0]) / (2 * thetas[:,1])
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp
    num = jnp.maximum(ASp1, ASp2)
    den = A1 * thetas[:,0]**2 # warning: this assumes m1 > m2
    chip = num / den
    chieff = (thetas[:, 0] * thetas[:, 7] + thetas[:, 1] * thetas[:, 10]) / (
        thetas[:, 0] + thetas[:, 1]
    )
    sc = plt.scatter(Mtot, chieff, c=np.log10(1.0 - matches), cmap=cm)
    plt.scatter(Mtot[mask], chieff[mask], color="C0")
    plt.colorbar(sc, label=r"$\log_{10}(1-\mathrm{Match})$")
    plt.xlabel(r"Total Mass, $M$")
    plt.ylabel(r"Effective Spin, $\chi_{\rm eff}$")
    # plt.xlim(1, 50)
    # plt.ylim(, 50)

    plt.savefig("../figures/phenomP_match_vs_lalsuite_qchieff.pdf", bbox_inches="tight")

    # plt.figure(figsize=(7, 5))
    # cm = plt.cm.get_cmap("inferno")
    # sc = plt.scatter(thetas[:, 0], thetas[:, 1], c=np.log10(1.0 - matches), cmap=cm)
    # plt.colorbar(sc, label=r"$\log_{10}(1-\mathrm{Match})$")
    # plt.xlabel(r"$m_1 \,\left[M_{\odot}\right]$")
    # plt.ylabel(r"$m_2 \,\left[M_{\odot}\right]$")
    # plt.xlim(1, 50)
    # plt.ylim(1, 50)

    # plt.savefig("../figures/test_match_vs_lalsuite_m1m2.pdf", bbox_inches="tight")

    # plt.figure(figsize=(7, 5))
    # cm = plt.cm.get_cmap("inferno")
    # sc = plt.scatter(thetas[:, 2], thetas[:, 3], c=np.log10(1.0 - matches), cmap=cm)
    # plt.colorbar(sc, label=r"$\log_{10}(1-\mathrm{Match})$")
    # plt.xlabel(r"$\chi_1$")
    # plt.ylabel(r"$\chi_2$")
    # plt.savefig("../figures/test_match_vs_lalsuite_s1s2.pdf", bbox_inches="tight")

    print(thetas, matches)

def random_match_waveforms_debug(n=500):
    # Get a frequency domain waveform
    thetas = []
    matches = []
    f_ASD, ASD = np.loadtxt("O3Livingston.txt", unpack=True)

    for i in tqdm(range(n)):
        m1 = np.random.uniform(1.0, 100.0)
        m2 = np.random.uniform(1.0, 100.0)
        s1 = np.random.uniform(-1.0, 1.0)
        s2 = np.random.uniform(-1.0, 1.0)

        s1_amp = np.random.uniform(0, 1.0)
        s1_theta = np.random.uniform(0, np.pi)
        s1_phi = np.random.uniform(0, 2 * np.pi)
        s2_amp = np.random.uniform(0, 1.0)
        s2_theta = np.random.uniform(0, np.pi)
        s2_phi = np.random.uniform(0, 2 * np.pi)
        #print(s2_theta, s2_phi)

        # translate that into cartesian
        s1x = s1_amp * np.sin(s1_theta) * np.cos(s1_phi)
        s1y = s1_amp * np.sin(s1_theta) * np.sin(s1_phi)
        s1z = s1_amp * np.cos(s1_theta)

        s2x = s2_amp * np.sin(s2_theta) * np.cos(s2_phi)
        s2y = s2_amp * np.sin(s2_theta) * np.sin(s2_phi)
        s2z = s2_amp * np.cos(s2_theta)
        M = m1 + m2
        tc = 0.0
        phic = 0.0
        dist_mpc = 440
        inclination = np.pi / 2.0
        phi_ref = 0


        f_l = 32.0
        f_u = 1024.0
        del_f = 0.0125

        f_l_idx = round(f_l / del_f)
        f_u_idx = f_u // del_f
        f_l = f_l_idx * del_f
        f_u = f_u_idx * del_f
        fs = np.arange(f_l_idx, f_u_idx) * del_f
        f_ref = f_l
        m1_kg = m1 * lal.MSUN_SI
        m2_kg = m2* lal.MSUN_SI
        distance = dist_mpc * 1e6 * lal.PC_SI

        if m1 < m2:
            theta = np.array(
                [
                    m2,
                    m1,
                    f_ref,
                    phi_ref,
                    dist_mpc,
                    inclination,
                    0, #s2x,
                    0, #s2y,
                    s1, #s2z,
                    0, #s1x,
                    0, #s1y,
                    s2, #s1z,
                ]
            )
            #thetaD = np.array([m2, m1, s1, s2])
        elif m1 > m2:
            theta = np.array(
                [
                    m1,
                    m2,
                    f_ref,
                    phi_ref,
                    dist_mpc,
                    inclination,
                    0, #s1x,
                    0, #s1y,
                    s1, #s1z,
                    0, #s2x,
                    0, #s2y,
                    s2, #s2z,
                ]
            )
            #thetaD = np.array([m1, m2, s1, s2])

        else:
            raise ValueError("Something went wrong with the parameters")
        approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")

        # coeffs = IMRPhenomD_utils.get_coeffs(theta)
        # _, _, f3, f4, _, _ = IMRPhenomD_utils.get_transition_frequencies(
        #     theta, coeffs[5], coeffs[6]
        # )

        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            theta[0]*lal.MSUN_SI,
            theta[1]*lal.MSUN_SI,
            0,
            0,
            s1,
            0,
            0,
            s2,
            distance,
            inclination,
            phi_ref,
            0,
            0.0,
            0.0,
            del_f,
            f_l,
            f_u,
            f_ref,
            None,
            approximant,
        )
        freqs = np.arange(len(hp.data.data)) * del_f

        # fs = np.arange(f_l, f_u, del_f)
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

        theta_ripple = np.array([Mc, eta, s1, s2, dist_mpc, tc, phic])
        #hp_ripple, hc_ripple = IMRPhenomD.PhenomPcore(fs, theta)
        ppes = np.zeros(15)
        h0_ripple = PPE_IMRPhenomD.gen_IMRPhenomD(fs, theta_ripple, f_ref, ppes)
        # hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple, f_ref)
        pad_low, pad_high = get_eff_pads(fs)
        PSD_vals = np.interp(fs, f_ASD, ASD) ** 2

        #try:
        mask_lal = (freqs >= f_l) & (freqs < f_u)
        h0_lalsuite = 2.0 * hp.data.data[mask_lal]
        matches.append(
            get_match_arr(
                pad_low,
                pad_high,
                # np.ones_like(fs) * 1.0e-42,
                PSD_vals,
                h0_ripple,
                h0_lalsuite,
            )
        )
        thetas.append(theta)
        #except:
        #    print("Arrays are wrong")

        # else:
        #     mask_lal = (freqs >= f_l) & (freqs <= f_u)
        #     matches.append(
        #         get_match_arr(
        #             pad_low,
        #             pad_high,
        #             np.ones_like(fs) * 1.0e-42,
        #             hp_ripple,
        #             hp.data.data[mask_lal],
        #         )
        #     )
        #     thetas.append(theta)

    thetas = np.array(thetas)
    print(thetas.shape)
    matches = np.array(matches)

    # np.savetxt(thetas)
    np.savetxt("ripple_phenomPv2_matches.txt", np.c_[thetas, matches])

    plt.figure(figsize=(7, 5))
    cm = plt.cm.get_cmap("inferno")
    q = thetas[:, 1] / thetas[:, 0]
    mask = matches == 1.0
    Mtot = thetas[:, 0] + thetas[:, 1]
        # Magnitude of the spin projections in the orbital plane
    #S1_perp = thetas[:,0]**2 * jnp.sqrt(thetas[:,6]**2 + thetas[:,7]**2)
    #S2_perp = thetas[:,1]**2 * jnp.sqrt(thetas[:,9]**2 + thetas[:,10]**2)

    # print("perps: ", S1_perp, S2_perp)
    #A1 = 2 + (3 * thetas[:,1]) / (2 * thetas[:,0])
    #A2 = 2 + (3 * thetas[:,0]) / (2 * thetas[:,1])
    #ASp1 = A1 * S1_perp
    #ASp2 = A2 * S2_perp
    #num = jnp.maximum(ASp1, ASp2)
    #den = A1 * thetas[:,0]**2 # warning: this assumes m1 > m2
    #chip = num / den
    chieff = (thetas[:, 0] * thetas[:, 8] + thetas[:, 1] * thetas[:, 11]) / (
        thetas[:, 0] + thetas[:, 1]
    )
    sc = plt.scatter(Mtot, chieff, c=np.log10(1.0 - matches), cmap=cm)
    plt.scatter(Mtot[mask], chieff[mask], color="C0")
    plt.colorbar(sc, label=r"$\log_{10}(1-\mathrm{Match})$")
    plt.xlabel(r"Total Mass, $M$")
    plt.ylabel(r"Effective Spin, $\chi_{\rm eff}$")
    # plt.xlim(1, 50)
    # plt.ylim(, 50)

    plt.savefig("../figures/test_match_vs_lalsuite_qchieff_DEBUG.pdf", bbox_inches="tight")

    # plt.figure(figsize=(7, 5))
    # cm = plt.cm.get_cmap("inferno")
    # sc = plt.scatter(thetas[:, 0], thetas[:, 1], c=np.log10(1.0 - matches), cmap=cm)
    # plt.colorbar(sc, label=r"$\log_{10}(1-\mathrm{Match})$")
    # plt.xlabel(r"$m_1 \,\left[M_{\odot}\right]$")
    # plt.ylabel(r"$m_2 \,\left[M_{\odot}\right]$")
    # plt.xlim(1, 50)
    # plt.ylim(1, 50)

    # plt.savefig("../figures/test_match_vs_lalsuite_m1m2.pdf", bbox_inches="tight")

    # plt.figure(figsize=(7, 5))
    # cm = plt.cm.get_cmap("inferno")
    # sc = plt.scatter(thetas[:, 2], thetas[:, 3], c=np.log10(1.0 - matches), cmap=cm)
    # plt.colorbar(sc, label=r"$\log_{10}(1-\mathrm{Match})$")
    # plt.xlabel(r"$\chi_1$")
    # plt.ylabel(r"$\chi_2$")
    # plt.savefig("../figures/test_match_vs_lalsuite_s1s2.pdf", bbox_inches="tight")

    print(thetas, matches)
def benchmark_waveform_call():
    # Get a frequency domain waveform
    f_l = 24
    f_u = 512
    del_f = 0.2
    tc = 0.0
    phic = 0.0
    dist_mpc = 440
    inclination = 0.0
    phi_ref = 0

    n = 1000
    m1 = np.random.uniform(1.0, 100.0, n) * MSUN
    m2 = np.random.uniform(1.0, 100.0, n) * MSUN

    # generate two spins in spherical coords
    s1_amp = np.random.uniform(0, 1.0, n)
    s1_theta = np.random.uniform(0, np.pi, n)
    s1_phi = np.random.uniform(0, 2 * np.pi, n)
    s2_amp = np.random.uniform(0, 1.0, n)
    s2_theta = np.random.uniform(0, np.pi, n)
    s2_phi = np.random.uniform(0, 2 * np.pi, n)
    print(s2_theta, s2_phi)

    # translate that into cartesian
    s1x = s1_amp * np.sin(s1_theta) * np.cos(s1_phi)
    s1y = s1_amp * np.sin(s1_theta) * np.sin(s1_phi)
    s1z = s1_amp * np.cos(s1_theta)

    s2x = s2_amp * np.sin(s2_theta) * np.cos(s2_phi)
    s2y = s2_amp * np.sin(s2_theta) * np.sin(s2_phi)
    s2z = s2_amp * np.cos(s2_theta)
    theta = []
    for i in range(n):
        theta.append(
            np.array(
                [
                    m1[i],
                    m2[i],
                    f_l,
                    phi_ref,
                    dist_mpc,
                    inclination,
                    s1x[i],
                    s1y[i],
                    s1z[i],
                    s2x[i],
                    s2y[i],
                    s2z[i],
                ]
            )
        )

    theta = np.array(theta)

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")

    f_ref = f_l
    distance = dist_mpc * 1e6 * lal.PC_SI

    start = time.time()
    for i in range(n):
        m1_kg = m1
        m2_kg = m2
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            m1_kg[i],
            m2_kg[i],
            theta[i, 6],
            theta[i, 7],
            theta[i, 8],
            theta[i, 9],
            theta[i, 10],
            theta[i, 11],
            distance,
            inclination,
            phi_ref,
            0,
            0.0,
            0.0,
            del_f,
            f_l,
            f_u,
            f_ref,
            None,
            approximant,
        )

    end = time.time()
    print("Lalsuite waveform call takes: %.6f ms" % ((end - start) * 1000 / n))

    fs = np.arange(f_l, f_u, del_f)
    print(fs.shape)
    f_ref = f_l

    @jax.jit
    def waveform(theta):
        return IMRPhenomP.PhenomPcore(fs, theta)

    print("JIT compiling")
    waveform(theta[0])[0].block_until_ready()
    print("Finished JIT compiling")

    start = time.time()
    for t in theta:
        waveform(t)[0].block_until_ready()
    end = time.time()

    print("Ripple waveform call takes: %.6f ms" % ((end - start) * 1000 / n))

    func = vmap(waveform)
    func(theta)[0].block_until_ready()

    start = time.time()
    func(theta)[0].block_until_ready()
    end = time.time()

    print("Vmapped ripple waveform call takes: %.6f ms" % ((end - start) * 1000 / n))

    # start = time.time()

    # @jax.jit
    # def grad_func(p):
    #     return IMRPhenomD.gen_IMRPhenomD(fs[10], p, f_ref).real

    # print(grad(grad_func)(theta_ripple))
    # end = time.time()
    # print("Grad compile time takes: %.4f ms" % ((end - start) * 1000 / n))

    return None


# benchmark_waveform_call()
# my_phenomP_test(phi_ref=0.0, s1z=0.4, s2z=0.4, incl=jnp.pi/2)
random_match_waveforms(n=500)
#lal_phenomD_phenomP_test()
# s1z_list = np.linspace(0,1, 30)
# b_list = []
# for s1z in s1z_list:
#    b_list.append(my_phenomP_test(0 ,s1z, 0, 0 ))
# plt.plot(s1z_list, b_list)
# plt.show()


# print("thetaJN, ", thetaJN)
# print(Y2)
# A_test = jnp.array([1+2j, 4+5j, 3-0.3j])

# hp,hc = IMRPhenomP.PhenomPCoreTwistUp(100, 1, eta, chi1_l, chi2_l, chip, M, angcoeffs, Y2, 0.01-alpha0, 0.01)
# print(hp, hc)
# A_trans = jnp.array(A_trans)
# print(A_trans/A_test)
