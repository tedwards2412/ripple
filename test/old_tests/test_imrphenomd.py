from math import pi
from tqdm import tqdm
import jax.numpy as jnp
import time
import jax
from jax import grad, vmap

from ripple import get_eff_pads, get_match_arr
from ripple.waveforms import PPE_IMRPhenomD, IMRPhenomD, IMRPhenomD_utils
import matplotlib.pyplot as plt
from ripple.constants import gt

#plt.style.use("../plot_style.mplstyle")
import numpy as np
import cProfile
import lalsimulation as lalsim
from ripple import ms_to_Mc_eta
import lal


def profile_grad():
    from jax import grad

    # Now lets compute the waveform ripple
    m1 = 20.0
    m2 = 10.0
    from ripple import ms_to_Mc_eta

    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    chi1 = 0.0
    chi2 = 0.0
    D = 100.0
    tc = 0.01
    phic = 0.0
    f_l = 32
    f_u = 512
    del_f = 0.25
    f_list = np.arange(f_l, f_u, del_f)

    params = jnp.array([Mc, eta, chi1, chi2, D, tc, phic])
    grad_func = lambda p: IMRPhenomD.gen_IMRPhenomD(f_list[10], p, f_ref=f_l).real
    # print(grad(grad_func)(params))
    grad(grad_func)(params)
    return None


def test_phase_phenomD():
    theta = np.array(
        [
            7.765270965631057720e01,
            4.909567250892310142e01,
            3.969370946508115061e-01,
            4.405203497762351095e-01,
        ]
    )
    Mc, eta = ms_to_Mc_eta(jnp.array([theta[0], theta[1]]))
    # f_l = 30.0
    # f_u = 1000.0
    # del_f = 0.0005
    # del_f = 0.0125
    inclination = 0.0
    psi = 0.0
    phi_ref = 0.0
    dist_mpc = 1.0
    tc = 0.0
    phic = 0.0

    f_l = 32
    f_u = 1024
    del_f = 0.0125

    f_l_idx = round(f_l / del_f)
    f_u_idx = f_u // del_f
    f_l = f_l_idx * del_f
    f_u = f_u_idx * del_f
    f = np.arange(f_l_idx, f_u_idx) * del_f

    # f = np.arange(f_l, f_u, del_f)
    Mf = f * (theta[0] + theta[1]) * 4.92549094830932e-6
    del_Mf = np.diff(Mf)

    # Calculate the frequency regions
    coeffs = IMRPhenomD_utils.get_coeffs(theta)
    f1, f2, _, _, _, _ = IMRPhenomD_utils.get_transition_frequencies(
        theta, coeffs[5], coeffs[6]
    )
    theta_ripple = np.array(
        [Mc, eta, theta[2], theta[3], dist_mpc, tc, phic, inclination, psi]
    )

    f_ref = f_l

    # Amp = IMRPhenomD.Amp(f, theta)
    hp_ripple, hc_ripple = PPE_IMRPhenomD.gen_IMRPhenomD_polar(f, theta_ripple, f_ref)

    # print(Mf[0])
    m1_kg = theta[0] * lal.MSUN_SI
    m2_kg = theta[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        theta[2],
        0.0,
        0.0,
        theta[3],
        distance,
        inclination,
        phi_ref,
        0.0,
        0.0,
        0.0,
        del_f,
        f_l,
        f_u,
        f_ref,
        None,
        approximant,
    )

    # print(hp.deltaF)
    # print(hp.name)
    # print(hp.f0)
    # quit()

    freq = np.arange(len(hp.data.data)) * del_f
    f_mask = (freq >= f_l) & (freq < f_u)
    # print(freq[f_mask], f)
    # quit()

    plt.figure(figsize=(7, 5))
    plt.plot(
        freq[f_mask],
        np.unwrap(np.angle(hp.data.data))[f_mask],
        label="lalsuite",
    )

    plt.plot(
        f,
        np.unwrap(np.angle(hp_ripple)),  # - np.unwrap(np.angle(hp_ripple))[0],
        label="ripple",
        alpha=0.3,
    )
    # plt.axvline(x=f1)
    # plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"$\Phi$")
    plt.savefig("../figures/test_phase.pdf", bbox_inches="tight")

    print(
        "Ripple Phase:",
        np.unwrap(np.angle(hp_ripple)),
    )
    print("Lalsuite Phase:", np.unwrap(np.angle(hp.data.data))[f_mask])

    print(
        "Difference:",
        np.unwrap(np.angle(hp_ripple)) - np.unwrap(np.angle(hp.data.data))[f_mask],
    )

    # ratio = np.unwrap(np.angle(hp_ripple)) / np.unwrap(np.angle(hp.data.data))[f_mask]
    # plt.figure(figsize=(7, 5))
    # plt.plot(
    #     freq[f_mask],
    #     ratio,
    #     label="ratio",
    # )
    # plt.legend()
    # plt.xlabel(r"Mf")
    # plt.ylabel(r"$\Phi$")
    # plt.savefig("../figures/test_phase_ratio.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    plt.plot(
        f,
        (np.unwrap(np.angle(hp.data.data))[f_mask]) - (np.unwrap(np.angle(hp_ripple))),
        label="phase difference",
    )
    # plt.axvline(x=f1)
    # plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"$\Phi_1 - \Phi_2$")
    plt.savefig("../figures/test_phasedifference.pdf", bbox_inches="tight")

    phase_deriv_lal = np.gradient(np.unwrap(np.angle(hp.data.data))[f_mask], del_Mf[0])
    phase_deriv = np.gradient(np.unwrap(np.angle(hp_ripple)), del_Mf[0])
    plt.figure(figsize=(6, 5))
    print("testing here", phase_deriv, phase_deriv_lal)
    plt.plot(f, phase_deriv, label="ripple", alpha=0.3)
    plt.plot(
        freq[f_mask],  # * ((theta[0] + theta[1]) * 4.92549094830932e-6),
        phase_deriv_lal,
        label="lalsuite",
        alpha=0.3,
        ls="--",
    )
    plt.legend()
    plt.xlabel(r"f")
    # plt.xlim(100, 120)
    # plt.ylim(100, 200)
    plt.ylabel(r"$\Phi^{\prime}$")
    plt.savefig("../figures/test_phase_derivative_full.pdf", bbox_inches="tight")

    plt.figure(figsize=(6, 5))
    plt.plot(f, phase_deriv_lal - phase_deriv, label="difference")
    plt.legend()
    plt.xlabel(r"f")
    # plt.xlim(100, 120)
    # plt.ylim(-0.00050, -0.00025)
    plt.ylabel(r"$\Phi^{\prime}_1-\Phi^{\prime}_2$")
    plt.savefig("../figures/test_phase_derivative_difference.pdf", bbox_inches="tight")

    return None


def test_Amp_phenomD():
    theta = np.array(
        [
            7.765270965631057720e00,
            4.909567250892310142e00,
            3.969370946508115061e-01,
            4.405203497762351095e-01,
        ]
    )
    Mc, eta = ms_to_Mc_eta(jnp.array([theta[0], theta[1]]))
    # f_l = 32
    # f_u = 1024
    # del_f = 0.0125
    f_l = 32
    f_u = 1024
    del_f = 0.0125

    f_l_idx = round(f_l / del_f)
    f_u_idx = f_u // del_f
    f_l = f_l_idx * del_f
    f_u = f_u_idx * del_f
    fs = np.arange(f_l_idx, f_u_idx) * del_f

    # f_l = 32
    # f_u = 1024
    # del_f = 0.0005
    psi = 0.0
    phi_ref = 0.0
    dist_mpc = 1.0

    f = np.arange(f_l, f_u, del_f)
    inclination = 0.0
    psi = 0.0
    phi_ref = 0.0
    dist_mpc = 1.0
    f = np.arange(f_l, f_u, del_f)
    Mf = f * (theta[0] + theta[1]) * 4.92549094830932e-6

    # Calculate the frequency regions
    coeffs = IMRPhenomD_utils.get_coeffs(theta)
    _, _, f3, f4, _, _ = IMRPhenomD_utils.get_transition_frequencies(
        theta, coeffs[5], coeffs[6]
    )
    theta_ripple = np.array(
        [Mc, eta, theta[2], theta[3], dist_mpc, 0.0, 0.0, inclination, psi]
    )

    # Amp = IMRPhenomD.Amp(f, theta)
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(f, theta_ripple, f_ref=f_l)

    f_ref = f_l
    m1_kg = theta[0] * lal.MSUN_SI
    m2_kg = theta[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        theta[2],
        0.0,
        0.0,
        theta[3],
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
    freq = np.arange(len(hp.data.data)) * del_f
    f_mask = (freq >= f_l) & (freq < f_u)

    plt.figure(figsize=(7, 5))
    plt.plot(
        freq[f_mask], #* ((theta[0] + theta[1]) * 4.92549094830932e-6),
        hp.data.data[f_mask],
        label="lalsuite",
    )
    plt.plot(f, hp_ripple, label="ripple", alpha=0.5, linewidth=3)
    # plt.axvline(x=f3 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    # plt.axvline(x=f4 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"Amplitude")
    plt.savefig("../figures/test_Amp_full.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    plt.plot(
        freq[f_mask] * ((theta[0] + theta[1]) * 4.92549094830932e-6),
        (abs(hp_ripple) - abs(hp.data.data)[f_mask]),
        label="difference",
    )
    # plt.axvline(x=f3 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    # plt.axvline(x=f4 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"Amplitude")
    plt.savefig("../figures/test_Amp_difference.pdf", bbox_inches="tight")

    return None


def test_frequency_calc():
    theta = np.array([5.0, 4.0, 0.0, 0.0])
    m1, m2, chi1, chi2 = theta

    fRD, fdamp = IMRPhenomD_utils.get_fRD_fdamp(m1, m2, chi1, chi2)
    print("Ringdown and damping frequency:", fRD, fdamp)
    return None


def plot_waveforms():
    # Get a frequency domain waveform
    # source parameters
    m1_msun = 15.0
    m2_msun = 10.0
    chi1 = [0, 0, 0.5]
    chi2 = [0, 0, 0.5]
    tc = 0.0
    phic = 0.0
    dist_mpc = 440
    inclination = 0.8
    phi_ref = 0

    Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

    theta_ripple = np.array(
        [Mc, eta, chi1[2], chi2[2], dist_mpc, tc, phic, inclination]
    )

    #ppes = np.random.uniform(0,1e-5, 15)
    #print(ppes)

    theta = np.array([m1_msun, m2_msun, chi1[2], chi2[2]])
    f_l = 32.0
    f_u = 1024.0
    del_f = 0.0125
    fs = np.arange(f_l, f_u, del_f)

    coeffs = IMRPhenomD_utils.get_coeffs(theta)
    _, _, f3, f4, _, _ = IMRPhenomD_utils.get_transition_frequencies(
        theta, coeffs[5], coeffs[6]
    )

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

    f_ref = f_l
    m1_kg = m1_msun * lal.MSUN_SI
    m2_kg = m2_msun * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        chi1[0],
        chi1[1],
        chi1[2],
        chi2[0],
        chi2[1],
        chi2[2],
        distance,
        inclination,
        phi_ref,
        0.0,
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
    mask_lal = (freqs >= f_l) & (freqs < f_u)

    hp_ripple, hc_ripple = PPE_IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple, ppes, f_ref)

    plt.figure(figsize=(15, 5))
    plt.plot(
        freqs[mask_lal],
        hp.data.data[mask_lal].real,
        label="hp lalsuite real",
        alpha=0.3,
        color="C0",
        ls="--",
        lw=5,
    )

    plt.plot(
        fs,
        hp_ripple.real,
        label="hp ripple real",
        alpha=0.8,
        color="C0",
    )

    plt.plot(
        freqs[mask_lal],
        hp.data.data[mask_lal].imag,
        label="hp lalsuite imag",
        alpha=0.3,
        color="C1",
        ls="--",
        lw=5,
    )

    plt.plot(
        fs,
        hp_ripple.imag,
        label="hp ripple imag",
        alpha=0.8,
        color="C1",
    )

    plt.axvline(x=f3, ls="--")
    plt.axvline(x=f4, ls="--")
    # plt.legend()
    plt.xlim(0, 300)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Strain")
    plt.show()
    plt.savefig("../figures/waveform_comparison_hp.pdf", bbox_inches="tight")

    plt.figure(figsize=(15, 5))
    plt.plot(
        freqs[mask_lal],
        hc.data.data[mask_lal].real,
        label="hc lalsuite real",
        alpha=0.3,
        color="C0",
        ls="--",
        lw=5,
    )

    plt.plot(
        fs,
        hc_ripple.real,
        label="hc ripple real",
        alpha=0.3,
        color="C0",
    )

    plt.plot(
        freqs[mask_lal],
        hc.data.data[mask_lal].imag,
        label="hc lalsuite imag",
        alpha=0.3,
        color="C1",
        ls="--",
        lw=5,
    )

    plt.plot(
        fs,
        hc_ripple.imag,
        label="hc ripple imag",
        alpha=0.3,
        color="C1",
    )

    plt.axvline(x=f3, ls="--")
    plt.axvline(x=f4, ls="--")
    plt.legend()
    plt.xlim(0, 300)
    plt.xlabel("Frequency")
    plt.ylabel("hf")
    plt.savefig("../figures/waveform_comparison_hc.pdf", bbox_inches="tight")

    pad_low, pad_high = get_eff_pads(fs)

    print(
        get_match_arr(
            pad_low,
            pad_high,
            np.ones_like(fs),
            hp_ripple,
            hp.data.data[mask_lal],
        )
    )


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
        f_u = 1024.0
        del_f = 0.0125

        f_l_idx = round(f_l / del_f)
        f_u_idx = f_u // del_f
        f_l = f_l_idx * del_f
        f_u = f_u_idx * del_f
        fs = np.arange(f_l_idx, f_u_idx) * del_f

        if m1 < m2:
            theta = np.array([m2, m1, s1, s2])
        elif m1 > m2:
            theta = np.array([m1, m2, s1, s2])
        else:
            raise ValueError("Something went wrong with the parameters")
        approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

        # coeffs = IMRPhenomD_utils.get_coeffs(theta)
        # _, _, f3, f4, _, _ = IMRPhenomD_utils.get_transition_frequencies(
        #     theta, coeffs[5], coeffs[6]
        # )

        f_ref = f_l
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        distance = dist_mpc * 1e6 * lal.PC_SI

        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            0.0,
            0.0,
            theta[2],
            0.0,
            0.0,
            theta[3],
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
        ppes = np.zeros(15)
        h0_ripple = PPE_IMRPhenomD.gen_IMRPhenomD(fs, theta_ripple, ppes, f_ref)
        # hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple, f_ref)
        pad_low, pad_high = get_eff_pads(fs)
        PSD_vals = np.interp(fs, f_ASD, ASD) ** 2

        try:
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
        except:
            print("Arrays are wrong")

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
    matches = np.array(matches)

    # np.savetxt(thetas)
    np.savetxt("ripple_phenomD_matches.txt", np.c_[thetas, matches])

    plt.figure(figsize=(7, 5))
    cm = plt.cm.get_cmap("inferno")
    q = thetas[:, 1] / thetas[:, 0]
    mask = matches == 1.0
    Mtot = thetas[:, 0] + thetas[:, 1]
    chieff = (thetas[:, 0] * thetas[:, 2] + thetas[:, 1] * thetas[:, 3]) / (
        thetas[:, 0] + thetas[:, 1]
    )
    sc = plt.scatter(Mtot, chieff, c=np.log10(1.0 - matches), cmap=cm)
    plt.scatter(Mtot[mask], chieff[mask], color="C0")
    plt.colorbar(sc, label=r"$\log_{10}(1-\mathrm{Match})$")
    plt.xlabel(r"Total Mass, $M$")
    plt.ylabel(r"Effective Spin, $\chi_{\rm eff}$")
    # plt.xlim(1, 50)
    # plt.ylim(, 50)

    plt.savefig("../figures/test_match_vs_lalsuite_qchieff.pdf", bbox_inches="tight")

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

    n = 10_000
    m1 = np.random.uniform(1.0, 100.0, n)
    m2 = np.random.uniform(1.0, 100.0, n)
    s1 = np.random.uniform(-1.0, 1.0, n)
    s2 = np.random.uniform(-1.0, 1.0, n)
    theta = []
    theta_ripple = []
    for i in range(n):
        if m1[i] < m2[i]:
            theta.append(np.array([m2[i], m1[i], s1[i], s2[i]]))
            Mc, eta = ms_to_Mc_eta(jnp.array([m2[i], m1[i]]))
            theta_ripple.append(
                np.array(
                    [
                        Mc,
                        eta,
                        s1[i],
                        s2[i],
                        dist_mpc,
                        tc,
                        phic,
                        inclination,
                    ]
                )
            )
        elif m1[i] > m2[i]:
            Mc, eta = ms_to_Mc_eta(jnp.array([m1[i], m2[i]]))
            theta_ripple.append(
                np.array(
                    [
                        Mc,
                        eta,
                        s1[i],
                        s2[i],
                        dist_mpc,
                        tc,
                        phic,
                        inclination,
                    ]
                )
            )
            theta.append(np.array([m1[i], m2[i], s1[i], s2[i]]))
        else:
            raise ValueError("Something went wrong with the parameters")

    theta = np.array(theta)
    theta_ripple = np.array(theta_ripple)

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

    f_ref = f_l
    distance = dist_mpc * 1e6 * lal.PC_SI

    start = time.time()
    for i in range(n):
        m1_kg = theta[i, 0] * lal.MSUN_SI
        m2_kg = theta[i, 1] * lal.MSUN_SI
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            0.0,
            0.0,
            theta[i, 2],
            0.0,
            0.0,
            theta[i, 3],
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
        return IMRPhenomD.gen_IMRPhenomD_polar(fs, theta, f_ref)

    print("JIT compiling")
    waveform(theta_ripple[0])[0].block_until_ready()
    print("Finished JIT compiling")

    start = time.time()
    for t in theta_ripple:
        waveform(t)[0].block_until_ready()
    end = time.time()

    print("Ripple waveform call takes: %.6f ms" % ((end - start) * 1000 / n))

    func = vmap(waveform)
    func(theta_ripple)[0].block_until_ready()

    start = time.time()
    func(theta_ripple)[0].block_until_ready()
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


if __name__ == "__main__":
    import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    # profile_grad()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()
    # profile_grad()
    # test_Amp_phenomD()
    # test_phase_phenomD()
    # test_frequency_calc()
    plot_waveforms()
    # benchmark_waveform_call()
    # random_match_waveforms(n=400)
    None
