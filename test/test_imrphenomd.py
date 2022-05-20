from math import pi
from tqdm import tqdm
import jax.numpy as jnp

from ripple import get_eff_pads, get_match_arr
from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
import matplotlib.pyplot as plt
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
    f_l = 40
    f_u = 512
    del_f = 0.25
    f_list = np.arange(f_l, f_u, del_f)

    params = jnp.array([Mc, eta, chi1, chi2, D, tc, phic])
    grad_func = lambda p: IMRPhenomD.gen_IMRPhenomD(f_list[10], p).real
    # print(grad(grad_func)(params))
    grad(grad_func)(params)
    return None


def test_phase_phenomD():
    theta = np.array([49.0, 48.0, 0.5, 0.5])
    Mc, eta = ms_to_Mc_eta(jnp.array([theta[0], theta[1]]))
    f_l = 32
    f_u = 400
    del_f = 0.0125
    # del_f = 0.001
    inclination = 0.0
    psi = 0.0
    phi_ref = 0.0
    dist_mpc = 1.0
    tc = 0.0
    phic = 0.0

    f = np.arange(f_l, f_u, del_f)
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

    # Amp = IMRPhenomD.Amp(f, theta)
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(f, theta_ripple)

    f_ref = f_l
    print(Mf[0])
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

    freq = np.arange(len(hp.data.data)) * del_f
    f_mask = (freq >= f_l) & (freq < f_u)
    print(f, freq[f_mask])

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
    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"$\Phi$")
    plt.savefig("../figures/test_phase.pdf", bbox_inches="tight")

    print(
        "Ripple Phase:",
        np.unwrap(np.angle(hp_ripple)),
    )
    print("Lalsuite Phase:", np.unwrap(np.angle(hp.data.data))[f_mask])

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
        np.unwrap(np.angle(hp.data.data))[f_mask] - np.unwrap(np.angle(hp_ripple)),
        label="phase difference",
    )
    plt.axvline(x=f1)
    plt.axvline(x=f2)
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
    theta = np.array([20.0, 19.99, -0.95, -0.95])
    Mc, eta = ms_to_Mc_eta(jnp.array([theta[0], theta[1]]))
    Mf0 = 0.003
    Mf1 = 0.11
    f_l = Mf0 / ((theta[0] + theta[1]) * 4.92549094830932e-6)
    f_u = Mf1 / ((theta[0] + theta[1]) * 4.92549094830932e-6)
    del_f = 0.001
    inclination = np.pi / 2
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
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(f, theta_ripple)

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
        freq[f_mask] * ((theta[0] + theta[1]) * 4.92549094830932e-6),
        abs(hp.data.data)[f_mask],
        label="lalsuite",
    )
    plt.loglog(Mf, abs(hp_ripple), label="ripple")
    plt.axvline(x=f3 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    plt.axvline(x=f4 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"Amplitude")
    plt.savefig("../figures/test_Amp_full.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    plt.semilogy(
        freq[f_mask] * ((theta[0] + theta[1]) * 4.92549094830932e-6),
        (abs(hp_ripple) / abs(hp.data.data)[f_mask]),
        label="ratio",
    )
    plt.axvline(x=f3 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    plt.axvline(x=f4 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"Amplitude")
    plt.savefig("../figures/test_Amp_ratio.pdf", bbox_inches="tight")

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
    m1_msun = 49.0
    m2_msun = 48.0
    chi1 = [0, 0, 0.5]
    chi2 = [0, 0, 0.5]
    tc = 0.0
    phic = 0.0
    dist_mpc = 440
    inclination = 0.0
    phi_ref = 0
    polarization_angle = 0.0

    Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

    theta_ripple = np.array(
        [Mc, eta, chi1[2], chi2[2], dist_mpc, tc, phic, inclination, polarization_angle]
    )

    theta = np.array([m1_msun, m2_msun, chi1[2], chi2[2]])
    f_l = 20
    f_u = 1024
    del_f = 0.01
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
    mask_lal = (freqs >= f_l) & (freqs < f_u)

    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple)

    plt.figure(figsize=(15, 5))
    plt.plot(
        freqs[mask_lal], hp.data.data[mask_lal].real, label="hp lalsuite", alpha=0.3
    )

    plt.plot(
        fs,
        hp_ripple.real,
        label="hp ripple",
        alpha=0.3,
    )

    plt.axvline(x=f3, ls="--")
    plt.axvline(x=f4, ls="--")
    plt.legend()
    plt.xlim(0, 300)
    plt.xlabel("Frequency")
    plt.ylabel("hf")
    plt.savefig("../figures/waveform_comparison.pdf", bbox_inches="tight")

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


def random_match_waveforms(n=100):
    # Get a frequency domain waveform
    f_l = 32
    f_u = 1024
    del_f = 0.0125
    thetas = []
    matches = []

    for i in tqdm(range(n)):
        m1 = np.random.uniform(1.0, 50.0)
        m2 = np.random.uniform(1.0, 50.0)
        s1 = np.random.uniform(-1.0, 1.0)
        s2 = np.random.uniform(-1.0, 1.0)
        tc = 0.0
        phic = 0.0
        dist_mpc = 440
        inclination = 0.0
        phi_ref = 0
        polarization_angle = 0.0

        if m1 < m2:
            theta = np.array([m2, m1, s1, s2])
        elif m1 > m2:
            theta = np.array([m1, m2, s1, s2])
        else:
            raise ValueError("Something went wrong with the parameters")
        approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

        coeffs = IMRPhenomD_utils.get_coeffs(theta)
        _, _, f3, f4, _, _ = IMRPhenomD_utils.get_transition_frequencies(
            theta, coeffs[5], coeffs[6]
        )

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
        mask_lal = (freqs >= f_l) & (freqs < f_u)

        fs = np.arange(f_l, f_u, del_f)
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

        theta_ripple = np.array(
            [
                Mc,
                eta,
                s1,
                s2,
                dist_mpc,
                tc,
                phic,
                inclination,
                polarization_angle,
            ]
        )
        hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple)
        pad_low, pad_high = get_eff_pads(fs)

        matches.append(
            get_match_arr(
                pad_low,
                pad_high,
                np.ones_like(fs),
                hp_ripple,
                hp.data.data[mask_lal],
            )
        )
        thetas.append(theta)

    thetas = np.array(thetas)
    matches = np.array(matches)

    plt.figure(figsize=(7, 5))
    cm = plt.cm.get_cmap("inferno")
    sc = plt.scatter(thetas[:, 0], thetas[:, 1], c=matches, cmap=cm, label="Match")
    plt.colorbar(sc)
    plt.xlabel("m1")
    plt.ylabel("m2")
    plt.savefig("../figures/test_match_vs_pycbc_m1m2.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    cm = plt.cm.get_cmap("inferno")
    sc = plt.scatter(thetas[:, 2], thetas[:, 3], c=matches, cmap=cm, label="Match")
    plt.colorbar(sc)
    plt.xlabel("s1")
    plt.ylabel("s2")
    plt.savefig("../figures/test_match_vs_pycbc_s1s2.pdf", bbox_inches="tight")

    print(thetas, matches)


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
    test_phase_phenomD()
    # test_frequency_calc()
    # plot_waveforms()
    # random_match_waveforms(n=1000)
    None
