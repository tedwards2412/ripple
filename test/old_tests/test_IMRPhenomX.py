from math import pi
from tqdm import tqdm
import jax.numpy as jnp
import time
import jax
from jax import grad, vmap

from ripple.waveforms import IMRPhenomXAS, IMRPhenomX_utils, IMRPhenomD
import matplotlib.pyplot as plt
from ripple.constants import gt
from ripple import get_eff_pads, get_match_arr

from WF4Py import waveforms

# plt.style.use("../plot_style.mplstyle")
import numpy as np
import cProfile
import lalsimulation as lalsim
from ripple import ms_to_Mc_eta, Mc_eta_to_ms
import lal


def test_phase_phenomXAS():
    theta = np.array([20, 19.0, 0.5, 0.5])
    events = {
        "Mc": np.array([16.969038560664817]),
        "dL": np.array([1000.0]),
        "iota": np.array([0.0]),
        "eta": np.array([0.24983563445101906]),
        "chi1z": np.array([theta[2]]),
        "chi2z": np.array([theta[3]]),
        "Lambda1": np.array([0.0]),
        "Lambda2": np.array([0.0]),
    }
    Mc, eta = ms_to_Mc_eta(jnp.array([theta[0], theta[1]]))
    print(Mc, eta)
    tc = 0.0
    phic = 0.0
    dist_mpc = 440
    inclination = np.pi / 2.0
    phi_ref = 0

    f_l = 10
    f_u = 1024
    del_f = 0.0125

    f_l_idx = round(f_l / del_f)
    f_u_idx = f_u // del_f
    f_l = f_l_idx * del_f
    f_u = f_u_idx * del_f
    f = np.arange(f_l_idx, f_u_idx) * del_f

    # f = np.arange(f_l, f_u, del_f)
    Mf = f * (theta[0] + theta[1]) * 4.92549094830932e-6
    M_s = (theta[0] + theta[1]) * 4.92549094830932e-6
    # del_Mf = np.diff(Mf)

    # Calculate the frequency regions
    theta_ripple = np.array([theta[0], theta[1], theta[2], theta[3]])

    f_ref = f_l
    phase_ripple = IMRPhenomXAS.Phase(
        f, theta_ripple, IMRPhenomX_utils.PhenomX_phase_coeff_table
    )
    phase_wf4py = waveforms.IMRPhenomXAS().Phi(f, **events) - 2 * np.pi

    ################ Just for display ##################
    fRD, fdamp, fMECO, fISCO = IMRPhenomX_utils.get_cutoff_fs(
        theta[0], theta[1], theta[2], theta[3]
    )
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fINmatch = fMECO
    deltaf = (fIMmatch - fINmatch) * 0.03
    # fPhaseInsMin = 0.0026
    # fPhaseInsMax = 1.020 * fMECO
    # fPhaseRDMin = fIMmatch
    # fPhaseRDMax = fRD + 1.25 * fdamp
    f1 = (fINmatch - 1.0 * deltaf) / M_s
    f2 = (fIMmatch + 0.5 * deltaf) / M_s
    ####################################################

    m1_kg = theta[0] * lal.MSUN_SI
    m2_kg = theta[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")

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
    h0_lalsuite = 2.0 * hp.data.data[f_mask]

    plt.figure(figsize=(7, 5))
    plt.plot(
        freq[f_mask],
        np.unwrap(np.angle(h0_lalsuite)),
        label="lalsuite",
    )

    # plt.plot(
    #     f,
    #     phase_ripple,
    #     label="ripple",
    #     alpha=0.3,
    # )
    plt.plot(
        f,
        phase_wf4py,
        label="wf4py",
        alpha=0.3,
    )
    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$\Phi$")
    # plt.xlim(f1 - 10, f1 + 10)
    plt.savefig("../figures/inspiral_phase_PhenomX.pdf", bbox_inches="tight")

    print(
        "Ripple Phase:",
        phase_ripple,
    )
    print("Lalsuite Phase:", np.unwrap(np.angle(h0_lalsuite)))

    # print(
    #     "Difference:",
    #     np.unwrap(np.angle(hp_ripple)) - np.unwrap(np.angle(hp.data.data))[f_mask],
    # )

    plt.figure(figsize=(7, 5))
    # plt.plot(
    #     f,
    #     (np.unwrap(np.angle(h0_lalsuite))) - phase_ripple,
    #     label="phase difference",
    # )
    plt.plot(
        f,
        (np.unwrap(np.angle(h0_lalsuite))) - phase_wf4py,
        label="wf4py phase difference",
    )
    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$\Phi_1 - \Phi_2$")
    plt.savefig("../figures/test_phasedifference_PhenomX.pdf", bbox_inches="tight")

    # phase_deriv_lal = np.gradient(np.unwrap(np.angle(hp.data.data))[f_mask], Mf)
    phase_deriv = np.gradient(phase_ripple, Mf)
    phase_deriv_wf4py = np.gradient(phase_wf4py, Mf)
    plt.figure(figsize=(6, 5))
    # plt.plot(f, phase_deriv, label="ripple", alpha=0.3)
    plt.plot(f, phase_deriv_wf4py, label="wf4py", alpha=0.3)
    # plt.plot(
    #     freq[f_mask],  # * ((theta[0] + theta[1]) * 4.92549094830932e-6),
    #     phase_deriv_lal,
    #     label="lalsuite",
    #     alpha=0.3,
    #     ls="--",
    # )
    plt.legend()
    plt.xlabel(r"Mf")
    # plt.xlim(100, 120)
    # plt.ylim(100, 200)
    plt.ylabel(r"$\Phi^{\prime}$")
    plt.savefig("../figures/test_phase_derivative_phenomX.pdf", bbox_inches="tight")

    plt.figure(figsize=(6, 5))
    # diff = (np.unwrap(np.angle(h0_lalsuite))) - phase_ripple
    # plt.plot(Mf, np.gradient(diff, np.diff(f)[0]), label="difference")
    diff_wf4py = (np.unwrap(np.angle(h0_lalsuite))) - phase_wf4py
    plt.plot(Mf, np.gradient(diff_wf4py, np.diff(f)[0]), label="wf4py")
    test_scaling = Mf ** (-1 / 3)
    plt.axvline(x=M_s * f1)
    plt.axvline(x=M_s * f2)
    # plt.plot(
    #     Mf,
    #     (test_scaling) + (np.gradient(diff, np.diff(f)[0])[0] - test_scaling[0]),
    #     label="test_scaling 4/3",
    # )
    plt.legend()
    plt.xlabel(r"f")
    # plt.xlim(100, 120)
    # plt.ylim(-0.00050, -0.00025)
    plt.ylabel(r"$(\Phi_1-\Phi_2)^{\prime}$")
    plt.savefig(
        "../figures/test_phase_difference_derivative_phenomX.pdf", bbox_inches="tight"
    )

    return None


def test_gen_phenomXAS(
    theta_intrinsic=jnp.array([42.0, 33.90281401, 0.1, -0.9]),
    f_l=20,
    f_u=620,
    del_f=0.0125,
):
    Mc, eta = ms_to_Mc_eta(jnp.array([theta_intrinsic[0], theta_intrinsic[1]]))
    print(f"Chirp Mass = {Mc:.2f} Msol, eta = {eta:.2f}")
    theta_intrinsic_lal = np.array(
        [theta_intrinsic[0], theta_intrinsic[1], theta_intrinsic[2], theta_intrinsic[3]]
    )
    events = {
        "Mc": np.array([Mc]),
        "dL": np.array([440.0]),
        "iota": np.array([0.0]),
        "eta": np.array([eta]),
        "chi1z": np.array([theta_intrinsic_lal[2]]),
        "chi2z": np.array([theta_intrinsic_lal[3]]),
        "Lambda1": np.array([0.0]),
        "Lambda2": np.array([0.0]),
    }
    dist_mpc = 440.0
    tc = 0.0
    phic = 0.0
    inclination = np.pi / 2.0
    phi_ref = 0.0
    theta_extrinsic = jnp.array([dist_mpc, tc, phic])

    f_l_idx = round(f_l / del_f)
    f_u_idx = f_u // del_f
    f_l = f_l_idx * del_f
    f_u = f_u_idx * del_f
    f = np.arange(f_l_idx, f_u_idx) * del_f
    f_ref = f_l

    phase_wf4py = waveforms.IMRPhenomXAS().Phi(f, **events) - 2 * np.pi

    Mf = f * (theta_intrinsic[0] + theta_intrinsic[1]) * 4.92549094830932e-6
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * 4.92549094830932e-6

    phase_ripple = IMRPhenomXAS._gen_IMRPhenomXAS(
        f, theta_intrinsic, theta_extrinsic, IMRPhenomX_utils.PhenomX_phase_coeff_table
    )

    ################ Just for display ##################
    fRD, fdamp, fMECO, fISCO = IMRPhenomX_utils.get_cutoff_fs(
        theta_intrinsic[0], theta_intrinsic[1], theta_intrinsic[2], theta_intrinsic[3]
    )
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fINmatch = fMECO
    deltaf = (fIMmatch - fINmatch) * 0.03

    f1 = (fINmatch - 1.0 * deltaf) / M_s
    f2 = (fIMmatch + 0.5 * deltaf) / M_s

    m1_kg = theta_intrinsic_lal[0] * lal.MSUN_SI
    m2_kg = theta_intrinsic_lal[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        theta_intrinsic_lal[2],
        0.0,
        0.0,
        theta_intrinsic_lal[3],
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
    h0_lalsuite = 2.0 * hp.data.data[f_mask]

    difference = np.unwrap(np.angle(h0_lalsuite)) - phase_ripple
    alpha = (difference[-1] - difference[0]) / (f[-1] - f[0])
    beta = difference[0] - alpha * f[0]

    plt.figure(figsize=(7, 5))
    plt.plot(
        freq[f_mask],
        np.unwrap(np.angle(h0_lalsuite)),
        label="lalsuite",
    )
    plt.plot(
        f,
        phase_ripple,
        label="ripple",
        alpha=0.3,
    )
    plt.plot(
        f,
        phase_wf4py,
        label="wf4py",
        alpha=0.3,
    )

    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$\Phi$")
    plt.savefig("../figures/gen_inspiral_phase_PhenomX.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    plt.plot(
        freq[f_mask],
        np.unwrap(np.angle(h0_lalsuite)) - phase_ripple,
        label="ripple",
    )
    # plt.plot(
    #     freq[f_mask],
    #     np.unwrap(np.angle(h0_lalsuite)) - phase_wf4py,
    #     label="wf4py",
    # )

    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$\Delta\Phi$")
    plt.savefig("../figures/gen_diff_phase_PhenomX.pdf", bbox_inches="tight")

    print(
        "Ripple Phase:",
        phase_ripple,
    )
    print("Lalsuite Phase:", np.unwrap(np.angle(h0_lalsuite)))
    return None


def test_amplitude_XAS(
    theta_intrinsic=jnp.array([55.0, 33.90281401, 0.0, 0.0]),
    f_l=20,
    f_u=620,
    del_f=0.0125,
):
    Mc, eta = ms_to_Mc_eta(jnp.array([theta_intrinsic[0], theta_intrinsic[1]]))
    print(f"Chirp Mass = {Mc:.2f} Msol, eta = {eta:.2f}")
    theta_intrinsic_lal = np.array(
        [theta_intrinsic[0], theta_intrinsic[1], theta_intrinsic[2], theta_intrinsic[3]]
    )
    events = {
        "Mc": np.array([Mc]),
        "dL": np.array([0.44]),
        "iota": np.array([0.0]),
        "eta": np.array([eta]),
        "chi1z": np.array([theta_intrinsic_lal[2]]),
        "chi2z": np.array([theta_intrinsic_lal[3]]),
        "Lambda1": np.array([0.0]),
        "Lambda2": np.array([0.0]),
    }
    dist_mpc = 440.0
    tc = 0.0
    phic = 0.0
    inclination = np.pi / 2.0
    phi_ref = 0.0
    theta_extrinsic = jnp.array([dist_mpc, tc, phic])

    f_l_idx = round(f_l / del_f)
    f_u_idx = f_u // del_f
    f_l = f_l_idx * del_f
    f_u = f_u_idx * del_f
    f = np.arange(f_l_idx, f_u_idx) * del_f
    f_ref = f_l

    amp_wf4py = waveforms.IMRPhenomXAS().Ampl(f, **events)

    Mf = f * (theta_intrinsic[0] + theta_intrinsic[1]) * 4.92549094830932e-6
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * 4.92549094830932e-6

    amp_ripple = IMRPhenomXAS.Amp(f, theta_intrinsic, D=dist_mpc)

    ################ Just for display ##################
    fRD, fdamp, fMECO, fISCO = IMRPhenomX_utils.get_cutoff_fs(
        theta_intrinsic[0], theta_intrinsic[1], theta_intrinsic[2], theta_intrinsic[3]
    )
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fINmatch = fMECO
    deltaf = (fIMmatch - fINmatch) * 0.03

    f1 = (fINmatch - 1.0 * deltaf) / M_s
    f2 = (fIMmatch + 0.5 * deltaf) / M_s

    m1_kg = theta_intrinsic_lal[0] * lal.MSUN_SI
    m2_kg = theta_intrinsic_lal[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        theta_intrinsic_lal[2],
        0.0,
        0.0,
        theta_intrinsic_lal[3],
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
    h0_lalsuite = 2.0 * hp.data.data[f_mask]

    difference = np.abs(h0_lalsuite) - amp_ripple

    plt.figure(figsize=(7, 5))
    plt.plot(
        freq[f_mask],
        np.abs(h0_lalsuite),
        label="lalsuite",
    )
    plt.plot(
        f,
        amp_ripple,
        label="ripple",
        alpha=0.3,
    )
    plt.plot(
        f,
        amp_wf4py,
        label="wf4py",
        alpha=0.3,
    )

    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$A(f)$")
    plt.savefig("../figures/gen_amp_PhenomX.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    plt.semilogy(
        freq[f_mask],
        np.abs((np.abs(h0_lalsuite) - amp_ripple) / np.abs(h0_lalsuite)),
        label="ripple",
    )
    plt.semilogy(
        freq[f_mask],
        np.abs((np.abs(h0_lalsuite) - amp_wf4py) / np.abs(h0_lalsuite)),
        label="wf4py",
    )
    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$\Delta A / A$")
    plt.savefig("../figures/gen_diff_amp_PhenomX.pdf", bbox_inches="tight")


def test_full_waveform_XAS(
    theta_intrinsic=jnp.array([55.0, 33.90281401, 0.8, -0.8]),
    f_l=20,
    f_u=620,
    del_f=0.0125,
):
    Mc, eta = ms_to_Mc_eta(jnp.array([theta_intrinsic[0], theta_intrinsic[1]]))
    print(f"Chirp Mass = {Mc:.2f} Msol, eta = {eta:.2f}")
    theta_intrinsic_lal = np.array(
        [theta_intrinsic[0], theta_intrinsic[1], theta_intrinsic[2], theta_intrinsic[3]]
    )
    dist_mpc = 440.0
    tc = 0.0
    phic = 0.0
    inclination = np.pi / 2.0
    phi_ref = 0.0
    theta_extrinsic = jnp.array([dist_mpc, tc, phic])

    f_l_idx = round(f_l / del_f)
    f_u_idx = f_u // del_f
    f_l = f_l_idx * del_f
    f_u = f_u_idx * del_f
    f = np.arange(f_l_idx, f_u_idx) * del_f
    f_ref = f_l

    Mf = f * (theta_intrinsic[0] + theta_intrinsic[1]) * 4.92549094830932e-6
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * 4.92549094830932e-6

    params = jnp.array(
        [Mc, eta, theta_intrinsic[2], theta_intrinsic[3], dist_mpc, tc, phic]
    )

    h0_ripple = IMRPhenomXAS.gen_IMRPhenomXAS(f, params)

    ################ Just for display ##################
    fRD, fdamp, fMECO, fISCO = IMRPhenomX_utils.get_cutoff_fMs(
        theta_intrinsic[0], theta_intrinsic[1], theta_intrinsic[2], theta_intrinsic[3]
    )
    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fINmatch = fMECO
    deltaf = (fIMmatch - fINmatch) * 0.03

    f1 = (fINmatch - 1.0 * deltaf) / M_s
    f2 = (fIMmatch + 0.5 * deltaf) / M_s

    m1_kg = theta_intrinsic_lal[0] * lal.MSUN_SI
    m2_kg = theta_intrinsic_lal[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        theta_intrinsic_lal[2],
        0.0,
        0.0,
        theta_intrinsic_lal[3],
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
    h0_lalsuite = 2.0 * hp.data.data[f_mask]

    plt.figure(figsize=(7, 5))
    plt.plot(
        freq[f_mask],
        h0_lalsuite,
        label="lalsuite",
    )
    plt.plot(
        f,
        h0_ripple,
        label="ripple",
        alpha=0.3,
    )
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"h0")
    plt.savefig("../figures/full_waveform_PhenomX.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    plt.loglog(
        freq[f_mask],
        np.abs(h0_lalsuite) ** 2,
        label="lalsuite",
    )
    plt.loglog(
        f,
        np.abs(h0_ripple) ** 2,
        label="ripple",
        alpha=0.3,
    )

    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$|A(f)|^2$")
    plt.savefig("../figures/full_waveform_PhenomX_amp.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    plt.loglog(
        freq[f_mask],
        np.abs((np.abs(h0_lalsuite) - np.abs(h0_ripple)) / np.abs(h0_ripple)),
        label="lalsuite",
    )

    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$|A(f)|$")
    plt.savefig("../figures/full_waveform_PhenomX_ampdiff.pdf", bbox_inches="tight")

    plt.figure(figsize=(7, 5))
    plt.plot(
        f,
        np.gradient(np.unwrap(np.angle(h0_lalsuite))),
        label="lalsuite",
        alpha=0.3,
    )
    plt.plot(
        f,
        -np.gradient(np.unwrap(np.angle(h0_ripple))),
        label="ripple",
        alpha=0.3,
    )

    plt.axvline(x=f1)
    plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$\mathrm{d}\Phi/\mathrm{d}f$")
    plt.savefig(
        "../figures/full_waveform_PhenomX_phase_gradient.pdf", bbox_inches="tight"
    )


def random_match_waveforms(n=100):
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
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

        events = {
            "Mc": np.array([Mc]),
            "dL": np.array([1000.0]),
            "iota": np.array([0.0]),
            "eta": np.array([eta]),
            "chi1z": np.array([s1]),
            "chi2z": np.array([s2]),
            "Lambda1": np.array([0.0]),
            "Lambda2": np.array([0.0]),
        }

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
        del_f = 0.125

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
        approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")

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

        theta_ripple = np.array([Mc, eta, s1, s2, dist_mpc, tc, phic])
        h0_ripple = IMRPhenomXAS.gen_IMRPhenomXAS(fs, theta_ripple, f_ref)
        # tmpWF = waveforms.IMRPhenomXAS()
        # hp_wfpy, hp_wfpy = tmpWF.hphc(fs, **events)
        # phase_wf4py = waveforms.IMRPhenomXAS().Phi(fs, **events)
        # Amp_wf4py = waveforms.IMRPhenomXAS().Ampl(fs, **events)
        # h0_wfpy = Amp_wf4py * np.exp(1j * phase_wf4py)
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
    # np.savetxt("ripple_phenomXAS_matches.txt", np.c_[thetas, matches])

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

    plt.savefig(
        "../figures/test_match_vs_lalsuite_qchieff_XAS.pdf", bbox_inches="tight"
    )

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


def plot_waveforms():
    # Get a frequency domain waveform
    # source parameters
    m1_msun = 15.0
    m2_msun = 15.0
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

    # theta = np.array([m1_msun, m2_msun, chi1[2], chi2[2]])
    f_l = 32.0
    f_u = 1024.0
    del_f = 0.0125
    fs = np.arange(f_l, f_u, del_f)

    # coeffs = IMRPhenomD_utils.get_coeffs(theta)
    # _, _, f3, f4, _, _ = IMRPhenomD_utils.get_transition_frequencies(
    #     theta, coeffs[5], coeffs[6]
    # )

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")

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

    hp_ripple, hc_ripple = IMRPhenomXAS.gen_IMRPhenomXAS_polar(fs, theta_ripple, f_ref)

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

    # plt.axvline(x=f3, ls="--")
    # plt.axvline(x=f4, ls="--")
    # plt.legend()
    plt.xlim(0, 300)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Strain")
    plt.savefig("../figures/waveform_comparison_hp_XAS.pdf", bbox_inches="tight")

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

    # plt.axvline(x=f3, ls="--")
    # plt.axvline(x=f4, ls="--")
    plt.legend()
    plt.xlim(0, 300)
    plt.xlabel("Frequency")
    plt.ylabel("hf")
    plt.savefig("../figures/waveform_comparison_hc_XAS.pdf", bbox_inches="tight")

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


if __name__ == "__main__":
    # test_phase_phenomXAS()
    # test_gen_phenomXAS()
    # test_amplitude_XAS()
    # test_full_waveform_XAS()
    random_match_waveforms()
    # plot_waveforms()
