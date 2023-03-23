from math import pi
from tqdm import tqdm
import jax.numpy as jnp
import time
import jax
from jax import grad, vmap

from ripple.waveforms import IMRPhenomXAS, IMRPhenomX_utils, IMRPhenomD
import matplotlib.pyplot as plt
from ripple.constants import gt

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
    # phase_ripple = IMRPhenomXAS.get_inspiral_phase(
    #     Mf, theta_ripple, IMRPhenomX_utils.PhenomX_coeff_table
    # )
    # phase_ripple = IMRPhenomXAS.get_mergerringdown_raw_phase(
    #     Mf, theta_ripple, IMRPhenomX_utils.PhenomX_coeff_table
    # )
    phase_ripple = IMRPhenomXAS.Phase(
        f, theta_ripple, IMRPhenomX_utils.PhenomX_coeff_table
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
        f, theta_intrinsic, theta_extrinsic, IMRPhenomX_utils.PhenomX_coeff_table
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

    amp_ripple = IMRPhenomXAS.Ampl(f, theta_intrinsic, D=dist_mpc)

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


if __name__ == "__main__":
    # test_phase_phenomXAS()
    # test_gen_phenomXAS()
    test_amplitude_XAS()
