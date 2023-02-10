from math import pi
from tqdm import tqdm
import jax.numpy as jnp
import time
import jax
from jax import grad, vmap

from ripple.waveforms import IMRPhenomXAS, IMRPhenomX_utils
import matplotlib.pyplot as plt
from ripple.constants import gt

plt.style.use("../plot_style.mplstyle")
import numpy as np
import cProfile
import lalsimulation as lalsim
from ripple import ms_to_Mc_eta
import lal


def test_phase_phenomXAS():
    theta = np.array([20, 19.0, 0.0, 0.0])
    Mc, eta = ms_to_Mc_eta(jnp.array([theta[0], theta[1]]))
    tc = 0.0
    phic = 0.0
    dist_mpc = 440
    inclination = np.pi / 2.0
    phi_ref = 0

    f_l = 400
    f_u = 1000
    del_f = 0.0125

    f_l_idx = round(f_l / del_f)
    f_u_idx = f_u // del_f
    f_l = f_l_idx * del_f
    f_u = f_u_idx * del_f
    f = np.arange(f_l_idx, f_u_idx) * del_f

    # f = np.arange(f_l, f_u, del_f)
    Mf = f * (theta[0] + theta[1]) * 4.92549094830932e-6
    # del_Mf = np.diff(Mf)

    # Calculate the frequency regions
    theta_ripple = np.array([theta[0], theta[1], theta[2], theta[3]])

    f_ref = f_l
    # phase_ripple = IMRPhenomXAS.get_inspiral_phase(
    #     Mf, theta_ripple, IMRPhenomX_utils.PhenomX_coeff_table
    # )
    phase_ripple = IMRPhenomXAS.get_mergerringdown_raw_phase(
        Mf, theta_ripple, IMRPhenomX_utils.PhenomX_coeff_table
    )

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
        freq[f_mask], np.unwrap(np.angle(h0_lalsuite)), label="lalsuite",
    )

    plt.plot(
        f,
        phase_ripple + f * 13 - (phase_ripple + f * 13)[0],
        label="ripple",
        alpha=0.3,
    )
    # plt.axvline(x=f1)
    # plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$\Phi$")
    plt.savefig("../figures/inspiral_phase_PhenomX.pdf", bbox_inches="tight")

    print(
        "Ripple Phase:", phase_ripple,
    )
    print("Lalsuite Phase:", np.unwrap(np.angle(h0_lalsuite)))

    # print(
    #     "Difference:",
    #     np.unwrap(np.angle(hp_ripple)) - np.unwrap(np.angle(hp.data.data))[f_mask],
    # )

    plt.figure(figsize=(7, 5))
    plt.plot(
        f, (np.unwrap(np.angle(h0_lalsuite))) - phase_ripple, label="phase difference",
    )
    # plt.axvline(x=f1)
    # plt.axvline(x=f2)
    plt.legend()
    plt.xlabel(r"f")
    plt.ylabel(r"$\Phi_1 - \Phi_2$")
    plt.savefig("../figures/test_phasedifference_PhenomX.pdf", bbox_inches="tight")

    # phase_deriv_lal = np.gradient(np.unwrap(np.angle(hp.data.data))[f_mask], del_Mf[0])
    # phase_deriv = np.gradient(np.unwrap(np.angle(hp_ripple)), del_Mf[0])
    # plt.figure(figsize=(6, 5))
    # print("testing here", phase_deriv, phase_deriv_lal)
    # plt.plot(f, phase_deriv, label="ripple", alpha=0.3)
    # plt.plot(
    #     freq[f_mask],  # * ((theta[0] + theta[1]) * 4.92549094830932e-6),
    #     phase_deriv_lal,
    #     label="lalsuite",
    #     alpha=0.3,
    #     ls="--",
    # )
    # plt.legend()
    # plt.xlabel(r"f")
    # plt.xlim(100, 120)
    # plt.ylim(100, 200)
    # plt.ylabel(r"$\Phi^{\prime}$")
    # plt.savefig("../figures/test_phase_derivative_full.pdf", bbox_inches="tight")

    plt.figure(figsize=(6, 5))
    diff = (np.unwrap(np.angle(h0_lalsuite))) - phase_ripple
    plt.plot(Mf, np.gradient(diff, np.diff(f)[0]), label="difference")
    test_scaling = Mf ** (-1 / 3)

    plt.plot(
        Mf,
        (test_scaling) + (np.gradient(diff, np.diff(f)[0])[0] - test_scaling[0]),
        label="test_scaling 4/3",
    )
    plt.legend()
    plt.xlabel(r"f")
    # plt.xlim(100, 120)
    # plt.ylim(-0.00050, -0.00025)
    plt.ylabel(r"$(\Phi_1-\Phi_2)^{\prime}$")
    plt.savefig(
        "../figures/test_phase_difference_derivative_phenomX.pdf", bbox_inches="tight"
    )

    return None


if __name__ == "__main__":
    test_phase_phenomXAS()
    None
