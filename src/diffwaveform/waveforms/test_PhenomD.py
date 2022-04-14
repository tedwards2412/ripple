import numpy as np
import PhenomD
import PhenomD_utils

# from nonstd_gwaves.waveforms import kappa_waveform
import matplotlib.pyplot as plt


def test_phase_phenomD():
    theta = np.array([20.0, 19.99, -0.95, -0.95])
    Mf0 = 0.02
    Mf1 = 0.15
    f_l = Mf0 / ((theta[0] + theta[1]) * 4.92549094830932e-6)
    f_u = Mf1 / ((theta[0] + theta[1]) * 4.92549094830932e-6)
    del_f = 0.001
    f = np.arange(f_l, f_u, del_f)
    Mf = f * (theta[0] + theta[1]) * 4.92549094830932e-6
    del_Mf = np.diff(Mf)

    # Calculate the frequency regions
    coeffs = PhenomD_utils.get_coeffs(theta)
    f1, f2, _, _, f_RD, _ = PhenomD_utils.get_transition_frequencies(
        theta, coeffs[5], coeffs[6]
    )

    phase = PhenomD.Phase(f, theta)

    plt.figure(figsize=(7, 5))
    plt.plot(Mf, phase, label="PhenomD")
    plt.axvline(x=f1 * (theta[0] + theta[1]) * 4.92549094830932e-6)
    plt.axvline(x=f2 * (theta[0] + theta[1]) * 4.92549094830932e-6)
    plt.axvline(
        x=f_RD * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C1"
    )
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"$\Phi$")
    plt.savefig("../../../plots/test_phase_full_fig5check.pdf", bbox_inches="tight")

    phase_deriv = np.gradient(phase, del_Mf[0])
    plt.figure(figsize=(6, 5))
    plt.plot(Mf, phase_deriv, label="PhenomD")
    plt.axvline(x=f1 * (theta[0] + theta[1]) * 4.92549094830932e-6)
    plt.axvline(x=f2 * (theta[0] + theta[1]) * 4.92549094830932e-6)
    plt.axvline(
        x=f_RD * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C1"
    )
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"$\Phi^{\prime}$")
    plt.savefig(
        "../../../plots/test_phase_derivative_full_fig5check.pdf", bbox_inches="tight"
    )

    return None


def test_Amp_phenomD():
    theta = np.array([20.0, 19.99, -0.95, -0.95])
    Mf0 = 0.003
    Mf1 = 0.11
    f_l = Mf0 / ((theta[0] + theta[1]) * 4.92549094830932e-6)
    f_u = Mf1 / ((theta[0] + theta[1]) * 4.92549094830932e-6)
    del_f = 0.001
    f = np.arange(f_l, f_u, del_f)
    Mf = f * (theta[0] + theta[1]) * 4.92549094830932e-6

    # Calculate the frequency regions
    coeffs = PhenomD_utils.get_coeffs(theta)
    _, _, f3, f4, _, _ = PhenomD_utils.get_transition_frequencies(
        theta, coeffs[5], coeffs[6]
    )

    Amp = PhenomD.Amp(f, theta)

    plt.figure(figsize=(7, 5))
    plt.loglog(Mf, Amp, label="PhenomD")
    plt.axvline(
        x=((f3 + f4) / 2) * (theta[0] + theta[1]) * 4.92549094830932e-6,
        ls="--",
        color="C0",
    )
    plt.axvline(x=f3 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    plt.axvline(x=f4 * (theta[0] + theta[1]) * 4.92549094830932e-6, ls="--", color="C0")
    plt.legend()
    plt.xlabel(r"Mf")
    plt.ylabel(r"Amplitude")
    plt.savefig("../../../plots/test_Amp_full.pdf", bbox_inches="tight")

    return None


def test_frequency_calc():
    theta = np.array([5.0, 4.0, 0.0, 0.0])
    m1, m2, chi1, chi2 = theta

    fRD, fdamp = PhenomD_utils.get_fRD_fdamp(m1, m2, chi1, chi2)
    print("Ringdown and damping frequency:", fRD, fdamp)
    return None


if __name__ == "__main__":
    test_Amp_phenomD()
    test_phase_phenomD()
    test_frequency_calc()
