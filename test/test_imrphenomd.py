from math import pi
from tqdm import tqdm

from diffwaveform import get_eff_pads, get_match_arr
from diffwaveform.waveforms import PhenomD, PhenomD_utils
import matplotlib.pyplot as plt
import numpy as np

# Only use pycbc if it's installed
try:
    from pycbc import waveform

    PYCBC_INSTALLED = True
except ModuleNotFoundError:
    PYCBC_INSTALLED = False


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
    plt.savefig("../figures/test_phase_full_fig5check.pdf", bbox_inches="tight")

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
        "../figures/test_phase_derivative_full_fig5check.pdf", bbox_inches="tight"
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
    plt.savefig("../figures/test_Amp_full.pdf", bbox_inches="tight")

    return None


def test_frequency_calc():
    theta = np.array([5.0, 4.0, 0.0, 0.0])
    m1, m2, chi1, chi2 = theta

    fRD, fdamp = PhenomD_utils.get_fRD_fdamp(m1, m2, chi1, chi2)
    print("Ringdown and damping frequency:", fRD, fdamp)
    return None


if PYCBC_INSTALLED:

    def plot_waveforms():
        # Get a frequency domain waveform
        theta = np.array([29.2470611, 9.23337794, -0.409828828, -0.522292775])
        f_l = 40
        f_u = 1024
        df = 1.0 / 1000.0

        sptilde, sctilde = waveform.get_fd_waveform(
            approximant="IMRPhenomD",
            mass1=theta[0],
            mass2=theta[1],
            spin1z=theta[2],
            spin2z=theta[3],
            delta_f=df,
            f_lower=f_l,
            f_final=f_u,
        )

        del_f = 0.001
        fs = np.arange(f_l, f_u, del_f)

        Amp = PhenomD.Amp(fs, theta)
        Phase = PhenomD.Phase(fs, theta)

        # FIXME: Seems to be an extra factor of pi needed here
        wv = (Amp / pi) * np.exp(-1.0j * Phase)

        plt.figure(figsize=(15, 5))
        plt.plot(
            sptilde.sample_frequencies[sptilde.sample_frequencies >= f_l],
            sptilde[sptilde.sample_frequencies >= f_l],
            label="pycbc",
        )
        plt.plot(fs, wv, label="ripple", alpha=0.3)
        plt.legend()
        plt.xlabel("Frequency")
        plt.ylabel("hf")
        plt.savefig("../figures/waveform_comparison.pdf", bbox_inches="tight")

        pad_low, pad_high = get_eff_pads(fs)

        print(
            get_match_arr(
                pad_low,
                pad_high,
                np.ones_like(fs),
                wv,
                sptilde[
                    (sptilde.sample_frequencies >= f_l)
                    & (sptilde.sample_frequencies < f_u)
                ],
            )
        )

    def random_match_waveforms(n=100):
        # Get a frequency domain waveform
        f_l = 40
        f_u = 1024
        df = 1.0 / 1000.0
        thetas = []
        matches = []

        for i in tqdm(range(n)):
            m1 = np.random.uniform(1.0, 50.0)
            m2 = np.random.uniform(1.0, 50.0)
            s1 = np.random.uniform(-1.0, 1.0)
            s2 = np.random.uniform(-1.0, 1.0)

            if m1 < m2:
                theta = np.array([m2, m1, s1, s2])
            elif m1 > m2:
                theta = np.array([m1, m2, s1, s2])
            else:
                raise ValueError("Something went wrong with the parameters")
            sptilde, sctilde = waveform.get_fd_waveform(
                approximant="IMRPhenomD",
                mass1=theta[0],
                mass2=theta[1],
                spin1z=theta[2],
                spin2z=theta[3],
                delta_f=df,
                f_lower=f_l,
                f_final=f_u,
            )

            del_f = 0.001
            fs = np.arange(f_l, f_u, del_f)

            Amp = PhenomD.Amp(fs, theta)
            Phase = PhenomD.Phase(fs, theta)

            pad_low, pad_high = get_eff_pads(fs)

            # FIXME: Seems to be an extra factor of pi needed here
            wv = (Amp / pi) * np.exp(-1.0j * Phase)

            matches.append(
                get_match_arr(
                    pad_low,
                    pad_high,
                    np.ones_like(fs),
                    wv,
                    sptilde[
                        (sptilde.sample_frequencies >= f_l)
                        & (sptilde.sample_frequencies < f_u)
                    ],
                )
            )
            thetas.append(theta)

        thetas = np.array(thetas)
        matches = np.array(matches)

        plt.figure(figsize=(7, 5))
        cm = plt.cm.get_cmap("inferno")
        sc = plt.scatter(thetas[:, 0], thetas[:, 1], c=matches, cmap=cm)
        plt.colorbar(sc)
        plt.xlabel("m1")
        plt.ylabel("m2")
        plt.savefig("../figures/test_match_vs_pycbc_m1m2.pdf", bbox_inches="tight")

        plt.figure(figsize=(7, 5))
        cm = plt.cm.get_cmap("inferno")
        sc = plt.scatter(thetas[:, 2], thetas[:, 3], c=matches, cmap=cm)
        plt.colorbar(sc)
        plt.xlabel("s1")
        plt.ylabel("s2")
        plt.savefig("../figures/test_match_vs_pycbc_s1s2.pdf", bbox_inches="tight")

        print(thetas, matches)


if __name__ == "__main__":
    test_Amp_phenomD()
    test_phase_phenomD()
    test_frequency_calc()
    if PYCBC_INSTALLED:
        plot_waveforms()
        random_match_waveforms(n=10)
