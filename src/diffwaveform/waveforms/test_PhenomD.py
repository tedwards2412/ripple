import numpy as np
import PhenomD
from nonstd_gwaves.waveforms import kappa_waveform
import matplotlib.pyplot as plt


def test_phenomD():
    theta = np.array([3.0, 1.0, 0.9, -0.9])
    f_l = 24
    f_u = 512
    del_f = 0.01
    f = np.arange(f_l, f_u, del_f)
    print(f)

    phase = PhenomD.Phase(f, theta)
    print(phase)

    TF2 = kappa_waveform(
        mass1=theta[0],
        mass2=theta[1],
        spin1z=theta[2],
        spin2z=theta[3],
        f_lower=f_l,
        f_upper=f_u,
        delta_f=del_f,
    )
    # print(TF2.Phif35PN())
    # plt.semilogy(TF2.frequencies, abs(TF2.Phif35PN()), label="old")
    # plt.semilogy(f, abs(phase), label="new")
    # plt.legend()
    # plt.savefig("test_phase.pdf")

    # coeffs = PhenomD.get_coeffs(theta)
    # print(coeffs)
    return None


def test_frequency_calc():
    theta = np.array([5.0, 4.0, 0.0, 0.0])
    m1, m2, chi1, chi2 = theta

    fRD, fdamp = PhenomD.get_fRD_fdamp(m1, m2, chi1, chi2)
    print(fRD, fdamp)
    return None


if __name__ == "__main__":
    test_phenomD()
    # test_frequency_calc()
