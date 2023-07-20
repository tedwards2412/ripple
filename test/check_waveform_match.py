import jax.numpy as jnp
import jax
from ripple import get_eff_pads, get_match_arr
from tqdm import tqdm

import numpy as np
from ripple import ms_to_Mc_eta
import lalsimulation as lalsim
import lal

from jax.config import config

config.update("jax_enable_x64", True)


def random_match_waveforms(n, IMRphenom):
    # Get a frequency domain waveform
    f_l = 16
    f_u = 1024
    T = 16

    if IMRphenom == "IMRPhenomD":
        from ripple.waveforms.IMRPhenomD import (
            gen_IMRPhenomD as waveform_generator,
        )

    if IMRphenom == "IMRPhenomXAS":
        from ripple.waveforms.IMRPhenomXAS import (
            gen_IMRPhenomXAS as waveform_generator,
        )

    f_sampling = 4096
    delta_t = 1 / f_sampling
    tlen = int(round(T / delta_t))
    freqs = np.fft.rfftfreq(tlen, delta_t)
    df = freqs[1] - freqs[0]
    fs = freqs[(freqs > f_l) & (freqs < f_u)]
    f_ref = f_l

    @jax.jit
    def waveform(theta):
        return waveform_generator(fs, theta, f_ref)

    # Get a frequency domain waveform
    thetas = []
    matches = []
    f_ASD, ASD = np.loadtxt("O3Livingston.txt", unpack=True)

    for i in tqdm(range(n)):
        m1 = np.random.uniform(1.0, 100.0)
        m2 = np.random.uniform(1.0, 100.0)
        s1 = np.random.uniform(-1.0, 1.0)
        s2 = np.random.uniform(-1.0, 1.0)

        tc = 0.0
        phic = 0.0
        dist_mpc = 440
        inclination = np.pi / 2.0
        phi_ref = 0

        if m1 < m2:
            theta = np.array([m2, m1, s1, s2])
        elif m1 > m2:
            theta = np.array([m1, m2, s1, s2])
        else:
            raise ValueError("Something went wrong with the parameters")
        approximant = lalsim.SimInspiralGetApproximantFromString(IMRphenom)

        f_ref = f_l
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        distance = dist_mpc * 1e6 * lal.PC_SI

        hp, _ = lalsim.SimInspiralChooseFDWaveform(
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
            df,
            f_l,
            f_u,
            f_ref,
            None,
            approximant,
        )
        freqs_lal = np.arange(len(hp.data.data)) * df

        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

        theta_ripple = np.array([Mc, eta, s1, s2, dist_mpc, tc, phic])
        h0_ripple = waveform(theta_ripple)
        pad_low, pad_high = get_eff_pads(fs)
        PSD_vals = np.interp(fs, f_ASD, ASD) ** 2

        mask_lal = (freqs_lal > f_l) & (freqs_lal < f_u)
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

    thetas = np.array(thetas)
    matches = np.array(matches)

    print("Mean match:", np.mean(matches))
    print("Median match:", np.median(matches))
    print("Minimum match:", np.min(matches))

    return None


if __name__ == "__main__":
    # Choose from "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomPv2"
    random_match_waveforms(1000, "IMRPhenomXAS")
    None
