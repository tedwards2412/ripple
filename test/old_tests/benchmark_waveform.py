"""This is the version as taken from Thomas Edward's Github repo"""

from math import pi
import jax.numpy as jnp
import time
import jax
from jax import vmap

import numpy as np
from ripple import ms_to_Mc_eta

from jax.config import config

config.update("jax_enable_x64", True)


def benchmark_waveform_call(IMRphenom: str):
    # Get a frequency domain waveform
    f_l = 16
    f_u = 512
    T = 16

    if IMRphenom == "IMRPhenomD":
        from ripple.waveforms.IMRPhenomD import (
            gen_IMRPhenomD_hphc as waveform_generator,
        )

    if IMRphenom == "IMRPhenomXAS":
        from ripple.waveforms.IMRPhenomXAS import (
            gen_IMRPhenomXAS_hphc as waveform_generator,
        )

    f_sampling = 4096
    delta_t = 1 / f_sampling
    tlen = int(round(T / delta_t))
    freqs = np.fft.rfftfreq(tlen, delta_t)
    fs = freqs[(freqs > f_l) & (freqs < f_u)]
    f_ref = f_l

    n = 1_0000
    # Intrinsic parameters
    m1 = np.random.uniform(1.0, 100.0, n)
    m2 = np.random.uniform(1.0, 100.0, n)
    s1 = np.random.uniform(-1.0, 1.0, n)
    s2 = np.random.uniform(-1.0, 1.0, n)

    # Extrinsic parameters
    dist_mpc = np.random.uniform(300.0, 1000.0, n)
    tc = np.random.uniform(-2.0, 2.0, n)
    phic = np.random.uniform(0.0, 0.0, n)
    inclination = np.random.uniform(0.0, pi, n)

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
        ]
    ).T

    @jax.jit
    def waveform(theta):
        return waveform_generator(fs, theta, f_ref)

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

    return None


if __name__ == "__main__":
    # Choose from "IMRPhenomD", "IMRPhenomXAS"
    benchmark_waveform_call("IMRPhenomD")
    None