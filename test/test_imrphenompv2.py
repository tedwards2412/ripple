from math import pi
from tqdm import tqdm
import jax.numpy as jnp
import time
import jax
from jax import grad, vmap

from ripple import get_eff_pads, get_match_arr
from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
from ripple.waveforms import IMRPhenomP
import matplotlib.pyplot as plt
from ripple.constants import gt, MSUN

#plt.style.use("../plot_style.mplstyle")
import numpy as np
import cProfile
import lalsimulation as lalsim
from ripple import ms_to_Mc_eta
import lal
dist_mpc = 100
m1_test = 3*10**30
m2_test = 4*10**30
f_ref = 30
phi_ref = 0.3
incl = 0.1
s1x = 0.0
s1y = 0.0
s1z = 0.0
s2x = 0.0
s2y = 0.0
s2z = 0.0
M = m1_test + m2_test
f_l= f_ref
f_u = 400
fs = np.arange(f_ref, f_u, 0.01)
#IMRPhenomP.PhenomPcore(m1_test, m2_test, f_ref, phi_ref, incl, s1x, s1y, s1z, s2x, s2y, s2z)
#phenom, phasing =IMRPhenomP.PhenomPOneFrequency(fs, m1_test/MSUN, m2_test/MSUN, s1z, s2z, phi_ref, M/MSUN)
#chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz =IMRPhenomP.LALtoPhenomP(m1_test, m2_test, f_ref, phi_ref, incl, s1x, s1y, s1z, s2x, s2y, s2z)
#print(chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz)
#print(phenom, phasing)
hps, hcs = IMRPhenomP.PhenomPcore(fs, m1_test, m2_test, f_ref, phi_ref, incl, s1x, s1y, s1z, s2x, s2y, s2z)

print(hps)
distance = dist_mpc * 1e6 * lal.PC_SI

approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")
hp, hc = lalsim.SimInspiralChooseFDWaveform(
    m1_test,
    m2_test,
    s1x,
    s1y,
    s1z,
    s2x,
    s2y,
    s2z,
    distance,
    incl,
    phi_ref,
    0.0,
    0.0,
    0.0,
    0.01,
    f_ref,
    f_u,
    f_ref,
    None,
    approximant,
)
freq = np.arange(len(hp.data.data)) * 0.01
f_mask = (freq >= f_l) & (freq < f_u)
#plt.plot(fs, hps, "--", label="ripple")
print(hps[-1])
plt.plot(
    freq[f_mask], hp.data.data[f_mask]/hps, label="lalsuite",
)
print(hp.data.data[f_mask])
plt.legend()
plt.show()