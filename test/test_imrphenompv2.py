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
m1_test = 3*10**30
m2_test = 4*10**30
f_ref = 30
phi_ref = 0.3
incl = 0.1
s1x = 0.3
s1y = 0.0
s1z = 0.34
s2x = 0.1
s2y = 0.1
s2z = 0.5
M = m1_test + m2_test
fs = np.arange(f_ref, 800, 0.01)
#IMRPhenomP.PhenomPcore(m1_test, m2_test, f_ref, phi_ref, incl, s1x, s1y, s1z, s2x, s2y, s2z)
#phenom, phasing =IMRPhenomP.PhenomPOneFrequency(fs, m1_test/MSUN, m2_test/MSUN, s1z, s2z, phi_ref, M/MSUN)
#chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz =IMRPhenomP.LALtoPhenomP(m1_test, m2_test, f_ref, phi_ref, incl, s1x, s1y, s1z, s2x, s2y, s2z)
#print(chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz)
#print(phenom, phasing)
hps, hcs = IMRPhenomP.PhenomPcore(fs, m1_test, m2_test, f_ref, phi_ref, incl, s1x, s1y, s1z, s2x, s2y, s2z)

print(hps, hcs)