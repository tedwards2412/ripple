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

# plt.style.use("../plot_style.mplstyle")
import numpy as np
import cProfile
import lalsimulation as lalsim
from ripple import ms_to_Mc_eta
import lal

# SPOT TEST
m1 = 6.0
m2 = 10.0
f_ref = 30
phiRef = 0.0
incl = jnp.pi / 2.0
s1x = 0.5
s1y = -0.2
s1z = -0.5
s2x = 0.1
s2y = 0.6
s2z = 0.5

m1_SI = m1 * MSUN
m2_SI = m2 * MSUN
(
    chi1_l,
    chi2_l,
    chip,
    thetaJN,
    alpha0,
    phi_aligned,
    zeta_polariz,
) = IMRPhenomP.LALtoPhenomP(
    m1_SI, m2_SI, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z
)

print("thetaJN: ", thetaJN)
print(
    "Output from LALtoPhenomP: ",
    chi1_l,
    chi2_l,
    chip,
    thetaJN,
    alpha0,
    phi_aligned,
    zeta_polariz,
)

finspin = IMRPhenomP.FinalSpin_inplane(m2, m1, chi2_l, chi1_l, chip)
print("finspin func output: ", finspin)

# The following codes are directly copied from phenompcore
q = m2 / m1  # q>=1
M = m1 + m2
chi_eff = (m1 * chi1_l + m2 * chi2_l) / M
chil = (1.0 + q) / q * chi_eff
eta = m1 * m2 / (M * M)
m_sec = M * gt
piM = jnp.pi * m_sec

angcoeffs = IMRPhenomP.ComputeNNLOanglecoeffs(q, chil, chip)

print("ComputeNNLO result: ", angcoeffs)

omega_ref = piM * f_ref
logomega_ref = jnp.log(omega_ref)
omega_ref_cbrt = (piM * f_ref) ** (1 / 3)  # == v0
omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

alphaNNLOoffset = (
    angcoeffs["alphacoeff1"] / omega_ref
    + angcoeffs["alphacoeff2"] / omega_ref_cbrt2
    + angcoeffs["alphacoeff3"] / omega_ref_cbrt
    + angcoeffs["alphacoeff4"] * logomega_ref
    + angcoeffs["alphacoeff5"] * omega_ref_cbrt
)
epsilonNNLOoffset = (
    angcoeffs["epsiloncoeff1"] / omega_ref
    + angcoeffs["epsiloncoeff2"] / omega_ref_cbrt2
    + angcoeffs["epsiloncoeff3"] / omega_ref_cbrt
    + angcoeffs["epsiloncoeff4"] * logomega_ref
    + angcoeffs["epsiloncoeff5"] * omega_ref_cbrt
)


Y2m2 = IMRPhenomP.SpinWeightedY(thetaJN, 0, -2, 2, -2)
Y2m1 = IMRPhenomP.SpinWeightedY(thetaJN, 0, -2, 2, -1)
Y20 = IMRPhenomP.SpinWeightedY(thetaJN, 0, -2, 2, -0)
Y21 = IMRPhenomP.SpinWeightedY(thetaJN, 0, -2, 2, 1)
Y22 = IMRPhenomP.SpinWeightedY(thetaJN, 0, -2, 2, 2)
Y2 = [Y2m2, Y2m1, Y20, Y21, Y22]
print("harmonics: ", Y2[0], Y2[1], Y2[2], Y2[3], Y2[4])

print("offsets: ", alphaNNLOoffset, epsilonNNLOoffset)
fake_hPhenom = jnp.array([1, 1 + 3j, 0.04 - 95j, -3.87 - 0.001j])
fake_fHz = jnp.array([100, 25, 96.699, 238.75565])
print("parameters of twistup:", eta, chi1_l, chi2_l, chip, M)
hp, hc = IMRPhenomP.PhenomPCoreTwistUp(
    fake_fHz,
    fake_hPhenom,
    eta,
    chi1_l,
    chi2_l,
    chip,
    M,
    angcoeffs,
    Y2,
    alphaNNLOoffset - alpha0,
    epsilonNNLOoffset,
)
for i in range(len(fake_fHz)):
    print(i, "interation:")
    print(hp[i], hc[i])
