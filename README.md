# Ripple

A small `jax` package for differentiable and fast gravitational wave data analysis.

# Getting Started

### Installation

Ripple is still in its development stage and therefore has not been made available installable directly through pip yet. Instead,
one can install using 

```
git clone git@github.com:tedwards2412/ripple.git
cd ripple
pip3 install .
```

### Generating a waveform and its derivative

Generating a waveform is increadibly easy. Below is an example of calling the PhenomD waveform model
to get the h_+ and h_x polarizations of the waveform model

We start with some basic imports:

```python
from math import pi
import jax.numpy as jnp

from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
import matplotlib.pyplot as plt
from ripple import ms_to_Mc_eta
```

And now we can just set the parameters and call the waveform!

```python
# Get a frequency domain waveform
# source parameters

m1_msun = 20.0 # In solar masses
m2_msun = 19.0
chi1 = 0.5 # Dimensionless spin
chi2 = -0.5
tc = 0.0 # Time of coalescence in seconds
phic = 0.0 # Time of coalescence
dist_mpc = 440 # Distance to source in Mpc
inclination = 0.0 # Inclination Angle
polarization_angle = 0.2 # Polarization angle

# The PhenomD waveform model is parameterized with the chirp mass and symmetric mass ratio
Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

# These are the parametrs that go into the waveform generator
# Note that JAX does not give index errors, so if you pass in the
# the wrong array it will behave strangely
theta_ripple = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle])

# Now we need to generate the frequency grid
f_l = 24
f_u = 512
del_f = 0.01
fs = jnp.arange(f_l, f_u, del_f)

# And finally lets generate the waveform!
hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple)
```



