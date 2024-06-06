# Ripple :ocean:

A small `jax` package for differentiable and fast gravitational wave data analysis. 

# Getting Started

Note that Thibeau Wouters and Kaze Wong will soon become the main developers of ripple so please contact them if you have comments/questions.

## Documentation

You can find the full documentation at [Read the Docs](https://ripplegw.readthedocs.io/).

### Installation

Both waveforms have been tested extensively and match `lalsuite` implementations to machine precision across all the parameter space. 

Ripple can be installed using 

```
pip3 install ripplegw
```

Note that by default we do not include enable float64 in `jax`` since we want allow users to use float32 to improve performance.
If you require float64, please include the following code at the start of the script:

```
from jax import config
config.update("jax_enable_x64", True)
```

See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html for other common `jax` gotchas.

### Supported waveforms

- IMRPhenomXAS (aligned spin)
- IMRPhenomD (aligned spin)
- IMRPhenomPv2 (Still finalizing sampling checks)
- TaylorF2 with tidal effects
- IMRPhenomD_NRTidalv2, verified for the low spin regime (chi1, chi2 < 0.05), further testing is required for higher spins

### Generating a waveform and its derivative

Generating a waveform is incredibly easy. Below is an example of calling the PhenomXAS waveform model
to get the h_+ and h_x polarizations of the waveform model

We start with some basic imports:

```python
import jax.numpy as jnp

from ripple.waveforms import IMRPhenomXAS
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

# The PhenomD waveform model is parameterized with the chirp mass and symmetric mass ratio
Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

# These are the parametrs that go into the waveform generator
# Note that JAX does not give index errors, so if you pass in the
# the wrong array it will behave strangely
theta_ripple = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])

# Now we need to generate the frequency grid
f_l = 24
f_u = 512
del_f = 0.01
fs = jnp.arange(f_l, f_u, del_f)
f_ref = f_l

# And finally lets generate the waveform!
hp_ripple, hc_ripple = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(fs, theta_ripple, f_ref)

# Note that we have not internally jitted the functions since this would
# introduce an annoying overhead each time the user evaluated the function with a different length frequency array
# We therefore recommend that the user jit the function themselves to accelerate evaluations. For example:

import jax

@jax.jit
def waveform(theta):
    return IMRPhenomXAS.gen_IMRPhenomXAS_hphc(fs, theta)
```



