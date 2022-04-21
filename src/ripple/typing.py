"""
Typing definitions to be shared across files.
"""

import jax
import jax.numpy as jnp


# TODO: what type should this be?
PRNGKeyArray = jax._src.prng.PRNGKeyArray  # type: ignore
Array = jnp.ndarray
