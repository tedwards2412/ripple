"""
Typing definitions to be shared across files.
"""

import jax
import jax.numpy as jnp


# TODO: what type should this be?
PRNGKeyArray = jax.ringdown.PRNGKeyArray  # type: ignore
Array = jnp.ndarray
