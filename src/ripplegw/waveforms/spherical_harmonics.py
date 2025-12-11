import jax
import jax.numpy as jnp
import math
from ..typing import Array


def compute_sminus2_l2(theta, m):
    """
    Spin -2 weighted spherical harmonic for l=2, phi=0.
    theta: float or array
    m: integer in [-2, -1, 0, 1, 2]
    """
    
    # Compute the fac factor based on m
    harmonics = jnp.where(m == -2, jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1.0 - jnp.cos(theta))**2,
                    jnp.where(m == -1, jnp.sqrt(5.0 / (16.0 * jnp.pi)) * jnp.sin(theta) * (1.0 - jnp.cos(theta)),
                        jnp.where(m == 0,  jnp.sqrt(15.0 / (32.0 * jnp.pi)) * jnp.sin(theta)**2,
                            jnp.where(m == 1,  jnp.sqrt(5.0 / (16.0 * jnp.pi)) * jnp.sin(theta) * (1.0 + jnp.cos(theta)),
                                jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1.0 + jnp.cos(theta))**2
                        ))))  
    
    return harmonics


def compute_sminus2_l3(theta, m):
    """
    Spin -2 weighted spherical harmonic for l=3, phi=0.
    theta: scalar or array
    m: integer in [-3, 3]
    """
    
    harmonics = jnp.where(m == -3, jnp.sqrt(21.0 / (2.0 * jnp.pi)) * jnp.cos(theta / 2.0) * (jnp.sin(theta / 2.0) ** 5),
                    jnp.where(m == -2, jnp.sqrt(7.0 / (4.0 * jnp.pi)) * (2.0 + 3.0 * jnp.cos(theta)) * (jnp.sin(theta / 2.0) ** 4),
                        jnp.where(m == -1, jnp.sqrt(35.0 / (2.0 * jnp.pi)) * (jnp.sin(theta) + 4.0 * jnp.sin(2.0*theta) - 3.0 * jnp.sin(3.0*theta)) / 32.0,
                            jnp.where(m == 0,  (jnp.sqrt(105.0 / (2.0 * jnp.pi)) * jnp.cos(theta) * (jnp.sin(theta) ** 2)) / 4.0,
                                jnp.where(m == 1, -jnp.sqrt(35.0 / (2.0 * jnp.pi)) * (jnp.sin(theta) - 4.0 * jnp.sin(2.0*theta) - 3.0 * jnp.sin(3.0*theta)) / 32.0,
                                    jnp.where(m == 2, jnp.sqrt(7.0 / jnp.pi) * (jnp.cos(theta/2.0)**4) * (-2.0 + 3.0 * jnp.cos(theta)) / 2.0,
                                        -jnp.sqrt(21.0 / (2.0 * jnp.pi)) * (jnp.cos(theta/2.0)**5) * jnp.sin(theta/2.0)
                                ))))))
    
    return harmonics


def compute_sminus2_l4(theta, m):
    """
    Spin -2 weighted spherical harmonic for l=4, phi=0.
    theta: scalar or array
    m: integer in [-4, 4]
    """
    
    harmonics = jnp.where(m == -4, 3.0 * jnp.sqrt(7.0 / jnp.pi) * (jnp.cos(theta/2.0)**2) * (jnp.sin(theta/2.0)**6),
                    jnp.where(m == -3, 3.0 * jnp.sqrt(7.0 / (2.0 * jnp.pi)) * jnp.cos(theta/2.0) * (1.0 + 2.0 * jnp.cos(theta)) * (jnp.sin(theta/2.0)**5),
                        jnp.where(m == -2, 3.0 * (9.0 + 14.0 * jnp.cos(theta) + 7.0 * jnp.cos(2.0*theta)) * (jnp.sin(theta/2.0)**4) / (4.0 * jnp.sqrt(jnp.pi)),
                            jnp.where(m == -1, 3.0 * (3.0 * jnp.sin(theta) + 2.0 * jnp.sin(2.0*theta) + 7.0 * jnp.sin(3.0*theta) - 7.0 * jnp.sin(4.0*theta)) / (32.0 * jnp.sqrt(2.0 * jnp.pi)),
                                jnp.where(m == 0, 3.0 * jnp.sqrt(5.0 / (2.0 * jnp.pi)) * (5.0 + 7.0 * jnp.cos(2.0*theta)) * (jnp.sin(theta)**2) / 16.0,
                                    jnp.where(m == 1, 3.0 * (3.0 * jnp.sin(theta) - 2.0 * jnp.sin(2.0*theta) + 7.0 * jnp.sin(3.0*theta) + 7.0 * jnp.sin(4.0*theta)) / (32.0 * jnp.sqrt(2.0 * jnp.pi)),
                                        jnp.where(m == 2, 3.0 * (jnp.cos(theta/2.0)**4) * (9.0 - 14.0 * jnp.cos(theta) + 7.0 * jnp.cos(2.0*theta)) / (4.0 * jnp.sqrt(jnp.pi)),
                                            jnp.where(m == 3, -3.0 * jnp.sqrt(7.0 / (2.0 * jnp.pi)) * (jnp.cos(theta/2.0)**5) * (-1.0 + 2.0 * jnp.cos(theta)) * jnp.sin(theta/2.0),
                                                3.0 * jnp.sqrt(7.0 / jnp.pi) * (jnp.cos(theta/2.0)**6) * (jnp.sin(theta/2.0)**2)
                                            ))))))))
    
    return harmonics



