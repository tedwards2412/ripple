"""
Elliptic integral utilities implemented with JAX.
Some functions are based on https://github.com/tagordon/ellip/tree/main
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# relative error will be "less in magnitude than r"
R = 1.0e-15

# GSL constants
GSL_DBL_EPSILON = 2.2204460492503131e-16


@jax.jit
@jnp.vectorize
def rf(x, y, z):
    r"""JAX implementation of Carlson's :math:`R_\mathrm{F}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf
    Code taken from https://github.com/tagordon/ellip/tree/main

      Args:
        x: arraylike, real valued.
        y: arraylike, real valued.
        z: arraylike, real valued.

      Returns:
        The value of the integral :math:`R_\mathrm{F}`

      Notes:
        ``rf`` does not support complex-valued inputs.
        ``rf`` requires `jax.config.update("jax_enable_x64", True)`
    """

    xyz = jnp.array([x, y, z])
    a0 = jnp.sum(xyz) / 3.0
    v = jnp.max(jnp.abs(a0 - xyz))
    q = (3 * R) ** (-1 / 6) * v

    # cond = lambda s: s["f"] * q > jnp.abs(s["An"])
    def cond(s):
        return s["f"] * q > jnp.abs(s["An"])

    def body(s):

        xyz = s["xyz"]
        lam = jnp.sqrt(xyz[0] * xyz[1]) + jnp.sqrt(xyz[0] * xyz[2]) + jnp.sqrt(xyz[1] * xyz[2])

        s["An"] = 0.25 * (s["An"] + lam)
        s["xyz"] = 0.25 * (s["xyz"] + lam)
        s["f"] = s["f"] * 0.25

        return s

    s = {"f": 1, "An": a0, "xyz": xyz}
    s = jax.lax.while_loop(cond, body, s)

    x = (a0 - x) / s["An"] * s["f"]
    y = (a0 - y) / s["An"] * s["f"]
    z = -(x + y)
    e2 = x * y - z * z
    e3 = x * y * z

    return (1 - 0.1 * e2 + e3 / 14 + e2 * e2 / 24 - 3 * e2 * e3 / 44) / jnp.sqrt(s["An"])


@jax.jit
@jnp.vectorize
def ellipfinc(phi, k):
    r"""JAX implementation of the incomplete elliptic integral of the first kind
        Code taken from https://github.com/tagordon/ellip/tree/main

        .. math::

            \[F\left(\phi,k\right)=\int_{0}^{\phi}\frac{\,\mathrm{d}\theta}{\sqrt{1-k^{2}{%
    \sin}^{2}\theta}}]

          Args:
            phi: arraylike, real valued.
            k: arraylike, real valued.

          Returns:
            The value of the complete elliptic integral of the first kind, :math:`F(\phi, k)`

          Notes:
            ``ellipfinc`` does not support complex-valued inputs.
            ``ellipfinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi) ** 2
    return rf(c - 1, c - k**2, c)


@jax.jit
@jnp.vectorize
def gsl_sf_elljac_e(u, m):  # double * sn, double * cn, double * dn
    """
    JAX implementation of the Jacobi elliptic functions sn(u|m), cn(u|m), dn(u|m)
    Based on https://github.com/ampl/gsl/blob/master/specfunc/elljac.c
    """

    def little_m_branch(u, m):
        _ = m
        sn = jnp.sin(u)
        cn = jnp.cos(u)
        dn = 1.0
        return sn, cn, dn

    def little_m_min_one_branch(u, m):
        _ = m
        sn = jnp.tanh(u)
        cn = 1.0 / jnp.cosh(u)
        dn = cn
        return sn, cn, dn

    def main_branch(u, m):
        _n = 16
        mu = jnp.zeros(_n, dtype=jnp.float64).at[0].set(1.0)
        nu = jnp.zeros(_n, dtype=jnp.float64).at[0].set(jnp.sqrt(jnp.clip(1.0 - m, 0.0, jnp.inf)))

        def cond(state):
            n, mu, nu = state
            diff = jnp.abs(mu[n] - nu[n])
            tol = 4.0 * GSL_DBL_EPSILON * jnp.abs(mu[n] + nu[n])
            return (diff > tol) & (n < _n - 1)

        def body(state):
            n, mu, nu = state
            mu = mu.at[n + 1].set(0.5 * (mu[n] + nu[n]))
            nu = nu.at[n + 1].set(jnp.sqrt(mu[n] * nu[n]))
            return n + 1, mu, nu

        n, mu, nu = jax.lax.while_loop(cond, body, (0, mu, nu))

        sin_umu = jnp.sin(u * mu[n])
        cos_umu = jnp.cos(u * mu[n])

        def cos_branch(args):
            sin_umu, cos_umu, n, mu, nu = args
            c = jnp.zeros_like(mu).at[n].set(mu[n] * (sin_umu / cos_umu))
            d = jnp.zeros_like(mu).at[n].set(1.0)

            def cond(kcd):
                k, _, _ = kcd
                return k > 0

            def body(kcd):
                k, c, d = kcd
                k = k - 1
                r = (c[k + 1] * c[k + 1]) / mu[k + 1]
                c = c.at[k].set(d[k + 1] * c[k + 1])
                d = d.at[k].set((r + nu[k]) / (r + mu[k]))
                return k, c, d

            _, c, d = jax.lax.while_loop(cond, body, (n, c, d))
            dn = jnp.sqrt(jnp.clip(1.0 - m, 0.0, jnp.inf)) / d[0]
            cn = dn * jnp.sign(cos_umu) / jnp.hypot(1.0, c[0])
            sn = cn * c[0] / jnp.sqrt(jnp.clip(1.0 - m, 0.0, jnp.inf))
            return sn, cn, dn

        def sin_branch(args):
            sin_umu, cos_umu, n, mu, nu = args
            c = jnp.zeros_like(mu).at[n].set(mu[n] * (cos_umu / sin_umu))
            d = jnp.zeros_like(mu).at[n].set(1.0)

            def cond(kcd):
                k, _, _ = kcd
                return k > 0

            def body(kcd):
                k, c, d = kcd
                k = k - 1
                r = (c[k + 1] * c[k + 1]) / mu[k + 1]
                c = c.at[k].set(d[k + 1] * c[k + 1])
                d = d.at[k].set((r + nu[k]) / (r + mu[k]))
                return k, c, d

            _, c, d = jax.lax.while_loop(cond, body, (n, c, d))
            dn = d[0]
            sn = jnp.sign(sin_umu) / jnp.hypot(1.0, c[0])
            cn = c[0] * sn
            return sn, cn, dn

        return jax.lax.cond(
            jnp.abs(sin_umu) < jnp.abs(cos_umu), cos_branch, sin_branch, operand=(sin_umu, cos_umu, n, mu, nu)
        )

    sn, cn, dn = jax.lax.cond(
        jnp.abs(m) < 2.0 * GSL_DBL_EPSILON,
        lambda args: little_m_branch(*args),
        lambda args: jax.lax.cond(
            jnp.abs(args[1] - 1.0) < 2.0 * GSL_DBL_EPSILON,
            lambda inner_args: little_m_min_one_branch(*inner_args),
            lambda inner_args: main_branch(*inner_args),
            args,
        ),
        (u, m),
    )

    return sn, cn, dn
