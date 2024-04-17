"""
Various constants, all in SI units.
"""

EulerGamma = 0.577215664901532860606512090082402431

MSUN = 1.988409902147041637325262574352366540e30  # kg
"""Solar mass"""

MRSUN = 1.476625038050124729627979840144936351e3
"""Geometrized nominal solar mass, m"""

G = 6.67430e-11  # m^3 / kg / s^2
"""Newton's gravitational constant"""

C = 299792458.0  # m / s
"""Speed of light"""

"""Pi"""
PI = 3.141592653589793238462643383279502884

TWO_PI = 6.283185307179586476925286766559005768

gt = G * MSUN / (C**3.0)
"""
G MSUN / C^3 in seconds
"""

m_per_Mpc = 3.085677581491367278913937957796471611e22
"""
Meters per Mpc.
"""

clightGpc = C / 3.0856778570831e22
"""
Speed of light in vacuum (:math:`c`), in gigaparsecs per second
"""
