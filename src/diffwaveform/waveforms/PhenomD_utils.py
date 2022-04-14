from typing import Tuple

import jax.numpy as jnp

from ..constants import gt
from ..typing import Array


# All these equations are defined in the papers
# Taken from https://github.com/scottperkins/phenompy/blob/master/utilities.py
def get_fRD_fdamp(m1, m2, chi1, chi2):
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s ** 2.0)
    S = (
        chi1 * m1_s ** 2 + chi2 * m2_s ** 2
    ) / M_s ** 2  # convert chi to spin s in z direction
    S_red = S / (1 - 2 * eta_s)

    a = (
        S
        + 2 * jnp.sqrt(3) * eta_s
        - 4.399 * eta_s ** 2
        + 9.397 * eta_s ** 3
        - 13.181 * eta_s ** 4
        + (-0.085 * S + 0.102 * S ** 2 - 1.355 * S ** 3 - 0.868 * S ** 4) * eta_s
        + (-5.837 * S - 2.097 * S ** 2 + 4.109 * S ** 3 + 2.064 * S ** 4) * eta_s ** 2
    )

    E_rad_ns = (
        0.0559745 * eta_s
        + 0.580951 * eta_s ** 2
        - 0.960673 * eta_s ** 3
        + 3.35241 * eta_s ** 4
    )

    E_rad = (
        E_rad_ns
        * (1 + S_red * (-0.00303023 - 2.00661 * eta_s + 7.70506 * eta_s ** 2))
        / (1 + S_red * (-0.67144 - 1.47569 * eta_s + 7.30468 * eta_s ** 2))
    )

    MWRD = 1.5251 - 1.1568 * (1 - a) ** 0.1292
    fRD = (1 / (2 * jnp.pi)) * (MWRD) / (M_s * (1 - E_rad))

    MWdamp = (1.5251 - 1.1568 * (1 - a) ** 0.1292) / (
        2 * (0.700 + 1.4187 * (1 - a) ** (-0.4990))
    )
    fdamp = (1 / (2 * jnp.pi)) * (MWdamp) / (M_s * (1 - E_rad))
    return fRD, fdamp


def get_transition_frequencies(
    theta: Array, gamma2: float, gamma3: float
) -> Tuple[float, float, float, float, float, float]:

    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    f_RD, f_damp = get_fRD_fdamp(m1, m2, chi1, chi2)

    # Phase transition frequencies
    f1 = 0.018 / (M * gt)
    f2 = f_RD / 2

    # Amplitude transition frequencies
    f3 = 0.014 / (M * gt)
    f4 = jnp.abs(f_RD + f_damp * gamma3 * (jnp.sqrt(1 - (gamma2 ** 2)) - 1) / gamma2)

    return f1, f2, f3, f4, f_RD, f_damp


def get_coeffs(theta: Array) -> Array:
    # Retrives the coefficients needed to produce the waveform

    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    chi_eff = (m1_s * chi1 + m2_s * chi2) / M_s
    chiPN = chi_eff - (38.0 * eta / 113.0) * (chi1 + chi2)

    def calc_coeff(i, eta, chiPN):
        coeff = (
            PhenomD_coeff_table[i, 0]
            + PhenomD_coeff_table[i, 1] * eta
            + (chiPN - 1.0)
            * (
                PhenomD_coeff_table[i, 2]
                + PhenomD_coeff_table[i, 3] * eta
                + PhenomD_coeff_table[i, 4] * (eta ** 2.0)
            )
            + (chiPN - 1.0) ** 2.0
            * (
                PhenomD_coeff_table[i, 5]
                + PhenomD_coeff_table[i, 6] * eta
                + PhenomD_coeff_table[i, 7] * (eta ** 2.0)
            )
            + (chiPN - 1.0) ** 3.0
            * (
                PhenomD_coeff_table[i, 8]
                + PhenomD_coeff_table[i, 9] * eta
                + PhenomD_coeff_table[i, 10] * (eta ** 2.0)
            )
        )

        return coeff

    rho1 = calc_coeff(0, eta, chiPN)
    rho2 = calc_coeff(1, eta, chiPN)
    rho3 = calc_coeff(2, eta, chiPN)
    v2 = calc_coeff(3, eta, chiPN)
    gamma1 = calc_coeff(4, eta, chiPN)
    gamma2 = calc_coeff(5, eta, chiPN)
    gamma3 = calc_coeff(6, eta, chiPN)
    sig1 = calc_coeff(7, eta, chiPN)
    sig2 = calc_coeff(8, eta, chiPN)
    sig3 = calc_coeff(9, eta, chiPN)
    sig4 = calc_coeff(10, eta, chiPN)
    beta1 = calc_coeff(11, eta, chiPN)
    beta2 = calc_coeff(12, eta, chiPN)
    beta3 = calc_coeff(13, eta, chiPN)
    a1 = calc_coeff(14, eta, chiPN)
    a2 = calc_coeff(15, eta, chiPN)
    a3 = calc_coeff(16, eta, chiPN)
    a4 = calc_coeff(17, eta, chiPN)
    a5 = calc_coeff(18, eta, chiPN)

    coeffs = jnp.array(
        [
            rho1,
            rho2,
            rho3,
            v2,
            gamma1,
            gamma2,
            gamma3,
            sig1,
            sig2,
            sig3,
            sig4,
            beta1,
            beta2,
            beta3,
            a1,
            a2,
            a3,
            a4,
            a5,
        ]
    )

    return coeffs


def get_delta0(f1, f2, f3, v1, v2, v3, d1, d3):
    return (
        -(d3 * f1 ** 2 * (f1 - f2) ** 2 * f2 * (f1 - f3) * (f2 - f3) * f3)
        + d1 * f1 * (f1 - f2) * f2 * (f1 - f3) * (f2 - f3) ** 2 * f3 ** 2
        + f3 ** 2
        * (
            f2
            * (f2 - f3) ** 2
            * (-4 * f1 ** 2 + 3 * f1 * f2 + 2 * f1 * f3 - f2 * f3)
            * v1
            + f1 ** 2 * (f1 - f3) ** 3 * v2
        )
        + f1 ** 2
        * (f1 - f2) ** 2
        * f2
        * (f1 * f2 - 2 * f1 * f3 - 3 * f2 * f3 + 4 * f3 ** 2)
        * v3
    ) / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (f2 - f3) ** 2)


def get_delta1(f1, f2, f3, v1, v2, v3, d1, d3):
    return (
        d3 * f1 * (f1 - f3) * (f2 - f3) * (2 * f2 * f3 + f1 * (f2 + f3))
        - (
            f3
            * (
                d1
                * (f1 - f2)
                * (f1 - f3)
                * (f2 - f3) ** 2
                * (2 * f1 * f2 + (f1 + f2) * f3)
                + 2
                * f1
                * (
                    f3 ** 4 * (v1 - v2)
                    + 3 * f2 ** 4 * (v1 - v3)
                    + f1 ** 4 * (v2 - v3)
                    + 4 * f2 ** 3 * f3 * (-v1 + v3)
                    + 2 * f1 ** 3 * f3 * (-v2 + v3)
                    + f1
                    * (
                        2 * f3 ** 3 * (-v1 + v2)
                        + 6 * f2 ** 2 * f3 * (v1 - v3)
                        + 4 * f2 ** 3 * (-v1 + v3)
                    )
                )
            )
        )
        / (f1 - f2) ** 2
    ) / ((f1 - f3) ** 3 * (f2 - f3) ** 2)


def get_delta2(f1, f2, f3, v1, v2, v3, d1, d3):
    return (
        d1
        * (f1 - f2)
        * (f1 - f3)
        * (f2 - f3) ** 2
        * (f1 * f2 + 2 * (f1 + f2) * f3 + f3 ** 2)
        - d3
        * (f1 - f2) ** 2
        * (f1 - f3)
        * (f2 - f3)
        * (f1 ** 2 + f2 * f3 + 2 * f1 * (f2 + f3))
        - 4 * f1 ** 2 * f2 ** 3 * v1
        + 3 * f1 * f2 ** 4 * v1
        - 4 * f1 * f2 ** 3 * f3 * v1
        + 3 * f2 ** 4 * f3 * v1
        + 12 * f1 ** 2 * f2 * f3 ** 2 * v1
        - 4 * f2 ** 3 * f3 ** 2 * v1
        - 8 * f1 ** 2 * f3 ** 3 * v1
        + f1 * f3 ** 4 * v1
        + f3 ** 5 * v1
        + f1 ** 5 * v2
        + f1 ** 4 * f3 * v2
        - 8 * f1 ** 3 * f3 ** 2 * v2
        + 8 * f1 ** 2 * f3 ** 3 * v2
        - f1 * f3 ** 4 * v2
        - f3 ** 5 * v2
        - (f1 - f2) ** 2
        * (
            f1 ** 3
            + f2 * (3 * f2 - 4 * f3) * f3
            + f1 ** 2 * (2 * f2 + f3)
            + f1 * (3 * f2 - 4 * f3) * (f2 + 2 * f3)
        )
        * v3
    ) / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (f2 - f3) ** 2)


def get_delta3(f1, f2, f3, v1, v2, v3, d1, d3):
    return (
        (d3 * (f1 - f3) * (2 * f1 + f2 + f3)) / (f2 - f3)
        - (d1 * (f1 - f3) * (f1 + f2 + 2 * f3)) / (f1 - f2)
        + (
            2
            * (
                f3 ** 4 * (-v1 + v2)
                + 2 * f1 ** 2 * (f2 - f3) ** 2 * (v1 - v3)
                + 2 * f2 ** 2 * f3 ** 2 * (v1 - v3)
                + 2 * f1 ** 3 * f3 * (v2 - v3)
                + f2 ** 4 * (-v1 + v3)
                + f1 ** 4 * (-v2 + v3)
                + 2
                * f1
                * f3
                * (f3 ** 2 * (v1 - v2) + f2 ** 2 * (v1 - v3) + 2 * f2 * f3 * (-v1 + v3))
            )
        )
        / ((f1 - f2) ** 2 * (f2 - f3) ** 2)
    ) / (f1 - f3) ** 3


def get_delta4(f1, f2, f3, v1, v2, v3, d1, d3):
    return (
        -(d3 * (f1 - f2) ** 2 * (f1 - f3) * (f2 - f3))
        + d1 * (f1 - f2) * (f1 - f3) * (f2 - f3) ** 2
        - 3 * f1 * f2 ** 2 * v1
        + 2 * f2 ** 3 * v1
        + 6 * f1 * f2 * f3 * v1
        - 3 * f2 ** 2 * f3 * v1
        - 3 * f1 * f3 ** 2 * v1
        + f3 ** 3 * v1
        + f1 ** 3 * v2
        - 3 * f1 ** 2 * f3 * v2
        + 3 * f1 * f3 ** 2 * v2
        - f3 ** 3 * v2
        - (f1 - f2) ** 2 * (f1 + 2 * f2 - 3 * f3) * v3
    ) / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (f2 - f3) ** 2)


PhenomD_coeff_table = jnp.array(
    [
        [  # rho1
            3931.9,
            -17395.8,
            3132.38,
            343966.0,
            -1.21626e6,
            -70698,
            1.38391e6,
            -3.96628e6,
            -60017.5,
            803515.0,
            -2.09171e6,
        ],
        [  # rho2
            -40105.5,
            112253.0,
            23561.7,
            -3.47618e6,
            1.13759e7,
            754313.0,
            -1.30848e7,
            3.64446e7,
            596227.0,
            -7.42779e6,
            1.8929e7,
        ],
        [  # rho3
            83208.4,
            -191238.0,
            -210916.0,
            8.71798e6,
            -2.69149e7,
            -1.98898e6,
            3.0888e7,
            -8.39087e7,
            -1.4535e6,
            1.70635e7,
            -4.27487e7,
        ],
        [  # v2
            0.814984,
            2.57476,
            1.16102,
            -2.36278,
            6.77104,
            0.757078,
            -2.72569,
            7.11404,
            0.176693,
            -0.797869,
            2.11624,
        ],
        [  # gamma1
            0.0069274,
            0.0302047,
            0.00630802,
            -0.120741,
            0.262716,
            0.00341518,
            -0.107793,
            0.27099,
            0.000737419,
            -0.0274962,
            0.0733151,
        ],
        [  # gamma2
            1.01034,
            0.000899312,
            0.283949,
            -4.04975,
            13.2078,
            0.103963,
            -7.02506,
            24.7849,
            0.030932,
            -2.6924,
            9.60937,
        ],
        [  # gamma2
            1.30816,
            -0.00553773,
            -0.0678292,
            -0.668983,
            3.40315,
            -0.0529658,
            -0.992379,
            4.82068,
            -0.00613414,
            -0.384293,
            1.75618,
        ],
        [  # sig1
            2096.55,
            1463.75,
            1312.55,
            18307.3,
            -43534.1,
            -833.289,
            32047.3,
            -108609.0,
            452.251,
            8353.44,
            -44531.3,
        ],
        [  # sig2
            -10114.1,
            -44631.0,
            -6541.31,
            -266959.0,
            686328.0,
            3405.64,
            -437508.0,
            1.63182e6,
            -7462.65,
            -114585.0,
            674402.0,
        ],
        [  # sig3
            22933.7,
            230960.0,
            14961.1,
            1.19402e6,
            -3.10422e6,
            -3038.17,
            1.87203e6,
            -7.30915e6,
            42738.2,
            467502.0,
            -3.06485e6,
        ],
        [  # sig4
            -14621.7,
            -377813.0,
            -9608.68,
            -1.71089e6,
            4.33292e6,
            -22366.7,
            -2.50197e6,
            1.02745e7,
            -85360.3,
            -570025.0,
            4.39684e6,
        ],
        [  # beta1
            97.8975,
            -42.6597,
            153.484,
            -1417.06,
            2752.86,
            138.741,
            -1433.66,
            2857.74,
            41.0251,
            -423.681,
            850.359,
        ],
        [  # beta2
            -3.2827,
            -9.05138,
            -12.4154,
            55.4716,
            -106.051,
            -11.953,
            76.807,
            -155.332,
            -3.41293,
            25.5724,
            -54.408,
        ],
        [  # beta3
            -2.51564e-5,
            1.97503e-5,
            -1.83707e-5,
            2.18863e-5,
            8.25024e-5,
            7.15737e-6,
            -5.578e-5,
            1.91421e-4,
            5.44717e-6,
            -3.22061e-5,
            7.97402e-5,
        ],
        [  # a1
            43.3151,
            638.633,
            -32.8577,
            2415.89,
            -5766.88,
            -61.8546,
            2953.97,
            -8986.29,
            -21.5714,
            981.216,
            -3239.57,
        ],
        [  # a2
            -0.0702021,
            -0.162698,
            -0.187251,
            1.13831,
            -2.83342,
            -0.17138,
            1.71975,
            -4.53972,
            -0.0499834,
            0.606207,
            -1.68277,
        ],
        [  # a3
            9.59881,
            -397.054,
            16.2021,
            -1574.83,
            3600.34,
            27.0924,
            -1786.48,
            5152.92,
            11.1757,
            -577.8,
            1808.73,
        ],
        [  # a4
            -0.0298949,
            1.40221,
            -0.0735605,
            0.833701,
            0.224001,
            -0.0552029,
            0.566719,
            0.718693,
            -0.0155074,
            0.157503,
            0.210768,
        ],
        [  # a5
            0.997441,
            -0.00788445,
            -0.0590469,
            1.39587,
            -4.51663,
            -0.0558534,
            1.75166,
            -5.99021,
            -0.0179453,
            0.59651,
            -2.06089,
        ],
    ]
)
