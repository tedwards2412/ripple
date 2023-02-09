from typing import Tuple

import jax.numpy as jnp
import jax

from ..constants import gt, PI
from ..typing import Array

# Dimensionless cutoff frequency for PhenomXAS
f_CUT = 0.3


def get_cutoff_fs(m1, m2, chi1, chi2):
    # This function returns a variety of frequencies needed for computing IMRPhenomXAS
    # In particular, we have fRD, fdamp, fMECO, FISCO
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s**2.0)
    # m1Sq = m1_s * m1_s
    # m2Sq = m2_s * m2_s

    delta = jnp.sqrt(1.0 - 4.0 * eta_s)
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)

    chi_eff = mm1 * chi1 + mm2 * chi2

    eta2 = eta_s * eta_s
    eta3 = eta2 * eta_s
    eta4 = eta3 * eta_s
    S = (chi_eff - (38.0 / 113.0) * eta_s * (chi1 + chi2)) / (
        1.0 - (76.0 * eta_s / 113.0)
    )
    S2 = S * S
    S3 = S2 * S

    dchi = chi1 - chi2
    dchi2 = dchi * dchi

    StotR = (mm1**2.0 * chi1 + mm2**2.0 * chi2) / (mm1**2.0 + mm2**2.0)
    StotR2 = StotR * StotR
    StotR3 = StotR2 * StotR

    # First we need to calculate the dimensionless final spin and the radiated energy
    # From https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_utilities_8c_source.html
    # (((3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta2*eta)/(1 + 7.2388440419467335*eta)) + ((m1ByMSq + m2ByMSq)*totchi + ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta2*eta)*totchi + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta2*eta)*totchi2 + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta2*eta)*totchi2*totchi) / (1 + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta2*eta)*totchi)) + (0.3223660562764661*dchi*Seta*(1 + 9.332575956437443*eta)*eta2 - 0.059808322561702126*dchi*dchi*eta2*eta + 2.3170397514509933*dchi*Seta*(1 - 3.2624649875884852*eta)*eta2*eta*totchi))
    a = (
        (
            3.4641016151377544 * eta_s
            + 20.0830030082033 * eta2
            - 12.333573402277912 * eta3
        )
        / (1 + 7.2388440419467335 * eta_s)
        + (
            (mm1**2.0 + mm2**2.0) * StotR
            + (
                (
                    -0.8561951310209386 * eta_s
                    - 0.09939065676370885 * eta2
                    + 1.668810429851045 * eta3
                )
                * StotR
                + (
                    0.5881660363307388 * eta_s
                    - 2.149269067519131 * eta2
                    + 3.4768263932898678 * eta3
                )
                * StotR2
                + (
                    0.142443244743048 * eta_s
                    - 0.9598353840147513 * eta2
                    + 1.9595643107593743 * eta3
                )
                * StotR3
            )
            / (
                1
                + (
                    -0.9142232693081653
                    + 2.3191363426522633 * eta_s
                    - 9.710576749140989 * eta3
                )
                * StotR
            )
        )
        + (
            0.3223660562764661 * dchi * delta * (1 + 9.332575956437443 * eta_s) * eta2
            - 0.059808322561702126 * dchi2 * eta3
            + 2.3170397514509933
            * dchi
            * delta
            * (1 - 3.2624649875884852 * eta_s)
            * eta3
            * StotR
        )
    )

    Erad = (
        (
            (
                0.057190958417936644 * eta_s
                + 0.5609904135313374 * eta2
                - 0.84667563764404 * eta3
                + 3.145145224278187 * eta4
            )
            * (
                1
                + (
                    -0.13084389181783257
                    - 1.1387311580238488 * eta_s
                    + 5.49074464410971 * eta2
                )
                * StotR
                + (-0.17762802148331427 + 2.176667900182948 * eta2) * StotR2
                + (
                    -0.6320191645391563
                    + 4.952698546796005 * eta_s
                    - 10.023747993978121 * eta2
                )
                * StotR3
            )
        )
        / (
            1
            + (
                -0.9919475346968611
                + 0.367620218664352 * eta_s
                + 4.274567337924067 * eta2
            )
            * StotR
        )
    ) + (
        -0.09803730445895877 * dchi * delta * (1 - 3.2283713377939134 * eta_s) * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653
        * dchi
        * delta
        * (1 - 4.91667749015812 * eta_s)
        * eta_s
        * StotR
    )

    # Taken from https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_t_h_m__fits_8c_source.html

    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a
    a7 = a6 * a

    # First the ringdown frequency
    fRD = (
        (
            0.05947169566573468
            - 0.14989771215394762 * a
            + 0.09535606290986028 * a2
            + 0.02260924869042963 * a3
            - 0.02501704155363241 * a4
            - 0.005852438240997211 * a5
            + 0.0027489038393367993 * a6
            + 0.0005821983163192694 * a7
        )
        / (
            1
            - 2.8570126619966296 * a
            + 2.373335413978394 * a2
            - 0.6036964688511505 * a4
            + 0.0873798215084077 * a6
        )
    ) / (1.0 - Erad)

    # Then the damping frequency
    fdamp = (
        (
            0.014158792290965177
            - 0.036989395871554566 * a
            + 0.026822526296575368 * a2
            + 0.0008490933750566702 * a3
            - 0.004843996907020524 * a4
            - 0.00014745235759327472 * a5
            + 0.0001504546201236794 * a6
        )
        / (
            1
            - 2.5900842798681376 * a
            + 1.8952576220623967 * a2
            - 0.31416610693042507 * a4
            + 0.009002719412204133 * a6
        )
    ) / (1.0 - Erad)

    Z1 = 1.0 + jnp.cbrt((1.0 - a2)) * (jnp.cbrt(1 + a) + jnp.cbrt(1 - a))
    Z1 = jnp.where(Z1 > 3.0, 3.0, Z1)
    Z2 = jnp.sqrt(3.0 * a2 + Z1 * Z1)
    rISCO = 3.0 + Z2 - jnp.sign(a) * jnp.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    rISCOsq = jnp.sqrt(rISCO)
    rISCO3o2 = rISCOsq * rISCOsq * rISCOsq
    OmegaISCO = 1.0 / (rISCO3o2 + a)
    fISCO = OmegaISCO / PI

    fMECO = (
        (
            0.018744340279608845
            + 0.0077903147004616865 * eta_s
            + 0.003940354686136861 * eta2
            - 0.00006693930988501673 * eta3
        )
        / (1.0 - 0.10423384680638834 * eta_s)
        + (
            (
                S
                * (
                    0.00027180386951683135
                    - 0.00002585252361022052 * S
                    + eta4
                    * (
                        -0.0006807631931297156
                        + 0.022386313074011715 * S
                        - 0.0230825153005985 * S2
                    )
                    + eta2
                    * (
                        0.00036556167661117023
                        - 0.000010021140796150737 * S
                        - 0.00038216081981505285 * S2
                    )
                    + eta_s
                    * (
                        0.00024422562796266645
                        - 0.00001049013062611254 * S
                        - 0.00035182990586857726 * S2
                    )
                    + eta3
                    * (
                        -0.0005418851224505745
                        + 0.000030679548774047616 * S
                        + 4.038390455349854e-6 * S2
                    )
                    - 0.00007547517256664526 * S2
                )
            )
            / (
                0.026666543809890402
                + (
                    -0.014590539285641243
                    - 0.012429476486138982 * eta_s
                    + 1.4861197211952053 * eta4
                    + 0.025066696514373803 * eta2
                    + 0.005146809717492324 * eta3
                )
                * S
                + (
                    -0.0058684526275074025
                    - 0.02876774751921441 * eta_s
                    - 2.551566872093786 * eta4
                    - 0.019641378027236502 * eta2
                    - 0.001956646166089053 * eta3
                )
                * S2
                + (
                    0.003507640638496499
                    + 0.014176504653145768 * eta_s
                    + 1.0 * eta4
                    + 0.012622225233586283 * eta2
                    - 0.00767768214056772 * eta3
                )
                * S3
            )
        )
        + (
            dchi2 * (0.00034375176678815234 + 0.000016343732281057392 * eta_s) * eta2
            + dchi
            * delta
            * eta_s
            * (
                0.08064665214195679 * eta2
                + eta_s * (-0.028476219509487793 - 0.005746537021035632 * S)
                - 0.0011713735642446144 * S
            )
        )
    )

    # return fRD / M_s, fdamp / M_s, fMECO / M_s, fISCO / M_s
    # print(fRD, fdamp, fMECO, fISCO)
    return fRD, fdamp, fMECO, fISCO


def nospin_CPvalue(NoSpin_coeffs, eta):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    return (
        NoSpin_coeffs[0]
        + NoSpin_coeffs[1] * eta
        + NoSpin_coeffs[2] * eta2
        + NoSpin_coeffs[3] * eta3
        + NoSpin_coeffs[4] * eta4
    ) / (
        1.0 + NoSpin_coeffs[5] * eta + NoSpin_coeffs[6] * eta2 + NoSpin_coeffs[7] * eta3
    )


def Eqspin_CPvalue(EqSpin_coeffs, eta, S):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    numerator = S * (
        EqSpin_coeffs[0]
        + EqSpin_coeffs[1] * S
        + EqSpin_coeffs[2] * S2
        + EqSpin_coeffs[3] * S3
        + EqSpin_coeffs[4] * S4
        + eta
        * (
            EqSpin_coeffs[5]
            + EqSpin_coeffs[6] * S
            + EqSpin_coeffs[7] * S2
            + EqSpin_coeffs[8] * S3
            + EqSpin_coeffs[9] * S4
        )
        + eta2
        * (
            EqSpin_coeffs[10]
            + EqSpin_coeffs[11] * S
            + EqSpin_coeffs[12] * S2
            + EqSpin_coeffs[13] * S3
            + EqSpin_coeffs[14] * S4
        )
        + eta3
        * (
            EqSpin_coeffs[15]
            + EqSpin_coeffs[16] * S
            + EqSpin_coeffs[17] * S2
            + EqSpin_coeffs[18] * S3
        )
        + eta4
        * (
            EqSpin_coeffs[19]
            + EqSpin_coeffs[20] * S
            + EqSpin_coeffs[21] * S2
            + EqSpin_coeffs[22] * S3
            + EqSpin_coeffs[23] * S4
        )
    )
    denominator = (
        EqSpin_coeffs[24]
        + EqSpin_coeffs[25] * S
        + EqSpin_coeffs[26] * S2
        + EqSpin_coeffs[27] * S3
    )
    return numerator / denominator


def Uneqspin_CPvalue(EqSpin_coeffs, eta, S, dchi):
    dchi2 = dchi * dchi
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    return (
        dchi
        * delta
        * eta
        * (EqSpin_coeffs[0] + EqSpin_coeffs[1] * eta + EqSpin_coeffs[2] * S)
        + EqSpin_coeffs[3] * dchi2 * eta
    )


PhenomX_coeff_table = jnp.array(
    [
        [  # Coeffs collocation point 0 of the inspiral phase
            -17294.000000000007,  # No spin
            -19943.076428555978,
            483033.0998073767,
            0.0,
            0.0,
            4.460294035404433,
            0.0,
            0.0,
            68384.62786426462,  # Eq spin
            67663.42759836042,
            -2179.3505885609297,
            19703.894135534803,
            32614.091002011017,
            -58475.33302037833,
            62190.404951852535,
            18298.307770807573,
            -303141.1945565486,
            0.0,
            -148368.4954044637,
            -758386.5685734496,
            -137991.37032619823,
            1.0765877367729193e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0412979553629143,
            1.0,
            0.0,
            0.0,
            12017.062595934838,  # UnEq Spin
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the inspiral phase
            -7579.300000000004,  # No spin
            -120297.86185566607,
            1.1694356931282217e6,
            -557253.0066989232,
            0.0,
            18.53018618227582,
            0.0,
            0.0,
            -27089.36915061857,  # Eq spin
            -66228.9369155027,
            -44331.41741405198,
            0.0,
            0.0,
            50644.13475990821,
            157036.45676788126,
            126736.43159783827,
            0.0,
            0.0,
            150022.21343386435,
            -50166.382087278434,
            -399712.22891153296,
            0.0,
            0.0,
            -593633.5370110178,
            -325423.99477314285,
            +847483.2999508682,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.5232497464826662,
            -3.062957826830017,
            -1.130185486082531,
            1.0,
            3843.083992827935,  # UnEq Spin
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 2 of the inspiral phase
            15415.000000000007,  # No spin
            873401.6255736464,
            376665.64637025696,
            -3.9719980569125614e6,
            8.913612508054944e6,
            46.83697749859996,
            0.0,
            0.0,
            397951.95299014193,  # Eq spin
            -207180.42746987,
            -130668.37221912303,
            0.0,
            0.0,
            -1.0053073129700898e6,
            1.235279439281927e6,
            -174952.69161683554,
            0.0,
            0.0,
            -1.9826323844247842e6,
            208349.45742548333,
            895372.155565861,
            0.0,
            0.0,
            4.662143741417853e6,
            -584728.050612325,
            -1.6894189124921719e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -9.675704197652225,
            3.5804521763363075,
            2.5298346636273306,
            1.0,
            -24708.109411857182,  # UnEq Spin
            24703.28267342699,
            47752.17032707405,
            -1296.9289110696955,
        ],
        [  # Coeffs collocation point 3 of the inspiral phase
            2439.000000000001,  # No spin
            -31133.52170083207,
            28867.73328134167,
            0.0,
            0.0,
            0.41143032589262585,
            0.0,
            0.0,
            16116.057657391262,  # Eq spin
            9861.635308837876,
            0.0,
            0.0,
            0.0,
            -82355.86732027541,
            -25843.06175439942,
            0.0,
            0.0,
            0.0,
            229284.04542668918,
            117410.37432997991,
            0.0,
            0.0,
            0.0,
            -375818.0132734753,
            -386247.80765802023,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.7385208695213668,
            0.25294420589064653,
            1.0,
            0.0,
            194.5554531509207,  # UnEq Spin
            0.0,
            0.0,
            0.0,
        ],
        [],  # Coeffs collocation point 0 of the intermediate phase
        [],  # Coeffs collocation point 1 of the intermediate phase
        [],  # Coeffs collocation point 2 of the intermediate phase
        [],  # Coeffs collocation point 3 of the intermediate phase
        [],  # Coeffs collocation point 4 of the intermediate phase
        [  # Coeffs collocation point 0 of the merger ringdown phase
            0.0,  # No spin
            0.7207992174994245,
            -1.237332073800276,
            6.086871214811216,
            0.0,
            0.006851189888541745,
            0.06099184229137391,
            -0.15500218299268662,
            1.0,
            0.06519048552628343,  # Eq spin
            0.0,
            0.20035146870472367,
            0.0,
            -0.2697933899920511,
            -25.25397971063995,
            -5.215945111216946,
            -0.28745205203100666,
            5.7756520242745735,
            +4.917070939324979,
            +58.59408241189781,
            +153.95945758807616,
            0.0,
            -43.97332874253772,
            -11.61488280763592,
            +160.14971486043524,
            -693.0504179144295,
            0.0,
            0.0,
            -308.62513664956975,
            +835.1725103648205,
            -47.56042058800358,
            +338.7263666984089,
            -22.384949087140086,
            1.0,
            -0.6628745847248266,
            0.0,
            0.0,
        ],
    ]
)
