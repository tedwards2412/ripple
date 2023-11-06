import lal
import lalsimulation as lalsim
import numpy as np
import argparse
import sys
import json

def calculate_waveform(m1, m2, chi1, chi2, lambda1, lambda2, distance, T, f_l, f_sampling, inclination):
    # Variables
    m1_msun = m1
    m2_msun = m2
    tc = 0
    phic = 0
    dist_mpc = distance

    polarization_angle = 0.0

    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD_NRTidalv3")

    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, lambda1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, lambda2)
    quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda1)
    quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda2)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)

    # Computations
    f_u = f_sampling // 2
    f_ref = f_l

    delta_t = 1 / f_sampling
    tlen = int(round(T / delta_t))
    freqs = np.fft.rfftfreq(tlen, delta_t)
    df = freqs[1] - freqs[0]
    fs = freqs[(freqs > f_l) & (freqs < f_u)]

    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        chi1,
        0.0,
        0.0,
        chi2,
        distance,
        inclination,
        phic,
        0,
        0,
        0,
        df,
        f_l,
        f_u,
        f_ref,
        laldict,
        approximant,
    )

    hp = hp.data.data
    np.save("hp.npy", hp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate gravitational waveform")
    parser.add_argument("--m1", type=float, default=2, help="Mass of the first object in solar masses")
    parser.add_argument("--m2", type=float, default=2, help="Mass of the second object in solar masses")
    parser.add_argument("--chi1", type=float, default=0.0, help="Spin of the first object")
    parser.add_argument("--chi2", type=float, default=0.0, help="Spin of the second object")
    parser.add_argument("--lambda1", type=float, default=0.0, help="Tidal lambda of the first object")
    parser.add_argument("--lambda2", type=float, default=0.0, help="Tidal lambda of the second object")
    parser.add_argument("--distance", type=float, default=440.0, help="Distance in Mpc")
    parser.add_argument("--T", type=float, default=16, help="Duration of the waveform")
    parser.add_argument("--f_l", type=float, default=20.0, help="Lower frequency bound")
    parser.add_argument("--f_sampling", type=float, default=2 * 2048, help="Sampling frequency")
    parser.add_argument("--inclination", type=float, default=0.0, help="Inclination angle")

    args = parser.parse_args()

    hp = calculate_waveform(args.m1, args.m2, args.chi1, args.chi2, args.lambda1, args.lambda2, args.distance, args.T, args.f_l, args.f_sampling, args.inclination)

    sys.exit(0)
