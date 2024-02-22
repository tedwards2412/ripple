"""
New, cleaned up version to test BNS waveforms and compare them against IMRPhenomD waveforms.

Both speed and mismatches are checked.

TODO: Implement precession here as well.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import jaxlib
# Choose device here
# jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_platform_name", "cpu")
print(jax.devices())
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from ripple import get_eff_pads, get_match_arr
from ripple import ms_to_Mc_eta, Mc_eta_to_ms, lambdas_to_lambda_tildes
from ripple.constants import PI

import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

########################### 
### Auxiliary functions ###
###########################

def check_is_tidal(IMRphenom: str):
    # Check if the given waveform is supported:
    bns_waveforms = ["IMRPhenomD_NRTidalv2", "TaylorF2"]
    bbh_waveforms = ["IMRPhenomD"]
    
    if IMRphenom in bns_waveforms:
        is_tidal = True
    else:
        is_tidal = False
    
    return is_tidal

def get_jitted_waveform(IMRphenom: str, fs: np.array, f_ref: float):
    # Check if the given waveform is supported:
    
    if IMRphenom == "IMRPhenomD":
        
        # Import the waveform
        from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc as waveform_generator
        
        # Get jitted version (note, use IMRPhenomD as underlying waveform model)
        @jax.jit
        def waveform(theta):
            hp, _ = waveform_generator(fs, theta, f_ref)
            return hp
    
    elif IMRphenom == "IMRPhenomD_NRTidalv2":
        
        # Import the waveform
        from ripple.waveforms.X_NRTidalv2 import gen_NRTidalv2_hphc as waveform_generator
        
        # Get jitted version (note, use IMRPhenomD as underlying waveform model)
        @jax.jit
        def waveform(theta):
            hp, _ = waveform_generator(fs, theta, f_ref, IMRphenom="IMRPhenomD")
            return hp
        
    elif IMRphenom == "TaylorF2":
        
        # Import the waveform
        from ripple.waveforms.TaylorF2 import gen_TaylorF2_hphc as waveform_generator
        
        # Get jitted version
        @jax.jit
        def waveform(theta):
            hp, _ = waveform_generator(fs, theta, f_ref)
            return hp
    
    else:
        raise ValueError("IMRPhenom not recognized ")
    
    return waveform

def get_freqs(f_l, f_u, f_sampling, T):
    # Build the frequency grid
    delta_t = 1 / f_sampling
    tlen = int(round(T / delta_t))
    freqs = np.fft.rfftfreq(tlen, delta_t)
    fs = freqs[(freqs > f_l) & (freqs < f_u)]
    
    return fs


#########################
### Match against LAL ###
#########################

def random_match(n: int, bounds: dict, IMRphenom: str = "IMRPhenomD_NRTidalv2", outdir: str = None, psd_file: str = "psds/psd.txt"):
    """
    Generates random waveform match scores between LAL and ripple.
    
    Note, currently only IMRPhenomD is supported.
    Args:
        n: int
            number of matches to be made
        bounds: dict
            bounds for the parameters TODO make sure format is correct
        IMRphenom: str
            string indicating which waveform to use
        outdir: str, optional
            If not None, will save the matches to a csv file, to this directory

    Returns:
        df: pd.DataFrame, showing the mismatches
    """

    # Specify frequency range
    f_l = 20
    f_sampling = 2 * 2048
    T = 256

    # TODO - check at higher frequency
    f_u = f_sampling // 2
    f_ref = f_l
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = fs[1] - fs[0]
    
    waveform = get_jitted_waveform(IMRphenom, fs, f_ref)
    is_tidal = check_is_tidal(IMRphenom)

    # Get a frequency domain waveform
    thetas = []
    matches = []
    f_ASD, ASD = np.loadtxt(psd_file, unpack=True)
    ASD = np.sqrt(ASD)

    ### Mismatches computations
    for i in tqdm(range(n)):
        non_precessing_matchmaking(
            bounds, IMRphenom, f_l, f_u, df, fs, waveform, f_ASD, ASD, thetas, matches  
        )

    # Save and report mismatches
    thetas = np.array(thetas)
    matches = np.array(matches)
    
    if outdir is not None:
        csv_name = f"{outdir}matches_data/matches_{IMRphenom}.csv"
        print(f"Saving matches to {csv_name}")
        df = save_matches(csv_name, thetas, matches, is_tidal=is_tidal, verbose=True)

    return df


def non_precessing_matchmaking(
    bounds, IMRphenom, f_l, f_u, df, fs, waveform, f_ASD, ASD, thetas, matches, fixed_extrinsic = False, fixed_intrinsic = False,
):
    
    is_tidal = check_is_tidal(IMRphenom)
    
    m1 = np.random.uniform(bounds["m"][0], bounds["m"][1])
    m2 = np.random.uniform(bounds["m"][0], bounds["m"][1])
    s1 = np.random.uniform(bounds["chi"][0], bounds["chi"][1])
    s2 = np.random.uniform(bounds["chi"][0], bounds["chi"][1])
    l1 = np.random.uniform(bounds["lambda"][0], bounds["lambda"][1])
    l2 = np.random.uniform(bounds["lambda"][0], bounds["lambda"][1])

    dist_mpc = np.random.uniform(bounds["d_L"][0], bounds["d_L"][1])
    tc = 0.0
    inclination = np.random.uniform(0, 2*PI)
    phi_ref = np.random.uniform(0, 2*PI)
    
    # TODO remove?
    # if fixed_extrinsic:
    #     dist_mpc = 40.0
    #     inclination = 0.0
    #     phi_ref = 0.0
        
    # if fixed_intrinsic:
    #     l1 = 20.0
    #     l2 = 20.0
        
    #     s1 = 1.0
    #     s2 = 1.0 
        
    # Ensure m1 > m2
    if m1 < m2:
        theta = np.array([m2, m1, s2, s1, l2, l1, dist_mpc, tc, phi_ref, inclination])
    elif m1 >= m2:
        theta = np.array([m1, m2, s1, s2, l1, l2, dist_mpc, tc, phi_ref, inclination])
    else:
        raise ValueError("Something went wrong with the parameters")
    
    # If not tidal, remove l1 and l2 from theta
    if not is_tidal:
        theta = np.delete(theta, [4, 5])
        l1 = 0.0
        l2 = 0.0
    
    # Get approximant for lal
    approximant = lalsim.SimInspiralGetApproximantFromString(IMRphenom)
    
    f_ref = f_l
    m1_kg = theta[0] * lal.MSUN_SI
    m2_kg = theta[1] * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI
    
    if is_tidal:
        laldict = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
        quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
        quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
        # Note that these are dquadmon, not quadmon, hence have to subtract 1 since that is added again later
        lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
        lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)
    else:
        laldict = None

    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        theta[2], # spin m1 zero component
        0.0,
        0.0,
        theta[3], # spin m2 zero component
        distance,
        inclination,
        phi_ref,
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

    freqs_lal = np.arange(len(hp.data.data)) * df
    mask_lal = (freqs_lal > f_l) & (freqs_lal < f_u)
    freqs_lal = freqs_lal[mask_lal]
    hp_lalsuite = hp.data.data[mask_lal]
    
    # Get the ripple waveform
    Mc, eta = ms_to_Mc_eta(jnp.array([theta[0], theta[1]]))
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))

    theta_ripple = jnp.array(
        [Mc, eta, theta[2], theta[3], lambda_tilde, delta_lambda_tilde, dist_mpc, tc, phi_ref, inclination]
    )
    
    # If not tidal, remove lambda parameters
    if not is_tidal:
        theta_ripple = jnp.delete(theta_ripple, jnp.array([4, 5]))
    
    hp_ripple = waveform(theta_ripple)
    # hp_ripple = hp_ripple[mask_lal]
    
    # Check if the ripple strain has NaNs
    if jnp.isnan(hp_ripple).any():
        print("NaNs in ripple strain")
    
    if jnp.isnan(hp_lalsuite).any():
        print("NaNs in lalsuite strain")
        
    # Compute match
    PSD_vals = np.interp(fs, f_ASD, ASD) ** 2
    pad_low, pad_high = get_eff_pads(fs)
    matches.append(
        get_match_arr(
            pad_low,
            pad_high,
            # np.ones_like(fs) * 1.0e-42,
            PSD_vals,
            hp_ripple,
            hp_lalsuite,
        )
    )
    thetas.append(theta)

def save_matches(filename, thetas, matches, verbose=True, is_tidal=False):

    # Get the parameters, which depends on whether or not tidal:
    if is_tidal:
        m1          = thetas[:, 0]
        m2          = thetas[:, 1]
        chi1        = thetas[:, 2]
        chi2        = thetas[:, 3]
        lambda1     = thetas[:, 4]
        lambda2     = thetas[:, 5]
        dist_mpc    = thetas[:, 6]
        tc          = thetas[:, 7]
        phi_ref     = thetas[:, 8]
        inclination = thetas[:, 9]
        
        mismatches = np.log10(1 - matches)
        
        my_dict = {'m1': m1, 
                'm2': m2, 
                'chi1': chi1, 
                'chi2': chi2, 
                'lambda1': lambda1, 
                'lambda2': lambda2,
                'dist_mpc': dist_mpc,
                'tc': tc,
                'phi_ref': phi_ref,
                'inclination': inclination,
                'match': matches, 
                'mismatch': mismatches}
    else:
        m1          = thetas[:, 0]
        m2          = thetas[:, 1]
        chi1        = thetas[:, 2]
        chi2        = thetas[:, 3]
        dist_mpc    = thetas[:, 4]
        tc          = thetas[:, 5]
        phi_ref     = thetas[:, 6]
        inclination = thetas[:, 7]
        
        mismatches = np.log10(1 - matches)

        my_dict = {'m1': m1, 
                'm2': m2, 
                'chi1': chi1, 
                'chi2': chi2, 
                'dist_mpc': dist_mpc,
                'tc': tc,
                'phi_ref': phi_ref,
                'inclination': inclination,
                'match': matches, 
                'mismatch': mismatches}
        
    # Sort the dict and print if desired
    df = pd.DataFrame.from_dict(my_dict)
    df = df.sort_values(by="mismatch", ascending=False)
    df.to_csv(filename)
    
    if verbose:
        print("Mean mismatch:", np.mean(mismatches))
        print("Median mismatch:", np.median(mismatches))
        print("Minimum mismatch:", np.min(mismatches))
        print("Maximum mismatch:", np.max(mismatches))

    return df

##########################
### Speed benchmarking ###
##########################

def benchmark_speed(IMRphenom: str, n: int = 10_000):
    
    # Specify frequency range
    f_l = 20
    f_sampling = 1 * 2048 # 2048 for IMRPhenomD benchmark
    T = 16

    f_u = f_sampling // 2
    f_ref = f_l
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = fs[1] - fs[0]
    
    # @jax.jit
    # def waveform(theta):
    #     return waveform_generator(fs, theta, f_ref)
    
    is_tidal = check_is_tidal(IMRphenom)
    
    # These ranges are taken from: https://wiki.ligo.org/CBC/Waveforms/WaveformTable
    m_l, m_u = 0.5, 3.0
    chi_l, chi_u = -1, 1
    lambda_l, lambda_u = 0, 5000

    m1 = np.random.uniform(m_l, m_u, n)
    m2 = np.random.uniform(m_l, m_u, n)
    s1 = np.random.uniform(chi_l, chi_u, n)
    s2 = np.random.uniform(chi_l, chi_u, n)
    l1 = np.random.uniform(lambda_l, lambda_u, n)
    l2 = np.random.uniform(lambda_l, lambda_u, n)

    dist_mpc = np.random.uniform(0, 1000, n)
    tc = np.zeros_like(dist_mpc)
    inclination = np.random.uniform(0, 2*PI, n)
    phi_ref = np.random.uniform(0, 2*PI, n)
    
    waveform = get_jitted_waveform(IMRphenom, fs, f_ref)
    
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))

    theta_ripple = np.array(
        [Mc, eta, s1, s2, lambda_tilde, delta_lambda_tilde, dist_mpc, tc, phi_ref, inclination]
    ).T
    
    # If not tidal, remove lambda parameters
    if not is_tidal:
        theta_ripple = np.delete(theta_ripple, [4, 5], axis=1)
    
    # Perform the compilation before we time
    print("JIT compiling")
    waveform(theta_ripple[0])[0].block_until_ready()
    print("Finished JIT compiling")
    
    # First, benchmark the jitted version
    print("Benchmarking . . .")
    start = time.time()
    for t in theta_ripple:
        waveform(t)[0].block_until_ready()
    end = time.time()
    print("Ripple waveform call takes: %.6f ms" % ((end - start) * 1000 / n))

    # Second, benchmark the vmapped version
    func = jax.vmap(waveform)
    func(theta_ripple)[0].block_until_ready()
    
    print("Benchmarking . . .")
    start = time.time()
    func(theta_ripple)[0].block_until_ready()
    end = time.time()
    print("Vmapped ripple waveform call takes: %.6f ms" % ((end - start) * 1000 / n))
    
    
def benchmark_speed_lal(IMRphenom, n: int = 10_000):
    
    # Specify frequency range
    f_l = 20
    f_sampling = 1 * 2048 # 2048 for IMRPhenomD benchmark
    T = 16

    f_u = f_sampling // 2
    f_ref = f_l
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = fs[1] - fs[0]
    
    is_tidal = check_is_tidal(IMRphenom)
    
    m_l, m_u = 0.5, 3.0
    chi_l, chi_u = -1, 1
    lambda_l, lambda_u = 0, 5000

    m1 = np.random.uniform(m_l, m_u, n)
    m2 = np.random.uniform(m_l, m_u, n)
    s1 = np.random.uniform(chi_l, chi_u, n)
    s2 = np.random.uniform(chi_l, chi_u, n)
    l1 = np.random.uniform(lambda_l, lambda_u, n)
    l2 = np.random.uniform(lambda_l, lambda_u, n)

    dist_mpc = np.random.uniform(0, 1000, n)
    tc = np.zeros_like(dist_mpc)
    inclination = np.random.uniform(0, 2*PI, n)
    phi_ref = np.random.uniform(0, 2*PI, n)
        
    theta = np.array([m1, m2, s1, s2, l1, l2, dist_mpc, tc, phi_ref, inclination]).T
    # Ensure m1 > m2, but now for all entries in theta
    booleans = theta[:, 0] < theta[:, 1]
    booleans = np.repeat(booleans[:, np.newaxis], 10, axis=1)
    theta = np.where(booleans, theta[:, [1, 0, 3, 2, 5, 4, 6, 7, 8, 9]], theta)
    
    # Get approximant for lal
    approximant = lalsim.SimInspiralGetApproximantFromString(IMRphenom)
    
    f_ref = f_l
    
    # Define the lal waveform generators
    def lal_waveform_tidal(theta):
        
        # Get the tidal parameters and create laldict
        l1 = theta[4]
        l2 = theta[5]
        
        laldict = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
        quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
        quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
        # Note that these are dquadmon, not quadmon, hence have to subtract 1 since that is added again later
        lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
        lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)
        
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        # Get distance parameter
        distance = theta[6] * 1e6 * lal.PC_SI
        # Get inclination and phi_ref
        phi_ref = theta[8]
        inclination = theta[9]

        hp, _ = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            0.0,
            0.0,
            theta[2], # spin m1 zero component
            0.0,
            0.0,
            theta[3], # spin m2 zero component
            distance,
            inclination,
            phi_ref,
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

    def lal_waveform_no_tidal(theta):
        
        # Note: theta still has lambda parameters, but we just select the ones we need here
        
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        # Get distance parameter
        distance = theta[6] * 1e6 * lal.PC_SI
        # Get inclination and phi_ref
        phi_ref = theta[8]
        inclination = theta[9]

        hp, _ = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            0.0,
            0.0,
            theta[2], # spin m1 zero component
            0.0,
            0.0,
            theta[3], # spin m2 zero component
            distance,
            inclination,
            phi_ref,
            0,
            0,
            0,
            df,
            f_l,
            f_u,
            f_ref,
            None,
            approximant,
        )
        
    # Start benchmarking
    if is_tidal:
        print("Benchmarking tidal waveform")
        start = time.time()
        for t in theta:
            lal_waveform_tidal(t)
        end = time.time()
        print("LAL tidal waveform call takes: %.6f ms" % ((end - start) * 1000 / n))
    else:
        print("Benchmarking non-tidal waveform")
        start = time.time()
        for t in theta:
            lal_waveform_no_tidal(t)
        end = time.time()
        print("LAL non-tidal waveform call takes: %.6f ms" % ((end - start) * 1000 / n))

    print("Done")

if __name__ == "__main__":
    
    check_mismatch = True
    check_speed = False
    check_speed_lal = False
    
    approximant = "IMRPhenomD_NRTidalv2" # "TaylorF2", "IMRPhenomD_NRTidalv2" or "IMRPhenomD"
    print(f"Checking approximant {approximant}")
    
    ### Computing and reporting mismatches
    if check_mismatch:
        print("Checking mismatches wrt LAL")
        df = random_match(1000, approximant)
        print("Done. The dataframe is:")
        print(df)
        
    if check_speed:
        print("Checking speed")
        benchmark_speed(approximant)
        print("Done.")
        
    if check_speed_lal:
        print("Checking speed for lalsuite")
        benchmark_speed_lal(approximant)
        print("Done.")
    

    
