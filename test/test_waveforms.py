import numpy as np
import jax.numpy as jnp
import pytest

from ripplegw.waveforms.JaxNRSur import NRSurHyb3dq8_FD, NRSur7dq4_FD
from ripplegw.waveforms.WaveformModel import Polarization
from ripplegw.waveforms.IMRPhenomD import IMRPhenomD, gen_IMRPhenomD_hphc
from ripplegw.waveforms.IMRPhenomPv2 import IMRPhenomPv2, gen_IMRPhenomPv2_hphc
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import IMRPhenomD_NRTidalv2, gen_IMRPhenomD_NRTidalv2_hphc
from ripplegw.waveforms.IMRPhenomXAS import IMRPhenomXAS, gen_IMRPhenomXAS_hphc
from ripplegw.waveforms.TaylorF2 import TaylorF2, gen_TaylorF2_hphc

def test_imrphenom_d():
    frequency = jnp.linspace(20, 2048, 1000)
    f_ref = 20.0
    params = jnp.array([30.0, 0.249, 0.5, 0.3, 500.0, 0.0, 0.0, 0.0])  # M_c, eta, s1_z, s2_z, d_L, phase_c, iota
    hp, hc = gen_IMRPhenomD_hphc(frequency, params, f_ref=f_ref)
    result = IMRPhenomD()(frequency, params, {'f_ref': f_ref})
    assert (hp == result[Polarization.P]).all()
    assert (hc == result[Polarization.C]).all()

def test_imrphenom_pv2():
    frequency = jnp.linspace(20, 2048, 1000)
    f_ref = 20.0
    # [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phiRef, incl]
    params = jnp.array([30.0, 0.249, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 500.0, 0.0, 0.0, 0.5])
    hp, hc = gen_IMRPhenomPv2_hphc(frequency, params, f_ref=f_ref)
    result = IMRPhenomPv2()(frequency, params, {'f_ref': f_ref})
    assert (hp == result[Polarization.P]).all()
    assert (hc == result[Polarization.C]).all()

def test_imrphenomd_nrtidalv2():
    frequency = jnp.linspace(20, 2048, 1000)
    f_ref = 20.0
    params = jnp.array([30.0, 0.249, 0.3, -0.2, 500.0, 0.0, 0.5, 0.0])
    hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(frequency, params, f_ref=f_ref)
    result = IMRPhenomD_NRTidalv2()(frequency, params, {'f_ref': f_ref})
    assert (hp == result[Polarization.P]).all()
    assert (hc == result[Polarization.C]).all()

def test_imrphenomxas():
    frequency = jnp.linspace(20, 2048, 1000)
    f_ref = 20.0
    # [M_c, eta, s1z, s2z, d_L, phase_c, iota, tc]
    params = jnp.array([30.0, 0.249, 0.3, -0.2, 500.0, 0.0, 0.5, 0.0])
    hp, hc = gen_IMRPhenomXAS_hphc(frequency, params, f_ref=f_ref)
    result = IMRPhenomXAS()(frequency, params, {'f_ref': f_ref})
    assert (hp == result[Polarization.P]).all()
    assert (hc == result[Polarization.C]).all()

def test_taylorf2():
    frequency = jnp.linspace(20, 2048, 1000)
    f_ref = 20.0
    params = jnp.array([30.0, 0.249, 0.3, -0.2, 500.0, 0.0, 0.5, 0.0])
    hp, hc = gen_TaylorF2_hphc(frequency, params, f_ref=f_ref, use_lambda_tildes=True)
    result = TaylorF2()(frequency, params, {'f_ref': f_ref, 'use_lambda_tildes': True})
    print(hp )
    assert (hp == result[Polarization.P]).all()
    assert (hc == result[Polarization.C]).all()



@pytest.mark.parametrize("segment_length,sampling_rate", [
    (4.0, 2048),
    (8.0, 4096),
])
def test_nrsurhyb3dq8_fd_basic(segment_length, sampling_rate):
    freqs = jnp.arange(20, 1024, 1.0 / (segment_length))
    model = NRSurHyb3dq8_FD(freqs, segment_length=segment_length, sampling_rate=sampling_rate)

    # M_tot, dist_mpc, q, chi_1z, chi_2z
    source_parameters = jnp.array([60.0, 500.0, 2.0, 0.5, -0.3])
    config_parameters = {}
    model_parameters = {}

    result = model.full_model(freqs, source_parameters, config_parameters, model_parameters)
    assert isinstance(result, dict)
    assert Polarization.P in result and Polarization.C in result
    assert result[Polarization.P].shape == freqs.shape
    assert result[Polarization.C].shape == freqs.shape

@pytest.mark.parametrize("segment_length,sampling_rate", [
    (4.0, 2048),
    (8.0, 4096),
])
def test_nrsur7dq4_fd_basic(segment_length, sampling_rate):
    freqs = jnp.arange(20, 1024, 1.0 / (segment_length))
    model = NRSur7dq4_FD(freqs, segment_length=segment_length, sampling_rate=sampling_rate)
    # M_tot, dist_mpc, q, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z
    source_parameters = jnp.array([60.0, 500.0, 1.5, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    config_parameters = {}
    model_parameters = {}

    result = model.full_model(freqs, source_parameters, config_parameters, model_parameters)
    assert isinstance(result, dict)
    assert Polarization.P in result and Polarization.C in result
    assert result[Polarization.P].shape == freqs.shape
    assert result[Polarization.C].shape == freqs.shape
