from jaxnrsur import JaxNRSur
from jaxnrsur.NRHybSur3dq8 import NRHybSur3dq8Model
from jaxnrsur.NRSur7dq4 import NRSur7dq4Model
from ripplegw.waveforms.WaveformModel import WaveformModel, Polarization
from jaxtyping import Float, Array
import jax.numpy as jnp

class NRSurHyb3dq8_FD(WaveformModel):
    """
    Wrapper class for NRSurHyb3dq8 waveform model in frequency domain.
    """

    def __init__(self, target_frequency: Float[Array, " n_sample"], segment_length: float, sampling_rate: int, alpha_window: float = 0.1):
        self.model_parameters = {}        
        self.surrogate = JaxNRSur(model=NRHybSur3dq8Model(), segment_length = segment_length, sampling_rate = sampling_rate, alpha_window = alpha_window)
        self.frequency_index = jnp.isin(self.surrogate.frequency, target_frequency)

    def full_model(self, sample_points, source_parameters, config_parameters, model_parameters):
        """
        The source parameters should be ordered as follows:
            - M_tot: Total mass in solar masses
            - dist_mpc: Distance in megaparsecs
            - theta: inclination angle of the binary in radians
            - phi: azimuthal angle of the binary in radians
            - q: Mass ratio of the binary (m1/m2) [1.0, 8.0]
            - chi_1z: Dimensionless spin of the primary black hole
            - chi_2z: Dimensionless spin of the secondary black hole
        """
        hp, hc = self.surrogate.get_waveform_fd(source_parameters)
        return {
            Polarization.P: hp[self.frequency_index],
            Polarization.C: hc[self.frequency_index],
        }


class NRSur7dq4_FD(WaveformModel):
    """
    Wrapper class for NRSurHyb3dq8 waveform model in frequency domain.
    """

    def __init__(self, target_frequency: Float[Array, " n_sample"], segment_length: float, sampling_rate: int, alpha_window: float = 0.1):
        self.model_parameters = {}        
        self.surrogate = JaxNRSur(model=NRSur7dq4Model(), segment_length = segment_length, sampling_rate = sampling_rate, alpha_window = alpha_window)
        self.frequency_index = jnp.isin(self.surrogate.frequency, target_frequency)

    def full_model(self, sample_points, source_parameters, config_parameters, model_parameters):
        """
        The source parameters should be ordered as follows:
            - M_tot: Total mass in solar masses
            - dist_mpc: Distance in megaparsecs
            - theta: inclination angle of the binary in radians
            - phi: azimuthal angle of the binary in radians
            - q: Mass ratio of the binary (m1/m2) [1.0, 4.0]
            - chi_1x: Dimensionless spin of the primary black hole
            - chi_1y: Dimensionless spin of the primary black hole
            - chi_1z: Dimensionless spin of the primary black hole
            - chi_2x: Dimensionless spin of the secondary black hole
            - chi_2y: Dimensionless spin of the secondary black hole
            - chi_2z: Dimensionless spin of the secondary black hole
        """
        
        hp, hc = self.surrogate.get_waveform_fd(source_parameters)
        return {
            Polarization.P: hp[self.frequency_index],
            Polarization.C: hc[self.frequency_index],
        }
