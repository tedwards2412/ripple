import jax.numpy as jnp
from ..typing import Array
from ripplegw import Mc_eta_to_ms
from jaxNRSur.SurrogateModel import NRHybSur3dq8Model
from ..constants import C, RSUN_SI, m_per_Mpc
from ripplegw import tukey
import jax


class NRSurrogate:
    def __init__(self, sample_rate: float, segment_len: float, model, tukey_alpha=0.4):
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.model = model
        self.alpha = tukey_alpha

        self.N = int(self.segment_len * self.sample_rate)
        self.time = jnp.arange(self.N) / self.sample_rate - self.segment_len + 2
        self.n = self.N // 2 + 1
        # Calculate the segment length from the duration and the sampling frequency
        segment_length = int(self.sample_rate * self.segment_len)

        # Calculate the frequency resolution
        n_positive = segment_length // 2 + 1
        self.freqs = jnp.linspace(0, self.sample_rate / 2, n_positive)

        # construct window for FFT
        self.window = tukey(self.N, self.alpha)

    def gen_NRSurrogate_td(self, params: Array):
        """Wrapper for jaxNR to produce waveform over given dimensionful time."""
        # get scaling parameters

        Mchirp, eta, chi1, chi2, D, tc, phic, inclination = params

        # evaluate the surrogate over the equivalent geometric time
        m1, m2 = Mc_eta_to_ms(jnp.array([Mchirp, eta]))
        M_tot = m1 + m2
        theta_NRSurrogate = jnp.array([m1 / m2, chi1, chi2])

        time_M = self.time * C / RSUN_SI / M_tot
        h_t = self.model(time_M, theta_NRSurrogate)

        # this is h * r / M, so scale by the mass and distance
        h = h_t * M_tot * RSUN_SI / D / m_per_Mpc
        return h, time_M

    def gen_NRSurrogate_hphc(self, params: Array):
        """
        params array contains both intrinsic and extrinsic variables
        Mchirp: Chirp mass of the system [solar masses]
        eta: Symmetric mass ratio [between 0.0 and 0.25]
        chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
        chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
        D: Luminosity distance to source [Mpc]
        tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
        phic: Phase of coalesence
        inclination: Inclination angle of the binary [between 0 and PI]

        f_ref: Reference frequency for the waveform

        Returns:
        --------
          hp (array): Strain of the plus polarization
          hc (array): Strain of the cross polarization
        """
        # Fourier transform
        h_td, _ = self.gen_NRSurrogate_td(params)
        h_fd = jnp.fft.fft(h_td * self.window)

        # get FD plus and cross polarizations
        h_fd_positive = h_fd[: self.n]
        conj_h_fd_negative = jnp.conj(jnp.fft.ifftshift(h_fd))[: self.n][::-1]

        hp_fd = (h_fd_positive + conj_h_fd_negative) / 2
        hc_fd = 1j * (h_fd_positive - conj_h_fd_negative) / 2

        return hp_fd, hc_fd
