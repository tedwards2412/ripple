from enum import Enum
from ripplegw import waveforms

class FDWaveform(Enum):
    IMRPhenomD = waveforms.IMRPhenomD
    IMRPhenomPv2 = waveforms.IMRPhenomPv2
    IMRPhenomD_NRTidalv2 = waveforms.IMRPhenomD_NRTidalv2
    IMRPhenomXAS = waveforms.IMRPhenomXAS
    TaylorF2 = waveforms.TaylorF2
    NRSurHyb3dq8_FD = waveforms.JaxNRSur.NRSurHyb3dq8_FD
    NRSur7dq4_FD = waveforms.JaxNRSur.NRSur7dq4_FD

class TDWaveform(Enum):
    pass
