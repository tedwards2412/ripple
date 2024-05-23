from ripplegw.waveforms import (
    IMRPhenomXAS,
    IMRPhenomD,
    IMRPhenomD_NRTidalv2,
    IMRPhenomPv2,
)


class FD_waveform_generator:
    def __init__(self):
        # Initialize the dictionary of generator functions
        self.generators = {
            "XAS": IMRPhenomXAS.gen_IMRPhenomXAS_hphc,
            "D": IMRPhenomD.gen_IMRPhenomD_hphc,
            "D_NRTidalv2": IMRPhenomD_NRTidalv2.gen_IMRPhenomD_NRTidalv2_hphc,
            "Pv2": IMRPhenomPv2.gen_IMRPhenomPv2,
        }

    def generate_waveform(self, model_type, fs, theta_ripple, f_ref):
        if model_type in self.generators:
            gen_func = self.generators[model_type]
            return gen_func(fs, theta_ripple, f_ref)
        else:
            raise ValueError(f"No generator found for model type '{model_type}'")
