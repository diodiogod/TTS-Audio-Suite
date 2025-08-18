from transformers import AutoConfig, AutoModel

from .configuration_higgs_audio import HiggsAudioConfig, HiggsAudioEncoderConfig
from .modeling_higgs_audio import HiggsAudioModel


AutoConfig.register("higgs_audio_encoder", HiggsAudioEncoderConfig)
AutoConfig.register("higgs_audio", HiggsAudioConfig)
AutoModel.register(HiggsAudioConfig, HiggsAudioModel)
