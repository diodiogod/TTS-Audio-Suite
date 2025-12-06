#\!/usr/bin/env python3
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.models.factory_config import ModelLoadConfig

    config = ModelLoadConfig(
        engine_name='step_audio_editx',
        model_type='tts',
        model_name='test',
        device='cuda',
        additional_params={'torch_dtype': 'bfloat16', 'quantization': None}
    )

    print('Config created successfully')
    print(config)
    print('Additional params:', config.additional_params)
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
