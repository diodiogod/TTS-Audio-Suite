# Copyright (c) 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from zonos3.models.third_party.antspeaker.models.backbones import *
from zonos3.models.third_party.antspeaker.models.poolings import *
from zonos3.models.third_party.antspeaker.utils.registry import register_model, create_backbone, create_pooling

@register_model.register_module("VPR")
class VPR(nn.Module):
    """
        For recognition tasks that need to extract vectors: speaker recognition
    """
    def __init__(self, config):
        super().__init__()
        self.cmn = config.get('cmn', True)
        self.feature_extractor = create_backbone(config["backbone"])
        config["pooling"]["input_dim"] = self.feature_extractor.get_out_dim()
        self.encoding_layers = create_pooling(config["pooling"])
        self.pool_out_dim = self.encoding_layers.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.embed_dim = config.get("embed_dim", 192)
        self.linear = nn.Linear(self.pool_out_dim, self.embed_dim, bias=False)

    def forward(self, inputs):
        if self.cmn:
            with torch.no_grad():
                inputs = inputs - torch.mean(inputs, -1, keepdim=True)
        x = self.feature_extractor(inputs)
        x = self.encoding_layers(x)
        embeddings = self.linear(self.bn(x))
        return embeddings
    
    def load_model(self, checkpoint_path):
        # TTS Audio Suite Patch: weights_only=True
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
        own_state = self.state_dict()
        for name, param in checkpoint["model"].items():
            if name not in own_state:
                print ('{} not found'.format(name))
                continue
            if param.data.shape != own_state[name].shape:
                print ('{} not found different shape'.format(name))
                continue
            # print ('{} loaded'.format(name))
            param = param.data
            own_state[name].copy_(param)
        for name in own_state:
            if name not in checkpoint["model"]:
                print('{} not found in checkpoint'.format(name))    
        return self, checkpoint

