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
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../../")
from zonos3.models.third_party.antspeaker.utils.registry import register_pooling, create_pooling


@register_pooling.register_module("stats")
class Stats_pooling(nn.Module):
    def __init__(self, input_dim=1500, config={}):
        super().__init__()
        self.input_dim = input_dim
        self.mean = config.get("mean", True)
        self.std = config.get("std", True)
        self.combine_type = config.get("combine_type", "cat") # cat or div

        assert self.mean or self.std
        assert self.combine_type == "cat"or self.combine_type == "div"
        if self.mean and self.std and self.combine_type == "cat":
            self.out_dim = 2 * self.input_dim
        else:
            self.out_dim = self.input_dim

    def forward(self, x):
        """
        x.size() = [batch_size, frame, input_dim]
        for x-vector the input_dim is 1500
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        mean_frame = torch.mean(x, -1, keepdim=False)
        std_frame = torch.sqrt(torch.var(x, dim=-1, keepdim=False) + 1e-7)
        if self.std:
            if self.mean:
                if self.combine_type == "cat":
                    output = torch.cat([mean_frame, std_frame], dim=-1)
                else:
                    output = mean_frame / std_frame.clamp(min=1e-10)
            else:
                output = std_frame
        else:
            output = mean_frame
        output = output.view(-1, self.out_dim)
        return output

    def get_out_dim(self):
        return self.out_dim


@register_pooling.register_module("MQMHA")
class MQMHA(nn.Module):
    def __init__(self, input_dim, config={}):
        super(MQMHA, self).__init__()
        self.input_dim = input_dim
        self.num_heads = config.get("num_heads", 1)
        self.num_query = config.get("num_query", 1)
        self.att_bias = config.get("att_bias", False)
        self.mean = config.get("mean", True)
        self.std = config.get("std", True)
        assert self.input_dim % self.num_heads == 0, \
        "head number should be divided by input dimension"
        self.d_k = self.input_dim // self.num_heads
        self.attention = nn.Conv1d(self.input_dim, self.num_query * self.num_heads, 1, stride=1, groups=self.num_heads, bias=self.att_bias)
        if self.std and self.mean:
            self.out_dim = input_dim * self.num_query * 2
        else:
            self.out_dim = input_dim * self.num_query

    def forward(self, x):
        if len(x.shape) == 4:
            # as for resnet, dimension is 4
            x = x.reshape(x.shape[0], -1, x.shape[-1])

        att = self.attention(x) # (b, num_head* num_query, t)
        att = torch.softmax(att, dim=-1) # (b, num_head* num_query, t)

        att = att.reshape(att.shape[0], self.num_heads, self.num_query, att.shape[-1]).unsqueeze(-2)
        x = x.reshape(x.shape[0], self.num_heads, self.d_k, x.shape[-1]).unsqueeze(-3)
        
        mean = torch.sum(att * x, dim=-1)
        mean = mean.flatten(1)
        if self.std:
            var = torch.sum(torch.mul(att, x ** 2), dim=-1).flatten(1) - mean ** 2
            std = torch.sqrt(var.clamp_min(1e-8))
            if self.mean:
                return torch.cat([mean, std], dim=-1)
            else:
                return std
        else:
            return mean
        


@register_pooling.register_module("ASTP")
class ASTP(nn.Module):
    """Attentive statistics pooling: Channel - and context-dependent statistics pooling, first used in ECAPA_TDNN.
    """
    def __init__(self,
                input_dim,
                config={}):
        super(ASTP, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = config.get("bottleneck_dim", 128)
        self.global_context_att = config.get("global_context_att", False)
        # Use Convld with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if self.global_context_att:
            self.linear1 = nn.Conv1d(
                input_dim * 3, self.bottleneck_dim,
                kernel_size=1) # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                input_dim, self.bottleneck_dim,
                kernel_size=1) # equals W and b in the paper
        self.linear2 = nn.Conv1d(self.bottleneck_dim, input_dim,
                                kernel_size=1) # equals V and k in the paper
        self.out_dim = 2 * self.input_dim

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
        or a 4-dimensional tensor in resnet architecture (B,C,F,T)
        0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(self.linear1(x_in)) # alpha= F.relu(self.linear1(x_ in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha* (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.input_dim
        return self.out_dim
    