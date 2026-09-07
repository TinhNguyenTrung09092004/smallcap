# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model with a reduced-dimension cross attention."""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    ALL_ATTENTION_FUNCTIONS,
    Conv1D,
    EncoderDecoderCache,
    GPT2Attention,
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Model,
    eager_attention_forward,
)


class ThisGPT2Config(GPT2Config):
    model_type = "this_gpt2"

    def __init__(
        self,
        cross_attention_reduce_factor = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor


class ThisGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        self.cross_split_size = int(self.split_size / self.cross_attention_reduce_factor)
        self.cross_head_dim = int(self.head_dim / self.cross_attention_reduce_factor)

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.cross_split_size, self.embed_dim)
            self.q_attn = Conv1D(self.cross_split_size, self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, self.cross_split_size)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is None:
            return super().forward(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `ThisGPT2Attention(..., is_cross_attention=True)`."
            )

        is_updated = False
        curr_past_key_values = None
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            curr_past_key_values = past_key_values.cross_attention_cache

        query_states = self.q_attn(hidden_states)
        attention_mask = encoder_attention_mask

        if curr_past_key_values is not None and is_updated:
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.cross_split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.cross_head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.cross_head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if curr_past_key_values is not None and not is_updated:
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": None}
            )
            past_key_values.is_updated[self.layer_idx] = True

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            **kwargs,
        )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class ThisGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size

        if config.add_cross_attention:
            self.crossattention = ThisGPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)


class ThisGPT2Model(GPT2Model):
    config_class = ThisGPT2Config
    config: ThisGPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([ThisGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.post_init()


class ThisGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = ThisGPT2Config
    config: ThisGPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ThisGPT2Model(config)
        self.post_init()
