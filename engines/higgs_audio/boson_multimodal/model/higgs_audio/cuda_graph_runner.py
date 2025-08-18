import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Union
import gc

from transformers.cache_utils import Cache


_NUM_WARMUP_ITERS = 2


class CUDAGraphRunner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Union[Cache, List[torch.FloatTensor]],
        use_cache: bool,
        audio_attention_mask: torch.Tensor,
        fast_forward_attention_mask: torch.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        is_decoding_audio_token: Optional[bool] = None,
        is_using_cuda_graph: Optional[bool] = False,
        stream: torch.cuda.Stream = None,
        memory_pool: Optional[Tuple[int, int]] = None,
    ):
        assert self._graph is None
        # Run warmup iterations
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                audio_discrete_codes_mask=audio_discrete_codes_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=use_cache,
                audio_attention_mask=audio_attention_mask,
                fast_forward_attention_mask=fast_forward_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                is_decoding_audio_token=is_decoding_audio_token,
                is_using_cuda_graph=is_using_cuda_graph,
            )

        torch.cuda.synchronize()

        # Capture the graph
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            out_hidden_states, all_hidden_states, all_self_attns = self.model(
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                audio_discrete_codes_mask=audio_discrete_codes_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=use_cache,
                audio_attention_mask=audio_attention_mask,
                fast_forward_attention_mask=fast_forward_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                is_decoding_audio_token=is_decoding_audio_token,
                is_using_cuda_graph=is_using_cuda_graph,
            )
            # hidden_states_out = torch.ops._C.weak_ref_tensor(outputs[0])
            # del outputs
            gc.collect()
        torch.cuda.synchronize()

        # Save input and output buffers
        self.input_buffers = {
            "hidden_states": hidden_states,
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "audio_discrete_codes_mask": audio_discrete_codes_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "audio_attention_mask": audio_attention_mask,
            "fast_forward_attention_mask": fast_forward_attention_mask,
        }
        self.output_buffers = {
            "hidden_states": out_hidden_states,
            "all_hidden_states": all_hidden_states,
            "all_self_attns": all_self_attns,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        fast_forward_attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Copy input tensors to buffers
        self.input_buffers["hidden_states"].copy_(hidden_states, non_blocking=True)
        self.input_buffers["causal_mask"].copy_(causal_mask, non_blocking=True)
        self.input_buffers["position_ids"].copy_(position_ids, non_blocking=True)
        self.input_buffers["audio_discrete_codes_mask"].copy_(audio_discrete_codes_mask, non_blocking=True)
        self.input_buffers["cache_position"].copy_(cache_position, non_blocking=True)
        self.input_buffers["audio_attention_mask"].copy_(audio_attention_mask, non_blocking=True)
        self.input_buffers["fast_forward_attention_mask"].copy_(fast_forward_attention_mask, non_blocking=True)

        # Run the captured graph
        self.graph.replay()

        return self.output_buffers["hidden_states"], None, None
