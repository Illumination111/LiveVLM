from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import math
import json
from pathlib import Path
import time

from transformers.cache_utils import Cache, DynamicCache


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed

class StreamCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        self.patches = 196
        self.window_len = 32

        self.MAX_GPU_MEMORY = 6000
        self.RECENT_FRMAES = 2 
        self.RECENT_GPU_MEMORY = self.RECENT_FRMAES*self.patches

        self.kvcache_comprssion = True

        self.backup_cache_len = 0 
        
        self._prev_temporal_tokens = 0
        self.temporal_tokens_in_generate = 0

        self.new_coming_tokens = 0
        self.scatter_ratio = 0.2

        # parameters for retrieval 
        self.enable_retrieval = False
        self.chunk_size = 8
        self.observation_len = 32
        self.tmp_key_cache = []
        self.tmp_value_cache = []

    def initialize(
        self, 
        cache_size, 
        recent_frames, 
        scatter_ratio,
        chunk_size=16,
    ):
        self.MAX_GPU_MEMORY = cache_size
        self.RECENT_FRMAES = recent_frames
        self.RECENT_GPU_MEMORY = self.RECENT_FRMAES*self.patches
        self.scatter_ratio = scatter_ratio
        self.chunk_size = chunk_size

        self.retrieval_overhead = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        response = cache_kwargs["response"]
        sparse = cache_kwargs["sparse"]
        if sparse.split('-')[2] != "no":
            retr_ratio = float(sparse.split('-')[2])
            if retr_ratio < 1.0:
                self.enable_retrieval = True
        else: 
            self.enable_retrieval = False

        self.new_coming_tokens = key_states.shape[2]
        if self.enable_retrieval and response:
            key_states_all, value_states_all = self.query_retrieval(key_states, value_states, layer_idx, cache_kwargs)
        else:
            key_states_all, value_states_all = super().update(key_states, value_states, layer_idx, cache_kwargs)

        return key_states_all, value_states_all
    
    def query_retrieval(
        self,
        new_key_states,
        new_value_states,
        layer_idx,
        cache_kwargs,
    ):
        if len(self.tmp_key_cache) <= layer_idx:
            start_time = time.time()
            query_states = cache_kwargs["query_states"]
            observation_len = self.observation_len
            query_states = query_states[:,:,:observation_len,:].mean(dim=-2, keepdim=True)
            prefix_len = cache_kwargs["prefix_len"]

            sparse = cache_kwargs["sparse"]
            retr_ratio = float(sparse.split('-')[2])
            key_states = self.key_cache[layer_idx]
            value_states = self.value_cache[layer_idx]
            bsz, num_kv_heads, kv_len, head_dim = key_states.shape
            num_heads = query_states.shape[1]
            cache_len = kv_len - prefix_len - self.RECENT_GPU_MEMORY
            remainder = cache_len % self.chunk_size

            video_end_idx = kv_len-(self.RECENT_GPU_MEMORY+remainder)
            if video_end_idx <= prefix_len:
                self.tmp_key_cache.append(key_states)
                self.tmp_value_cache.append(value_states)
                self.tmp_key_cache[layer_idx] = torch.cat([self.tmp_key_cache[layer_idx], new_key_states], dim=-2)
                self.tmp_value_cache[layer_idx] = torch.cat([self.tmp_value_cache[layer_idx], new_value_states], dim=-2)
                return self.tmp_key_cache[layer_idx], self.tmp_value_cache[layer_idx]
            
            video_keys = key_states[:,:,prefix_len:kv_len-(self.RECENT_GPU_MEMORY+remainder),:]
            video_keys = video_keys.reshape(bsz, num_kv_heads, -1, self.chunk_size, head_dim)
            video_values = value_states[:,:,prefix_len:kv_len-(self.RECENT_GPU_MEMORY+remainder),:]
            video_values = video_values.reshape(bsz, num_kv_heads, -1, self.chunk_size, head_dim)
            mean_video_keys = video_keys.mean(dim=-2, keepdim=True)
            
            mean_video_keys = mean_video_keys.squeeze(-2)
            mean_video_keys = repeat_kv(mean_video_keys, num_heads // num_kv_heads)
            attn_weights = torch.matmul(query_states, mean_video_keys.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            ).detach() # [bsz, num_heads, q_len, q_len(k)]
            attn_weights = attn_weights.mean(dim=-2)
            attn_weights = attn_weights.reshape(bsz, num_kv_heads, num_heads//num_kv_heads, -1).mean(dim=-2) 
            retr_chunk_num = int(attn_weights.shape[2] * retr_ratio)
            top_indices = attn_weights.sort(dim=-1, descending=True)[1]
            top_indices = top_indices[:,:,:retr_chunk_num]
            top_indices = top_indices.sort(dim=-1)[0]
            top_indices = top_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.chunk_size, head_dim)
            video_keys = video_keys.gather(index=top_indices, dim=2).reshape(bsz, num_kv_heads, -1, head_dim)
            video_values = video_values.gather(index=top_indices, dim=2).reshape(bsz, num_kv_heads, -1, head_dim)
            key_states = torch.cat([
                key_states[:,:,:prefix_len,:],
                video_keys,
                key_states[:,:,kv_len-(self.RECENT_GPU_MEMORY+remainder):,:]
            ], dim=-2)
            value_states = torch.cat([
                value_states[:,:,:prefix_len,:],
                video_values,
                value_states[:,:,kv_len-(self.RECENT_GPU_MEMORY+remainder):,:]
            ], dim=-2)
            self.tmp_key_cache.append(key_states)
            self.tmp_value_cache.append(value_states)
            end_time = time.time()
            self.retrieval_overhead.append(end_time - start_time)

        self.tmp_key_cache[layer_idx] = torch.cat([self.tmp_key_cache[layer_idx], new_key_states], dim=-2)
        self.tmp_value_cache[layer_idx] = torch.cat([self.tmp_value_cache[layer_idx], new_value_states], dim=-2)
        return self.tmp_key_cache[layer_idx], self.tmp_value_cache[layer_idx]
  
    def cache_compression_diversity(
        self,
        key_states_all: torch.Tensor,
        value_states_all: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        query_states = cache_kwargs['query_states']
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_key_value_heads, k_len = key_states_all.shape[1:3]
        assert bsz == 1

        prefix_len = cache_kwargs['prefix_len']

        key_states_all_repeated = repeat_kv(key_states_all, num_heads // num_key_value_heads)[:,:,:k_len-self.RECENT_GPU_MEMORY,:]
        window_len = self.window_len
        query_states = query_states[:,:,-window_len:,:]
        
        attn_weights = torch.matmul(query_states, key_states_all_repeated.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        ).detach() # [bsz, num_heads, q_len, q_len(k)]
        attn_weights = attn_weights.mean(dim=-2)[:,:,prefix_len:]

        attn_weights = attn_weights.reshape(bsz, num_key_value_heads, num_heads//num_key_value_heads, -1)
        keep_cache_len = self.MAX_GPU_MEMORY - self.RECENT_GPU_MEMORY - prefix_len
        # diversity
        scatter_ratio = self.scatter_ratio
        attn_weights = attn_weights.mean(dim=-2)
        top_indices = attn_weights.sort(dim=-1, descending=True)[1]
        if scatter_ratio > 0:
            top_indices = top_indices.squeeze(0)
            topk_len = int(keep_cache_len*(1-scatter_ratio))
            scatter_len = keep_cache_len - topk_len

            def uniform_pick_2d(x: torch.Tensor, N: int, C: int):
                B, L = x.shape
                K = N + C
                w = (L + K - 1) // K          

                bucket = x // w              

                mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
                mask.scatter_(1, x[:, :N], True)
                full = torch.zeros(B, K, dtype=torch.bool, device=x.device)
                full.scatter_(1, bucket[:, :N], True)

                remain_idx = x[:, N:]          
                remain_bkt = bucket[:, N:]     
                can_fill = ~full.gather(1, remain_bkt) 
                cum = can_fill.cumsum(dim=1)
                pick = (cum <= C) & can_fill  
                row, col = pick.nonzero(as_tuple=False).T
                orig_col = remain_idx[row, col]
                mask[row, orig_col] = True

                selected_num = mask.sum(dim=1)    
                need = K - selected_num
                if need.gt(0).any():
                    left = (~mask).float()
                    score = left * (L + 1) - x.float()
                    _, order = score.sort(dim=1, descending=True)
                    for b in range(B):
                        k = need[b].item()
                        if k <= 0:
                            continue
                        fill_col = order[b, :k]
                        mask[b, fill_col] = True

                return mask
            
            mask = uniform_pick_2d(top_indices, topk_len, scatter_len)
          
            top_indices = torch.arange(top_indices.shape[1], device=top_indices.device).unsqueeze(0).repeat(top_indices.shape[0], 1)[mask].reshape(top_indices.shape[0], -1)
            top_indices = top_indices.unsqueeze(0)
        else:
            top_indices = top_indices[:,:,:keep_cache_len]
            top_indices = top_indices.sort(dim=-1)[0]
        
        top_indices = top_indices.unsqueeze(-1).repeat(1, 1, 1, head_dim)
        
        long_term_key_states = key_states_all[:,:,prefix_len:k_len-self.RECENT_GPU_MEMORY,:]
        long_term_value_states = value_states_all[:,:,prefix_len:k_len-self.RECENT_GPU_MEMORY,:]
        long_term_key_states = long_term_key_states.gather(dim=2, index=top_indices)
        long_term_value_states = long_term_value_states.gather(dim=2, index=top_indices)

        key_states_all = torch.cat([key_states_all[:,:,:prefix_len,:], long_term_key_states, key_states_all[:,:,k_len-self.RECENT_GPU_MEMORY:,:]], dim=-2)
        value_states_all = torch.cat([value_states_all[:,:,:prefix_len,:], long_term_value_states, value_states_all[:,:,k_len-self.RECENT_GPU_MEMORY:,:]], dim=-2)
      
        del key_states_all_repeated
        del attn_weights
        del long_term_key_states
        del long_term_value_states

        return key_states_all, value_states_all

 
    def backup_cache(self):
        if len(self.key_cache)>0:
            self.backup_cache_len = self.key_cache[0].shape[2]
    
    def reset_cache(self):
        if not self.enable_retrieval:
            if self.backup_cache_len > 0:
                for layer_idx in range(len(self.key_cache)):
                    self.key_cache[layer_idx] = self.key_cache[layer_idx][:,:,:self.backup_cache_len,:]
                    self.value_cache[layer_idx] = self.value_cache[layer_idx][:,:,:self.backup_cache_len,:]
                self.temporal_tokens_in_generate = self._prev_temporal_tokens
            else:
                self.key_cache.clear()
                self.value_cache.clear()
        else:
            self.tmp_key_cache.clear()
            self.tmp_value_cache.clear()
            self.temporal_tokens_in_generate = self._prev_temporal_tokens

    def flush_cache(self):
        self._prev_temporal_tokens = self.temporal_tokens_in_generate
        gc.collect()
        torch.cuda.empty_cache()

    def compression_after_attn(self, layer_idx, cache_kwargs):
        response = cache_kwargs["response"]
        if self.kvcache_comprssion and not response and self.key_cache[layer_idx].shape[2]>=self.MAX_GPU_MEMORY:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = self.cache_compression_diversity(self.key_cache[layer_idx], self.value_cache[layer_idx], layer_idx, cache_kwargs)
            
  
    def update_pos_ids(
        self,
        is_last_layer,
        q_len,
    ):
        if is_last_layer:
           self.temporal_tokens_in_generate += q_len
    


