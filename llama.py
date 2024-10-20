import torch
import math
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from weights import XfmrWeights, LayerWeights
from kvcache import KVCache
from stats import AttnStats
from typing import Dict, List, NamedTuple, Optional, Tuple
from transformers import AutoTokenizer



params = {
  "dim": 2048,
  "n_layers": 16,
  "n_heads": 32,
  "n_kv_heads": 8,
  "vocab_size": 128256,
  "ffn_dim_multiplier": 1.5,
  "multiple_of": 256,
  "norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "use_scaled_rope": True,
  "max_seq_len": 4096
}


class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool


LLAMA_1B_PARAMS = ModelParams(
  n_layers=params["n_layers"],
  n_local_heads=params["n_heads"],
  n_local_kv_heads=params["n_kv_heads"],
  head_dim=params["dim"] // params["n_heads"],
  max_seq_len=params["max_seq_len"],
  rope_theta=params["rope_theta"],
  use_scaled_rope=params["use_scaled_rope"]
)

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print(f"Using Device {device}")


def rms_norm(x: torch.tensor, w: torch.tensor, eps: float = 1e-6) -> torch.tensor:
  return w * (x * torch.rsqrt(torch.pow(x,2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.tensor, xk: torch.tensor, freqs_cis: torch.tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.tensor,torch.tensor]:
  reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
  reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
  xq_out = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1]) * freqs_cis.unsqueeze(0).unsqueeze(2)
  xk_out = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1]) * freqs_cis.unsqueeze(0).unsqueeze(2)
  xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
  xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
  return xq_out.to(dtype), xk_out.to(dtype)

def attention(x: torch.tensor, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: torch.tensor, kvcache: KVCache, attn_mask: Optional[torch.tensor] = None) -> Tuple[torch.tensor, KVCache, torch.tensor]:
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = F.linear(x, layer_weights.wq).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = torch.permute(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
    keys = torch.permute(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = torch.permute(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq.to(torch.bfloat16), keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32
    if cur_pos == 0:
        scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = F.softmax(padded_logits, dim=-1).to(torch.bfloat16)
    output = torch.matmul(scores, values)
    output = output.transpose(1, 2).reshape(xq.shape[0], xq.shape[2], -1)
    out = F.linear(output, layer_weights.wo)
    return out, kvcache, pre_scores

def feed_forward(x: torch.tensor, layer_weights: LayerWeights) -> torch.tensor:
 return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2)

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: torch.tensor, cur_pos: int, freqs_cis: torch.tensor, kvcache: KVCache, attn_mask: Optional[torch.tensor]=None) -> Tuple[torch.tensor, KVCache, torch.tensor, AttnStats]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:,:,-1,:], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats
