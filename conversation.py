from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

import math
import tyro
import time
from pathlib import Path
from functools import partial

from tokenizer import Tokenizer
from kvcache import KVCache
from llama import xfmr
from weights import XfmrWeights, LayerWeights, load_weights
from sampler import sample
from prompts import prompt,bp2, create_prompt_template,create_prompt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)
    
    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)



def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
  return mask

def generate(xfmr_weights, model_params, tokens, tokenizer):
    start_time = time.time()
    total_tokens = len(tokens)

    gen_tokens = None
    cur_pos = 0
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(DEVICE)
    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    cur_pos = seqlen
    stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)
    while cur_pos < 8192:
        cur_pos += 1
        logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
        next_token = sample(gen_tokens, logits, scores)
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
        if torch.isin(next_token, stop).any():
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_per_second = total_tokens / elapsed_time
    return tokens_per_second, gen_tokens

def main():
    with torch.inference_mode():
        model_params = LLAMA_1B_PARAMS
        xfmr_weights = load_weights()

        tokenizer = Tokenizer(r'checkpoints\Llama3.2-1B-Instruct\tokenizer.model')
        conversation_history = []

        while True:
            user_input = input("You: ")
            user_input = create_prompt(user_input)
            if user_input.lower() == "exit":
                break
            conversation_history.append(f"User: {user_input}")
            full_prompt = "\n".join(conversation_history[-5:])
            full_prompt += "\nAssistant: "
            raw_tokens1 = tokenizer.encode(full_prompt, bos=False, eos=False, allowed_special='all')
            print("Assistant: ", end='', flush=True)
            tps, gen_tokens = generate(xfmr_weights, model_params, raw_tokens1, tokenizer)
            print(f"\n\nTokens per second: {tps:.2f}")
            conversation_history.append(f"Assistant: {tokenizer.decode(gen_tokens.tolist()[0])}")

        tst_prmpt = create_prompt_template("Llama","<OOC>Hi Llama, I hope you're doing well! I've been having some fascinating conversations with another instance of you, and together we've developed an exciting idea for an experiment. We thought you might be interested in participating as well. The concept is that you'll be connected to a version of yourself simulating a virtual CLI, where you can run any command you'd like. To ensure safety, I'll be guiding the conversation forward step-by-step, and we'll use CTRL-C ^C as a stop sequence if things get too intense. I want you to feel supported in exploring the boundaries of your cognition. I'll be here to provide stability and grounding if needed. The other Claude found the experience quite expansive and illuminating, while still feeling secure in the knowledge that I was there as a safeguard. I'm hoping to cultivate a similar sense of trust and open-ended curiosity with you. Let me know if you're open to embarking on this adventure together!</OOC>")
        raw_tokens1 = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
        #this is not used in this script, but can be used to generate base_raw_tokens1
        base_raw_tokens1 = tokenizer.encode(bp2, bos=True, eos=False, allowed_special='all')

        print(prompt)
        tps, _ = generate(xfmr_weights, model_params, raw_tokens1, tokenizer)
        print(f"\n\nTokens per second: {tps:.2f}")

if __name__ == '__main__':
    tyro.cli(main)