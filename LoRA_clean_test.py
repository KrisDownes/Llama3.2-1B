import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict,NamedTuple
import math
import json
import logging
import random
from tqdm import tqdm
import gc
from llama import attention, apply_rotary_emb, rms_norm, feed_forward,LLAMA_1B_PARAMS
from kvcache import KVCache
from tokenizer import Tokenizer
from kvcache import KVCache


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print(f"Using Device {device}")
@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16
    dropout: float = 0.05
    target_modules: tuple = ("wq", "wk", "wv", "wo")

@dataclass
class LayerWeights(NamedTuple):
    wq: torch.tensor
    wk: torch.tensor
    wv: torch.tensor
    wo: torch.tensor
    w1: torch.tensor
    w2: torch.tensor
    w3: torch.tensor
    ffn_norm: torch.tensor
    attention_norm: torch.tensor

class XfmrWeights(NamedTuple):
    tok_embeddings: torch.tensor
    norm: torch.tensor
    output: torch.tensor
    layer_weights: List[LayerWeights]

def load_weights(ckpt_path: Path = Path('checkpoints\Llama3.2-1B-Instruct\consolidated.00.pth'), n_layers: int = 16):
        # Load the entire state dict
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)        
        layer_weights = []
        for i in range(n_layers):
            layer_weights.append(LayerWeights(
                wq=state_dict[f'layers.{i}.attention.wq.weight'].to(torch.bfloat16),
                wk=state_dict[f'layers.{i}.attention.wk.weight'].to(torch.bfloat16),
                wv=state_dict[f'layers.{i}.attention.wv.weight'].to(torch.bfloat16),
                wo=state_dict[f'layers.{i}.attention.wo.weight'].to(torch.bfloat16),
                w1=state_dict[f'layers.{i}.feed_forward.w1.weight'].to(torch.bfloat16),
                w2=state_dict[f'layers.{i}.feed_forward.w2.weight'].to(torch.bfloat16),
                w3=state_dict[f'layers.{i}.feed_forward.w3.weight'].to(torch.bfloat16),
                ffn_norm=state_dict[f'layers.{i}.ffn_norm.weight'].to(torch.bfloat16),
                attention_norm=state_dict[f'layers.{i}.attention_norm.weight'].to(torch.bfloat16),
            ))
        
        xfmr_weights = XfmrWeights(
            tok_embeddings=state_dict['tok_embeddings.weight'].to(torch.bfloat16),
            norm=state_dict['norm.weight'].to(torch.bfloat16),
            output=state_dict['output.weight'].to(torch.bfloat16),
            layer_weights=layer_weights
        )
        
        return xfmr_weights


def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    """
    Applies scaled frequency scaling for RoPE embeddings.
    """
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

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32, device: str = "cuda") -> torch.Tensor:
    """
    Precomputes frequency cis for rotary position embeddings.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int, device: str = "cuda") -> torch.Tensor:
    """
    Builds causal attention mask for transformer.
    """
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
    return mask

class LoRALinear(nn.Module):
    def __init__(self, base_weight: torch.Tensor, config: LoRAConfig):
        super().__init__()
        self.base_weight = nn.Parameter(base_weight, requires_grad=False)  # Freeze base weights
        self.rank = config.r
        self.scaling = config.alpha / config.r
        
        # Initialize LoRA matrices
        in_features = base_weight.shape[1]
        out_features = base_weight.shape[0]
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank), requires_grad=True)
        self.dropout = nn.Dropout(p=config.dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base_weight)
        if self.rank > 0:
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            return base_out + (lora_out * self.scaling)
        return base_out
    def get_effective_weight(self) -> torch.Tensor:
        """Compute the effective weight matrix including LoRA adaptation"""
        if self.rank > 0:
            return self.base_weight + (self.lora_B @ self.lora_A * self.scaling)
        return self.base_weight
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Make LoRALinear compatible with torch functions"""
        if kwargs is None:
            kwargs = {}
            
        if func is F.linear:
            # For F.linear, replace self with effective weight
            new_args = list(args)
            idx = new_args.index(self)
            new_args[idx] = self.get_effective_weight()
            return func(*new_args, **kwargs)
            
        return NotImplemented
    
    @property
    def shape(self):
        """Return shape of effective weight matrix"""
        return self.base_weight.shape
    
    def to(self, *args, **kwargs):
        """Ensure proper device movement"""
        super().to(*args, **kwargs)
        self.base_weight = self.base_weight.to(*args, **kwargs)
        return self


class LoRALayerWeights(nn.Module):
    def __init__(self, base_layer, config: LoRAConfig):
        super().__init__()
        self.attention_norm = nn.Parameter(base_layer.attention_norm, requires_grad=False)
        self.ffn_norm = nn.Parameter(base_layer.ffn_norm, requires_grad=False)
        
        self.wq = LoRALinear(base_layer.wq, config) if "wq" in config.target_modules else base_layer.wq
        self.wk = LoRALinear(base_layer.wk, config) if "wk" in config.target_modules else base_layer.wk
        self.wv = LoRALinear(base_layer.wv, config) if "wv" in config.target_modules else base_layer.wv
        self.wo = LoRALinear(base_layer.wo, config) if "wo" in config.target_modules else base_layer.wo
        
        # FFN weights - detach to prevent gradient updates
        self.w1 = nn.Parameter(base_layer.w1, requires_grad=False)
        self.w2 = nn.Parameter(base_layer.w2, requires_grad=False)
        self.w3 = nn.Parameter(base_layer.w3, requires_grad=False)

    def requires_grad_(self, requires_grad: bool = True):
        """Override requires_grad to only affect LoRA parameters"""
        # Only set requires_grad for LoRA parameters
        for name, param in self.named_parameters():
            if "lora_" in name:  # Only LoRA parameters should have gradients
                param.requires_grad = requires_grad
            else:
                param.requires_grad = False
        return self
    @property
    def device(self):
        """Helper to get device of the module"""
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        """Override to method to properly handle device transfers"""
        device = kwargs.get('device', args[0] if args else None)
        if device:
            # Handle non-parameter tensors
            if isinstance(self.w1, torch.Tensor):
                self.w1 = self.w1.to(device)
            if isinstance(self.w2, torch.Tensor):
                self.w2 = self.w2.to(device)
            if isinstance(self.w3, torch.Tensor):
                self.w3 = self.w3.to(device)
        return super().to(*args, **kwargs)

class LoRAModel(nn.Module):
    def __init__(self, base_weights, config: Optional[LoRAConfig] = None):
        super().__init__()
        if config is None:
            config = LoRAConfig()
            
        self.tok_embeddings = base_weights.tok_embeddings
        self.norm = base_weights.norm
        self.output = base_weights.output
        
        self.layers = nn.ModuleList([
            LoRALayerWeights(layer, config) 
            for layer in base_weights.layer_weights
        ])
        
        self.attention_fn = attention
        self.feed_forward_fn = feed_forward
        self.rms_norm_fn = rms_norm

        self._gradient_checkpointing = False
        self.n_layers = len(self.layers)

    def _initialize_kvcache(self, batch_size: int, max_seq_len: int, model_params) -> KVCache:
        """Initialize KVCache with proper dimensions"""
        return KVCache.new(
            layers=self.n_layers,
            bsz=batch_size,
            max_seq_len=max_seq_len,
            kv_heads=model_params.n_local_kv_heads,
            head_dim=model_params.head_dim
        ) 
    
    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for memory efficiency"""
        self._gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing"""
        self._gradient_checkpointing = False

    def _forward_with_checkpointing(self, h, layer, model_params, cur_pos, 
                                  layer_idx, freqs_cis, kvcache, attn_mask):
        """Forward pass for a single layer with gradient checkpointing"""
        def create_custom_forward(module):
            def custom_forward(*args):
                print("In Custom forward.")
                norm_x = self.rms_norm_fn(args[0], module.attention_norm)
                print(module)
                if kvcache is None:
                    batch_size = args[0].size(0)
                    seq_len = args[0].size(1)
                    current_kvcache = self._initialize_kvcache(batch_size, seq_len, model_params)
                else:
                    current_kvcache = kvcache
                h_attn, kv, scores = self.attention_fn(
                    norm_x, module, model_params, cur_pos, layer_idx, 
                    freqs_cis, current_kvcache, attn_mask
                )
                out = args[0] + h_attn
                out = out + self.feed_forward_fn(
                    self.rms_norm_fn(out, module.ffn_norm), module
                )
                print("End of custom forward")
                print(f"out grad: {out.requires_grad} scores grad: {scores.requires_grad}")
                return out, kv, scores
            return custom_forward

        if self.training and self._gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer),
                h,
                preserve_rng_state=True
            )
        else:
            norm_x = self.rms_norm_fn(h, layer.attention_norm)
            if kvcache is None:
                batch_size = h.size(0)
                seq_len = h.size(1)
                kvcache = self._initialize_kvcache(batch_size, seq_len, model_params)
            h_attn, kv, scores = self.attention_fn(
                norm_x, layer, model_params, cur_pos, layer_idx, 
                freqs_cis, kvcache, attn_mask
            )
            out = h + h_attn
            out = out + self.feed_forward_fn(
                self.rms_norm_fn(out, layer.ffn_norm), layer
            )
            return out, kv, scores

    def forward(self, tokens: torch.Tensor, model_params, cur_pos: int, 
                freqs_cis: Optional[torch.Tensor] = None, kvcache=None, attn_mask=None):
        # Compute frequencies if not provided
        if freqs_cis is None:
            freqs_cis = precompute_freqs_cis(
                model_params.head_dim,
                tokens.shape[1],
                model_params.rope_theta,
                model_params.use_scaled_rope,
                device=tokens.device
            )
        
        # Build attention mask if not provided
        if attn_mask is None:
            attn_mask = build_attn_mask(tokens.shape[1], cur_pos, device=tokens.device)
            
        h = self.tok_embeddings[tokens]

        if kvcache is None:
            batch_size = tokens.size(0)
            seq_len = tokens.size(1)
            kvcache = self._initialize_kvcache(batch_size, seq_len, model_params)

        new_kvcache = []
        all_scores = []
        
        for i, layer in enumerate(self.layers):
            h, kv, scores = self._forward_with_checkpointing(
                h, layer, model_params, cur_pos, i, freqs_cis, 
                kvcache if kvcache else None, attn_mask
            )
            new_kvcache.append(kv)
            all_scores.append(scores)
        
        h = self.rms_norm_fn(h, self.norm)
        output = F.linear(h, self.output)
        return output, new_kvcache, all_scores, {}

class CodeDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 2048,
        prompt_template: Optional[Dict[str, str]] = None,
        add_code_markers: bool = True,
        num_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        self.prompt_template = prompt_template or {
            "prefix": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n",
            "instruction": "### Instruction:\n{instruction}\n\n",
            "input": "### Input:\n{input}\n\n" if "{input}" in "{input}" else "",
            "response": "### Response:\n{response}"
        }
        
        logging.info(f"Loading dataset from {data_path}")
        
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f) if data_path.endswith('.json') else [
                json.loads(line) for line in f
            ]
            
        if num_samples and num_samples < len(data):
            data = random.sample(data, num_samples)
        
        for example in data:
            instruction = example.get('instruction', example.get('prompt', ''))
            input_text = example.get('input', '')
            output = example.get('output', example.get('completion', ''))
            
            if add_code_markers and self._is_code_content(output):
                output = f"```python\n{output}\n```"
            
            prompt = self._format_prompt(instruction, input_text)
            prompt_tokens = self.tokenize(prompt, add_eos=False)
            output_tokens = self.tokenize(output, add_eos=True)
            
            total_length = len(prompt_tokens) + len(output_tokens)
            if total_length > max_length:
                logging.warning(f"Skipping example with length {total_length} > {max_length}")
                continue
                
            self.examples.append({
                "prompt": prompt_tokens,
                "completion": output_tokens
            })
        
        logging.info(f"Loaded {len(self.examples)} examples")
    
    def _is_code_content(self, text: str) -> bool:
        code_indicators = ['def ', 'class ', 'import ', 'return ', '    ']
        return any(indicator in text for indicator in code_indicators)
    
    def _format_prompt(self, instruction: str, input_text: str = '') -> str:
        prompt = self.prompt_template['prefix']
        prompt += self.prompt_template['instruction'].format(instruction=instruction)
        if input_text:
            prompt += self.prompt_template['input'].format(input=input_text)
        return prompt
    
    def tokenize(self, text: str, add_eos: bool = False) -> List[int]:
        return self.tokenizer.encode(text, bos=False, eos=add_eos, allowed_special='all')

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        tokens = example['prompt'] + example['completion']
        attention_mask = [1] * len(example['prompt']) + [0] * len(example['completion'])
        
        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
        return {
            'input_ids': torch.tensor(tokens),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(tokens)
        }
def create_code_dataloaders(
    train_path: str,
    val_path: Optional[str],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    num_workers: int = 1,
    val_split: float = 0.1,
    prompt_template: Optional[Dict[str, str]] = None
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders for code data.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        val_split: Validation split ratio if val_path not provided
        prompt_template: Custom prompt template
    
    Returns:
        Dictionary containing 'train' and 'val' dataloaders
    """
    train_dataset = CodeDataset(
        train_path,
        tokenizer,
        max_length=max_length,
        prompt_template=prompt_template
    )
    
    if val_path:
        val_dataset = CodeDataset(
            val_path,
            tokenizer,
            max_length=max_length,
            prompt_template=prompt_template
        )
    else:
        # Split training data for validation
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }

class MemoryEfficientTrainer:
    def __init__(
        self,
        model: LoRAModel,
        tokenizer,
        train_path: str,
        val_path: Optional[str] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        grad_accum_steps: int = 8,
        max_length: int = 2048,
        val_split: float = 0.1,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        
        # Create dataloaders
        self.dataloaders = create_code_dataloaders(
            train_path=train_path,
            val_path=val_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=1,  # Lower for memory efficiency
            val_split=val_split
        )
        
        # Initialize optimizer with gradient accumulation
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
    
    def train_epoch(self, model_params, epoch: int):
        self.model.train()
        total_loss = 0
        num_batches = len(self.dataloaders['train'])
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}") as pbar:
            for i, batch in enumerate(self.dataloaders['train']):
                loss = self.train_step(batch, model_params)
                total_loss += loss
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Clear cache periodically
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        return total_loss / num_batches
    
    @torch.amp.autocast(device_type="cuda")  # Mixed precision for memory efficiency
    def train_step(self, batch, model_params):
        print("In training step function: ")
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        # Forward pass
        print("Before forward pass:")
        logits, _, _, _ = self.model(
            tokens=input_ids,
            model_params=model_params,
            cur_pos=0,
            freqs_cis=None  # Will be computed in forward
        )
        print(f"Logits grad: {logits.requires_grad}")
        # Calculate loss
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        print(f"Loss grad: {loss.requires_grad}")
        scaled_loss = loss / self.grad_accum_steps
        print(f"scaled loss grad: {scaled_loss.requires_grad}")
        # Backward pass
        scaled_loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"No gradient for {name}")
                else:
                    print(f"Gradient norm for {name}: {param.grad.norm()}")
        
        # Step optimizer
        if (self.optimizer.state['step'] + 1) % self.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
    
    def validate(self, model_params):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.dataloaders['val'])
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits, _, _, _ = self.model(
                    tokens=input_ids,
                    model_params=model_params,
                    cur_pos=0
                )
                
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                total_loss += loss.item()
                
        return total_loss / num_batches

def train_model(
    base_weights,
    train_path: str,
    model_params,
    val_path: Optional[str] = None,
    config: Optional[LoRAConfig] = None,
    num_epochs: int = 3,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda"
):
    # Initialize model
    model = LoRAModel(base_weights, config).to(device)
    for layer in model.layers:
        layer.requires_grad_(True)
    model.train()
    # Verify trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found!")
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        tokenizer=tokenizer,
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        device=device
    )
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
        train_loss = trainer.train_epoch(model_params, epoch)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = trainer.validate(model_params)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'lora_checkpoint_best.pt')
    
    return model

# Usage example:
if __name__ == "__main__":
    import logging
    import argparse
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('lora_training.log')
        ]
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train LoRA model on code data')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_path', type=str, help='Path to validation data (optional)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to base model weights')
    parser.add_argument('--output_dir', type=str, default='lora_checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load base model and tokenizer
        logging.info("Loading base model and tokenizer...")
        base_weights = load_weights(args.model_path)
        tokenizer = Tokenizer(r"checkpoints\Llama3.2-1B-Instruct\tokenizer.model")  # Initialize your tokenizer
        model_params = LLAMA_1B_PARAMS  # Your model parameters
        
        # Configure LoRA
        lora_config = LoRAConfig(
            r=8,  # LoRA rank
            alpha=16,  # LoRA scaling
            dropout=0.05,
            target_modules=("wq", "wk", "wv", "wo")  # Target attention matrices
        )
        
        # Memory optimization settings
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            logging.info(f"Available GPU memory: {gpu_memory / 1e9:.2f} GB")
            
            # Adjust batch size based on available memory
            if gpu_memory < 8e9:  # Less than 8GB
                args.batch_size = 1
                args.grad_accum_steps = 16
                logging.info("Limited GPU memory detected, adjusting batch settings")
        
        # Train model
        logging.info("Starting training...")
        trained_model = train_model(
            base_weights=base_weights,
            train_path=args.train_path,
            val_path=args.val_path,
            model_params=model_params,
            config=lora_config,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Save final model
        final_path = output_dir / 'lora_final.pt'
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': lora_config,
            'model_params': model_params,
        }, final_path)
        
        logging.info(f"Training complete. Final model saved to {final_path}")
        
        # Optional: Generate sample predictions
        trained_model.eval()
        sample_prompt = """
        ### Instruction:
        Write a Python function to calculate the Fibonacci sequence up to n terms.

        ### Response:
        """
        
        with torch.no_grad():
            tokens = tokenizer.encode(sample_prompt, bos=True, eos=False)
            tokens = torch.tensor([tokens]).cuda()
            
            output, _, _, _ = trained_model(
                tokens=tokens,
                model_params=model_params,
                cur_pos=0
            )
            
            # Generate response
            predicted_tokens = torch.argmax(output[0], dim=-1)
            response = tokenizer.decode(predicted_tokens.tolist())
            
            logging.info("\nSample generation:")
            logging.info(f"Prompt: {sample_prompt}")
            logging.info(f"Response: {response}")
            
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise