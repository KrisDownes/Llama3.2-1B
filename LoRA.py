#Implentation of https://arxiv.org/pdf/2106.09685
import json
import torch
import random
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, List
from tqdm import tqdm
import math
import logging
from torch.amp import autocast, GradScaler
from llama import xfmr, apply_rotary_emb, rms_norm, feed_forward,LLAMA_1B_PARAMS
from weights import XfmrWeights, LayerWeights, load_weights
from kvcache import KVCache
from tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
@dataclass
class LoRAConfig:
    r: int = 8  # LoRA rank
    alpha: int = 16  # LoRA scaling factor
    dropout: float = 0.05

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
        """
        Enhanced dataset class for code-alpaca format.
        
        Args:
            data_path: Path to the JSON dataset file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            prompt_template: Custom prompt template with keys: prefix, instruction, input, response
            add_code_markers: Whether to add ```python``` markers around code
            num_samples: Number of samples to load (None for all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Default prompt template
        self.prompt_template = prompt_template or {
            "prefix": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n",
            "instruction": "### Instruction:\n{instruction}\n\n",
            "input": "### Input:\n{input}\n\n" if "{input}" in "{input}" else "",
            "response": "### Response:\n{response}"
        }
        
        logging.info(f"Loading dataset from {data_path}")
        
        # Load and process data
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f) if data_path.endswith('.json') else [
                json.loads(line) for line in f
            ]
            
        # Sample subset if requested
        if num_samples and num_samples < len(data):
            data = random.sample(data, num_samples)
        
        for example in data:
            # Handle different possible field names
            instruction = example.get('instruction', example.get('prompt', ''))
            input_text = example.get('input', '')
            output = example.get('output', example.get('completion', ''))
            
            # Add code markers if enabled
            if add_code_markers and self._is_code_content(output):
                output = f"```python\n{output}\n```"
            
            # Format prompt using template
            prompt = self._format_prompt(instruction, input_text)
            
            # Tokenize with truncation
            prompt_tokens = self.tokenize(prompt, add_eos=False)
            output_tokens = self.tokenize(output, add_eos=True)
            
            # Skip if too long
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
        """Simple heuristic to detect if content is code"""
        code_indicators = ['def ', 'class ', 'import ', 'return ', '    ']
        return any(indicator in text for indicator in code_indicators)
    
    def _format_prompt(self, instruction: str, input_text: str = '') -> str:
        """Format prompt using template"""
        prompt = self.prompt_template['prefix']
        prompt += self.prompt_template['instruction'].format(instruction=instruction)
        
        if input_text:
            prompt += self.prompt_template['input'].format(input=input_text)
            
        return prompt
    
    def tokenize(self, text: str, add_eos: bool = False) -> List[int]:
        """Tokenize text with optional EOS token"""
        tokens = self.tokenizer.encode(
            text,
            bos=False,
            eos=add_eos,
            allowed_special='all'
        )
        
        return tokens

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized example with attention mask"""
        example = self.examples[idx]
        
        # Combine prompt and completion
        tokens = example['prompt'] + example['completion']
        
        # Create attention mask (1 for prompt, 0 for completion)
        attention_mask = [1] * len(example['prompt']) + [0] * len(example['completion'])
        
        # Pad sequences
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

def create_attention_mask(seq_len: int, start_pos: int ,device: str = "cuda") -> torch.Tensor:
    """Create causal attention mask"""
    if seq_len <= 1:
        return None
        
    # Create mask for current sequence positions
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    
    # Add mask for previous positions if any
    if start_pos > 0:
        prev_mask = torch.zeros((seq_len, start_pos), device=device)
        mask = torch.cat([prev_mask, mask], dim=1)
        
    return mask


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        with torch.no_grad():
            return run_function(*args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not any(grad_outputs):
            return (None,) + tuple(None for _ in range(len(ctx.saved_tensors)))

        inputs = ctx.saved_tensors
        with torch.enable_grad():
            detached_inputs = [x.detach().requires_grad_() for x in inputs]
            outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        torch.autograd.backward(
            outputs,
            grad_outputs,
            allow_unused=True
        )

        return (None,) + tuple(inp.grad for inp in detached_inputs)



class LoRALinearFunction:
    """LoRA adapter that preserves the original weights"""
    def __init__(self, weight: torch.Tensor, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        self.weight = weight.detach()  # Original pretrained weight
        self.rank = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.training = True
        
        # Initialize LoRA A and B matrices
        self.lora_A = torch.nn.Parameter(torch.zeros(weight.shape[1], r, device=weight.device, requires_grad=True))
        self.lora_B = torch.nn.Parameter(torch.zeros(r, weight.shape[0], device=weight.device, requires_grad= True))
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # Initialize with scaled random weights
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass with original weights
        base_output = F.linear(x, self.weight)
        
        # Add LoRA contribution during training
        if self.training:
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            lora_output = (self.dropout(x) @ self.lora_A) @ self.lora_B * self.scaling
            return base_output + lora_output
        return base_output

    def parameters(self):
        yield self.lora_A
        yield self.lora_B

def get_lora_model(xfmr_weights: XfmrWeights, model_params, config: LoRAConfig):
    """Wraps the original xfmr function with LoRA layers"""
    
    # Create LoRA layers for attention weights
    lora_layers = {}
    for i in range(model_params.n_layers):
        layer = xfmr_weights.layer_weights[i]
        lora_layers[f'layer_{i}_wq'] = LoRALinearFunction(layer.wq.clone(), config.r, config.alpha)
        lora_layers[f'layer_{i}_wk'] = LoRALinearFunction(layer.wk.clone(), config.r, config.alpha)
        lora_layers[f'layer_{i}_wv'] = LoRALinearFunction(layer.wv.clone(), config.r, config.alpha)
        lora_layers[f'layer_{i}_wo'] = LoRALinearFunction(layer.wo.clone(), config.r, config.alpha)
    
    def get_named_parameters():
        """Returns iterator of (name, parameter) pairs"""
        for name, layer in lora_layers.items():
            yield f"{name}.lora_A", layer.lora_A
            yield f"{name}.lora_B", layer.lora_B
    
    def get_parameters():
        """Returns just the parameters"""
        for layer in lora_layers.values():
            yield from layer.parameters()

    def lora_attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, cur_pos: int, 
                      layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, 
                      attn_mask: Optional[torch.Tensor] = None):
        """Modified attention function using LoRA layers"""
        bsz, seqlen, _ = x.shape
        n_rep = model_params.n_local_heads // model_params.n_local_kv_heads

        x.requires_grad_(True)
        # Use LoRA layers instead of original linear layers
        xq = lora_layers[f'layer_{layer_idx}_wq'](x)
        xk = lora_layers[f'layer_{layer_idx}_wk'](x)
        xv = lora_layers[f'layer_{layer_idx}_wv'](x)
        
        # Reshape and continue with original attention logic
        xq = xq.reshape(bsz, seqlen, model_params.n_local_heads, model_params.head_dim)
        xk = xk.reshape(bsz, seqlen, model_params.n_local_kv_heads, model_params.head_dim)
        xv = xv.reshape(bsz, seqlen, model_params.n_local_kv_heads, model_params.head_dim)

        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        #print(xk.shape,xv.shape,layer_idx,cur_pos,n_rep)
        keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)

        keys = keys.requires_grad_(True)
        values = values.requires_grad_(True)


        xq = xq.transpose(1, 2)  # [bsz, n_heads, seq_len, head_dim]
        keys = keys.transpose(1, 2).transpose(2, 3)  # [bsz, n_heads, head_dim, seq_len]
        values = values.transpose(1, 2)  # [bsz, n_heads, seq_len, head_dim]

    
        scores = torch.matmul(xq.to(torch.bfloat16), keys)
        scores = scores / math.sqrt(model_params.head_dim)
        scores = scores.to(torch.float32)
        if attn_mask is not None:
        # Expand mask for broadcasting across batch and head dimensions
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        
        scores = F.softmax(scores, dim=-1).to(torch.bfloat16)
        output = torch.matmul(scores, values)

        output = output.transpose(1, 2).reshape(bsz, seqlen, model_params.n_local_heads * model_params.head_dim)
        output = lora_layers[f'layer_{layer_idx}_wo'](output)
        return output, kvcache, scores
    
    def lora_forward(tokens: torch.Tensor, cur_pos: int, freqs_cis: torch.Tensor, 
                    kvcache: Optional[KVCache] = None, attn_mask: Optional[torch.Tensor] = None, 
                    training: bool = True):
        """Forward pass with LoRA layers"""
        # Set training mode for all LoRA layers
        for layer in lora_layers.values():
            layer.training = training
            
        h = xfmr_weights.tok_embeddings[tokens]
        if training:
            h.requires_grad_(True)
        for i in range(model_params.n_layers):
            norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
            h_attn, kvcache, scores = lora_attention(
                norm_x, xfmr_weights.layer_weights[i], model_params,
                cur_pos, i, freqs_cis, kvcache, attn_mask
            )
            h = h + h_attn
            h = h + feed_forward(
                rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm),
                xfmr_weights.layer_weights[i]
            )
            
        logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
        logits.requires_grad_(True)
        return logits, kvcache, scores, None
    
    def get_lora_state_dict():
        """Get LoRA weights for saving"""
        return {name: {'lora_A': layer.lora_A, 'lora_B': layer.lora_B}
                for name, layer in lora_layers.items()}
    
    def load_lora_state_dict(state_dict):
        """Load saved LoRA weights"""
        for name, weights in state_dict.items():
            if name in lora_layers:
                lora_layers[name].lora_A.data = weights['lora_A']
                lora_layers[name].lora_B.data = weights['lora_B']
    
    def get_parameters():
        """Get trainable LoRA parameters"""
        for layer in lora_layers.values():
            yield from layer.parameters()
    
    return lora_forward, get_parameters, get_named_parameters, get_lora_state_dict, load_lora_state_dict

def setup_training(xfmr_weights, model_params, device):
    """Setup LoRA training"""
    config = LoRAConfig(r=8, alpha=16, dropout=0.05)
    
    lora_forward, get_parameters, get_named_parameters, get_state_dict, load_state_dict = get_lora_model(
        xfmr_weights, model_params, config
    )
    
    optimizer = torch.optim.AdamW(
        get_parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Initialize scaler with higher initial scale
    scaler = GradScaler(
        init_scale=2**14,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True
    )
    
    return {
        'forward': lora_forward,
        'optimizer': optimizer,
        'scaler': scaler,
        'save_weights': get_state_dict,
        'load_weights': load_state_dict,
        'get_parameters': get_parameters,
        'get_named_parameters': get_named_parameters
    }

def forward_chunk(chunk_tokens, chunk_pos, chunk_mask, lora_forward, freqs_cis, model_params, device):
    """Wrapper for checkpointed forward pass with proper gradient handling"""
    chunk_size = chunk_tokens.shape[1]
    chunk_kvcache = KVCache(
        layers=model_params.n_layers,
        bsz=chunk_tokens.shape[0],
        max_seq_len=chunk_pos + chunk_size, 
        kv_heads=model_params.n_local_kv_heads,
        head_dim=model_params.head_dim
    )
    
    # Ensure tensors are properly cloned and detached
    #chunk_tokens = chunk_tokens.to(torch.float32)
    chunk_tokens = chunk_tokens.detach()
    #chunk_tokens.requires_grad_(True)
    
    chunk_freqs = freqs_cis[chunk_pos:chunk_pos + chunk_tokens.shape[1]]
    #chunk_freqs = chunk_freqs.real.to(torch.float32)
    #chunk_freqs.requires_grad_(True)

    
    attn_mask = create_attention_mask(chunk_size, chunk_pos, device=device)
    #if attn_mask is not None:
        #attn_mask = attn_mask.to(torch.float32)
        #attn_mask.requires_grad_(True)

    def run_forward(*args):
        tokens, freqs, mask = args
        # Convert tokens back to long for embedding lookup
        tokens = tokens.to(torch.long)
        with autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            logits, kvcache, scores, _ = lora_forward(
                tokens,
                cur_pos=chunk_pos,
                freqs_cis=freqs,
                kvcache=chunk_kvcache,
                attn_mask=mask,
                training=True
            )
        if torch.isnan(logits).any():
            logging.error(f"NaN detected in logits")
        return logits
    
    logits = CheckpointFunction.apply(run_forward, chunk_tokens, chunk_freqs, attn_mask)

    return logits
def training_step(batch_idx, accumulated_loss, valid_forward_passes, scaler, optimizer, max_grad_norm, 
                 train_setup, global_step, warmup_steps, learning_rate, gradient_accumulation_steps):
    """Optimized training step with better gradient handling"""
    if valid_forward_passes > 0 and (batch_idx + 1) % gradient_accumulation_steps == 0:
        try:
            # Log pre-scaling gradients if needed
            if batch_idx == 15:
                logging.info("Pre-scaling gradient norms:")
                for name, param in train_setup['get_named_parameters']():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        logging.info(f"{name}: {grad_norm}")

            # This is crucial - unscale before clip
            scaler.unscale_(optimizer)
            
            # Get parameters and check for NaN/inf before clipping
            parameters = list(train_setup['get_parameters']())
            has_nan_inf = False
            for p in parameters:
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        has_nan_inf = True
                        break
            
            if has_nan_inf:
                logging.warning(f"Found NaN/inf gradients at step {global_step}. Skipping optimization.")
                optimizer.zero_grad(set_to_none=True)
                return global_step
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            
            # Check if gradients are zero
            if grad_norm == 0.0:
                logging.warning(f"Gradient norm is zero at step {global_step}. This might indicate an issue.")
                optimizer.zero_grad(set_to_none=True)
                return global_step
            
            if torch.isfinite(grad_norm):
                if batch_idx == 15:
                    logging.info(f"Grad norm after clipping: {grad_norm}")
                    for name, param in train_setup['get_named_parameters']():
                        if param.grad is not None:
                            grad_max = param.grad.abs().max().item()
                            grad_mean = param.grad.abs().mean().item()
                            logging.info(f"{name}: grad_max = {grad_max}, grad_mean = {grad_mean}")
                
                # Perform optimization step
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                
                # Check if scale was reduced (indicating possible inf/nan)
                if scale_after < scale_before:
                    logging.warning(f"Gradient scaler reduced scale from {scale_before} to {scale_after}")
                
                optimizer.zero_grad(set_to_none=True)
                
                # Learning rate warmup
                if global_step < warmup_steps:
                    lr = learning_rate * (global_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                
                if global_step % 10 == 0:
                    avg_loss = accumulated_loss / (valid_forward_passes * gradient_accumulation_steps)
                    logging.info(
                        f"Step {global_step}: loss = {avg_loss:.4f}, "
                        f"lr = {optimizer.param_groups[0]['lr']:.2e}, "
                        f"grad_norm = {grad_norm:.2f}, "
                        f"scaler_scale = {scale_after}"
                    )
                
                return global_step + 1
            else:
                logging.warning(f"Skipping step {global_step} due to infinite gradients")
                optimizer.zero_grad(set_to_none=True)
                return global_step
                
        except RuntimeError as e:
            logging.error(f"Error in optimization step: {str(e)}")
            optimizer.zero_grad(set_to_none=True)
            return global_step


def train_lora(
    xfmr_weights,
    model_params,
    train_data_path: str,
    output_dir: str,
    tokenizer_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
    save_steps: int = 1000,
    max_length: int = 2048,
    checkpoint_factor: int = 4,
    val_split: float = 0.1 
):
    """Memory-efficient LoRA training with fixed gradient handling"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize training components
    train_setup = setup_training(xfmr_weights, model_params, device)
    lora_forward = train_setup['forward']
    optimizer = train_setup['optimizer']
    scaler = train_setup['scaler']
    
    # Setup data
    tokenizer = Tokenizer(tokenizer_path)
    dataloaders = create_code_dataloaders(
        train_path=train_data_path,
        val_path=None, 
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=1,
        val_split=val_split
    )
    
    freqs_cis = precompute_freqs_cis(
        model_params.head_dim,
        max_length * 2,
        device=device
    )
    
    # Training loop
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloaders['train'])):
            optimizer.zero_grad(set_to_none=True)
            
            # Move tensors to device
            tokens = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get proper sequence length
            valid_mask = labels != 128004  # padding token
            seq_length = valid_mask.sum(dim=1).max().item()
            chunk_size = max(seq_length // checkpoint_factor, 1)
            
            if batch_idx == 15:
                logging.info(f"Total sequence length: {seq_length}")
                logging.info(f"Chunk size: {chunk_size}")
                logging.info(f"Number of full chunks: {seq_length // chunk_size}")
                logging.info(f"Remainder tokens: {seq_length % chunk_size}")
            
            accumulated_loss = 0
            valid_forward_passes = 0
            
            # Process chunks
            for chunk_start in range(0, seq_length - chunk_size + 1, chunk_size):
                chunk_end = chunk_start + chunk_size
                
                # Extract chunk data
                chunk_tokens = tokens[:, chunk_start:chunk_end].clone()
                chunk_attention = attention_mask[:, chunk_start:chunk_end].clone()
                chunk_labels = labels[:, chunk_start:chunk_end].clone()
                
                # Verify chunk has valid tokens
                chunk_valid = (chunk_labels != 128004).any()
                if not chunk_valid:
                    if batch_idx == 15:
                        logging.info(f"Skipping chunk {chunk_start}-{chunk_end} - no valid tokens")
                    continue
                    
                try:
                    with autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
                        logits = forward_chunk(
                            chunk_tokens,
                            chunk_start,
                            chunk_attention,
                            lora_forward,
                            freqs_cis,
                            model_params,
                            device
                        )
                        
                        pred_logits = logits[:, :-1, :]
                        target_tokens = chunk_labels[:, 1:]
                        
                        valid_targets = (target_tokens != 128004)
                        if valid_targets.any():
                            pred_logits = pred_logits.to(torch.float32)
                            pred_logits.requires_grad_(True)
                            chunk_loss = F.cross_entropy(
                                pred_logits.reshape(-1, pred_logits.size(-1))[valid_targets.reshape(-1)],
                                target_tokens.reshape(-1)[valid_targets.reshape(-1)],
                                reduction='mean'
                            )
                            
                            # Scale the loss properly
                            scaled_loss = chunk_loss / (gradient_accumulation_steps * checkpoint_factor)
                            # Make sure to scale the loss before backward
                            scaler.scale(scaled_loss).backward()
                            
                            accumulated_loss += chunk_loss.item()
                            valid_forward_passes += 1
                        
                except RuntimeError as e:
                    logging.error(f"Error processing chunk {chunk_start}-{chunk_end}: {str(e)}")
                    continue
                    
                del logits
                if 'chunk_loss' in locals():
                    del chunk_loss, scaled_loss
                torch.cuda.empty_cache()
            
            # Process final chunk if needed
            remainder = seq_length % chunk_size
            if remainder > 0:
                chunk_start = seq_length - remainder
                try:
                    chunk_tokens = tokens[:, chunk_start:].clone()
                    chunk_attention = attention_mask[:, chunk_start:].clone()
                    chunk_labels = labels[:, chunk_start:].clone()
                    
                    if (chunk_labels != 128004).any():
                        with autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
                            logits = forward_chunk(
                                chunk_tokens,
                                chunk_start,
                                chunk_attention,
                                lora_forward,
                                freqs_cis,
                                model_params,
                                device
                            )
                            
                            pred_logits = logits[:, :-1, :]
                            target_tokens = chunk_labels[:, 1:]
                            
                            valid_targets = (target_tokens != 128004)
                            if valid_targets.any():
                                pred_logits = pred_logits.to(torch.float32)
                                pred_logits.requires_grad_(True)
                                chunk_loss = F.cross_entropy(
                                    pred_logits.reshape(-1, pred_logits.size(-1))[valid_targets.reshape(-1)],
                                    target_tokens.reshape(-1)[valid_targets.reshape(-1)],
                                    reduction='mean'
                                )
                                
                                scaled_loss = chunk_loss / (gradient_accumulation_steps * checkpoint_factor)
                                scaler.scale(scaled_loss).backward()
                                
                                accumulated_loss += chunk_loss.item()
                                valid_forward_passes += 1
                        
                        del logits
                        if 'chunk_loss' in locals():
                            del chunk_loss, scaled_loss
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    logging.error(f"Error processing final chunk: {str(e)}")
            
            # Optimization step using the new training_step function
            global_step = training_step(
                batch_idx=batch_idx,
                accumulated_loss=accumulated_loss,
                valid_forward_passes=valid_forward_passes,
                scaler=scaler,
                optimizer=optimizer,
                max_grad_norm=max_grad_norm,
                train_setup=train_setup,
                global_step=global_step,
                warmup_steps=warmup_steps,
                learning_rate=learning_rate,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            # Clear batch tensors
            del tokens, attention_mask, labels
            torch.cuda.empty_cache()
            
            num_batches += 1
            epoch_loss += accumulated_loss
        
        # End of epoch logging and checkpointing
        avg_epoch_loss = epoch_loss / (num_batches + 1e-8)
        logging.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        torch.save(
            {
                'lora_weights': train_setup['save_weights'](),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'loss': avg_epoch_loss
            },
            output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        )
    
    # Save final model
    torch.save(
        {
            'lora_weights': train_setup['save_weights'](),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'global_step': global_step,
            'epoch': num_epochs,
            'loss': avg_epoch_loss
        },
        output_dir / "final_model.pt"
    )

def precompute_freqs_cis(dim: int, end: int, device: str = "cuda") -> torch.Tensor:
    """Precompute the frequency cis matrix for rotary embeddings."""
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

# Example usage
if __name__ == "__main__":
    train_lora(
        xfmr_weights=load_weights(r'checkpoints\Llama3.2-1B-Instruct\consolidated.00.pth'),
        model_params=LLAMA_1B_PARAMS,
        train_data_path=r"D:\Code\ML_LEARN\.llama3\Llama3.2-1B\fineTune\code_alpaca_20k.json",
        output_dir="lora_checkpoints",
        tokenizer_path=r"checkpoints\Llama3.2-1B-Instruct\tokenizer.model",
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,  # Reduced learning rate
        max_grad_norm=0.5,
        checkpoint_factor=4,
        warmup_steps=500  # Adjust based on your GPU memory
    )