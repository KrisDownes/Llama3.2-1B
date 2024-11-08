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
        self.weight = weight  # Original pretrained weight
        self.rank = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.training = False
        
        # Initialize LoRA A and B matrices
        self.lora_A = torch.nn.Parameter(torch.zeros(weight.shape[1], r, device=weight.device))
        self.lora_B = torch.nn.Parameter(torch.zeros(r, weight.shape[0], device=weight.device))
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # Initialize with scaled random weights
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass with original weights
        base_output = F.linear(x, self.weight)
        
        # Add LoRA contribution during training
        if self.training:
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
        lora_layers[f'layer_{i}_wq'] = LoRALinearFunction(layer.wq, config.r, config.alpha)
        lora_layers[f'layer_{i}_wk'] = LoRALinearFunction(layer.wk, config.r, config.alpha)
        lora_layers[f'layer_{i}_wv'] = LoRALinearFunction(layer.wv, config.r, config.alpha)
        lora_layers[f'layer_{i}_wo'] = LoRALinearFunction(layer.wo, config.r, config.alpha)
    
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

        x = x.detach().requires_grad_()
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

        keys = keys.detach().requires_grad_()
        values = values.detach().requires_grad_()


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
                    training: bool = False):
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
    
    scaler = GradScaler(device=device)
    
    return {
        'forward': lora_forward,
        'optimizer': optimizer,
        'scaler': scaler,
        'save_weights': get_state_dict,
        'load_weights': load_state_dict,
        'get_parameters': get_parameters,
        'get_named_parameters': get_named_parameters
    }

def train_lora(
    xfmr_weights,
    model_params,
    train_data_path: str,
    output_dir: str,
    tokenizer_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,  # Increased for smaller batch size
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
    save_steps: int = 1000,
    max_length: int = 2048,
    checkpoint_factor: int = 4,# Split sequence into this many chunks for checkpointing
    val_split: float = 0.1 
):
    """
    Memory-efficient LoRA training implementation
    """
    def checkpoint(function, *args):
        """
        Checkpoint wrapper for memory efficient computation.
        Moved inside training function for proper scoping.
        """
        return CheckpointFunction.apply(function, *args)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize training components
    train_setup = setup_training(xfmr_weights, model_params, device)
    lora_forward = train_setup['forward']
    optimizer = train_setup['optimizer']
    scaler = train_setup['scaler']
    
    # Setup data
    tokenizer = Tokenizer(tokenizer_path)

    prompt_template = {
        "prefix": "",  # Empty prefix since we're using direct instruction format
        "instruction": "### Instruction:\n{instruction}\n\n",
        "input": "### Input:\n{input}\n\n",
        "response": "### Response:\n{response}"
    }
    dataloaders = create_code_dataloaders(
        train_path=train_data_path,
        val_path=None, 
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=1,
        val_split=val_split,
        prompt_template=prompt_template
    )
    freqs_cis = precompute_freqs_cis(
        model_params.head_dim,
        max_length * 2,
        device=device
    )
    
    
    
    def forward_chunk(chunk_tokens, chunk_pos, chunk_mask):
        """Wrapper for checkpointed forward pass"""
        chunk_size = chunk_tokens.shape[1]
        chunk_kvcache = KVCache(
            layers=model_params.n_layers,
            bsz=chunk_tokens.shape[0],
            max_seq_len=chunk_pos + chunk_size, 
            kv_heads=model_params.n_local_kv_heads,
            head_dim=model_params.head_dim
        )
         
        attn_mask = create_attention_mask(chunk_size, chunk_pos, device=chunk_tokens.device)
        def _forward():
            with torch.set_grad_enabled(True):
                logits, _, _, _ = lora_forward(
                    chunk_tokens,
                    cur_pos=chunk_pos,
                    freqs_cis=freqs_cis[chunk_pos:chunk_pos + chunk_tokens.shape[1]],
                    kvcache=chunk_kvcache,
                    attn_mask=attn_mask,
                    training=True
                )
            return logits
        return checkpoint(_forward)    
    # Training loop
    global_step = 0
    optimizer.zero_grad(set_to_none = True)
    best_loss = float('inf')

    # Add debug logging
    logging.info("Initializing training loop with settings:")
    logging.info(f"Device: {device}")
    logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logging.info(f"Checkpoint factor: {checkpoint_factor}")
    logging.info(f"Mixed precision enabled: {True}")

    # Verify optimizer setup
    logging.info(f"Optimizer type: {type(optimizer)}")
    logging.info(f"Number of parameter groups: {len(optimizer.param_groups)}")

    # Initialize grad scaler state explicitly
    scaler_state = scaler.state_dict()
    logging.info(f"Initial scaler state: {scaler_state}")

    # Ensure all LoRA parameters require gradients
    for param in train_setup['get_parameters']():
        param.requires_grad = True
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}")):

            optimizer.zero_grad(set_to_none=True)

            tokens = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            batch_size = tokens.shape[0]
            seq_len = tokens.shape[1]
            
            # Split sequence into chunks for checkpointing
            chunk_size = max(seq_len // checkpoint_factor, 1)
            
            accumulated_loss = 0
            valid_forward_passes = 0
            
            # Process sequence in chunks
            for chunk_idx in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_idx + chunk_size, seq_len)
                chunk_tokens = tokens[:, chunk_idx:chunk_end]
                chunk_attention = attention_mask[:, chunk_idx:chunk_end]
                chunk_labels = labels[:, chunk_idx:chunk_end]
                # Create attention mask for chunk
                chunk_mask = create_attention_mask(chunk_end - chunk_idx, chunk_idx, device)
                
                try:
                    with autocast(device_type=device, enabled=True):

                        logits = forward_chunk(chunk_tokens, chunk_idx, chunk_mask)
                        if not logits.requires_grad:
                            logging.warning(f"Logits missing gradient at chunk {chunk_idx}")
                            continue
                        
                        if chunk_end < seq_len:
                            target_tokens = chunk_labels[:, 1:]
                            pred_logits = logits[:, :-1]
                        else:
                            target_tokens = chunk_labels[:, 1:chunk_end-chunk_idx]
                            pred_logits = logits[:, :chunk_end-chunk_idx-1]
                        
                        # Verify shapes before loss calculation
                        logging.debug(f"Pred logits shape: {pred_logits.shape}")
                        logging.debug(f"Target tokens shape: {target_tokens.shape}")
                        


                        if pred_logits.shape[:-1] != target_tokens.shape:
                            logging.error(f"Shape mismatch: pred_logits {pred_logits.shape}, targets {target_tokens.shape}")
                            continue
                        if not pred_logits.requires_grad:
                            logging.warning("Pred logits doesn't require grad, enabling...")
                            pred_logits.requires_grad_(True)
                        
                        chunk_loss = F.cross_entropy(
                            pred_logits.reshape(-1, pred_logits.size(-1)),
                            target_tokens.reshape(-1),
                            ignore_index=tokenizer.pad_id
                        )
                        
                        scaled_loss = chunk_loss / (gradient_accumulation_steps * checkpoint_factor)
                        scaler.scale(scaled_loss).backward()
                        
                        # Verify loss values
                        if not torch.isfinite(chunk_loss):
                            logging.error(f"Non-finite loss detected: {chunk_loss.item()}")
                            continue
                            
                        logging.debug(f"Chunk loss: {chunk_loss.item()}, Scaled loss: {scaled_loss.item()}")
                    
                    # Scale and backward
                    scaler.scale(scaled_loss).backward()
                    valid_forward_passes += 1
                    accumulated_loss += chunk_loss.item()
                    
                except RuntimeError as e:
                    logging.error(f"Error in forward/backward pass: {str(e)}")
                    continue
                
                del logits, chunk_loss
                torch.cuda.empty_cache()
            
            # Gradient accumulation step
            if valid_forward_passes > 0 and (batch_idx + 1) % gradient_accumulation_steps == 0:
                try:
                    # Check gradient norms before unscaling
                    grad_norms_before = {}
                    for name, param in train_setup['get_named_parameters']():
                        if param.grad is not None:
                            grad_norms_before[name] = param.grad.norm().item()
                    logging.debug(f"Gradient norms before unscaling: {grad_norms_before}")
                    
                    # Unscale gradients
                    scaler.unscale_(optimizer)
                    
                    # Check gradient norms after unscaling
                    grad_norms_after = {}
                    for name, param in train_setup['get_named_parameters']():
                        if param.grad is not None:
                            grad_norms_after[name] = param.grad.norm().item()
                    logging.debug(f"Gradient norms after unscaling: {grad_norms_after}")

                    # Get all parameters as a list for gradient clipping
                    parameters = list(train_setup['get_parameters']())
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                        scaler.update()
                        logging.debug(f"Successfully completed optimizer step {global_step}")
                    else:
                        logging.warning(f"Skipping step {global_step} due to infinite gradients")
                    
                    optimizer.zero_grad(set_to_none=True)
                except RuntimeError as e:
                    logging.error(f"Error in optimization step: {str(e)}")
                    continue
                if batch_idx == 14:  # The batch before the error occurs
                    logging.info("Detailed state at batch 14:")
                    logging.info(f"Scaler state: {scaler.state_dict()}")
                    logging.info(f"Optimizer state keys: {optimizer.state_dict().keys()}")
                    logging.info(f"Valid forward passes: {valid_forward_passes}")
                    logging.info(f"Accumulated loss: {accumulated_loss}")
                # Learning rate warmup
                if global_step < warmup_steps:
                        lr = learning_rate * (global_step + 1) / warmup_steps
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    
                epoch_loss += accumulated_loss
                num_batches += 1
                
                if global_step % 10 == 0:
                        avg_loss = accumulated_loss / gradient_accumulation_steps
                        logging.info(
                            f"Step {global_step}: loss = {avg_loss:.4f}, "
                            f"lr = {optimizer.param_groups[0]['lr']:.2e}, "
                            f"grad_norm = {grad_norm:.2f}"
                        )
                    
                global_step += 1

                # Validation phase
                if global_step > 0 and global_step % save_steps == 0:
                    model_val_loss = 0
                    num_val_batches = 0
                    
                    # Switch to evaluation mode
                    for layer in train_setup['get_parameters']():
                        layer.training = False
                    
                    with torch.no_grad():
                        for val_batch in dataloaders['val']:
                            val_tokens = val_batch['input_ids'].to(device)
                            val_attention = val_batch['attention_mask'].to(device)
                            val_labels = val_batch['labels'].to(device)
                            
                            # Simple forward pass for validation (no chunking needed)
                            with autocast(device_type=device):
                                val_logits = lora_forward(
                                    val_tokens,
                                    cur_pos=0,
                                    freqs_cis=freqs_cis[:val_tokens.shape[1]],
                                    attn_mask=create_attention_mask(val_tokens.shape[1], 0, device),
                                    training=False
                                )[0]
                                
                                val_loss = F.cross_entropy(
                                    val_logits[:, :-1].reshape(-1, val_logits.size(-1)),
                                    val_labels[:, 1:].reshape(-1),
                                    ignore_index=tokenizer.pad_id
                                )
                            
                            model_val_loss += val_loss.item()
                            num_val_batches += 1
                # Switch back to training mode
                    for layer in train_setup['get_parameters']():
                        layer.training = True
                # Calculate average validation loss
                    avg_val_loss = model_val_loss / num_val_batches

                if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        is_best = True

                        # Save best model
                        checkpoint = {
                            'lora_weights': train_setup['save_weights'](),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                            'train_loss': avg_loss,
                            'val_loss': avg_val_loss
                        }
                        
                        torch.save(checkpoint, output_dir / "best_model.pt")
                        logging.info(f"Saved new best model with validation loss: {avg_val_loss:.4f}")
                
                global_step += 1
        # End of epoch logging
        avg_epoch_loss = epoch_loss / num_batches
        logging.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")

    
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