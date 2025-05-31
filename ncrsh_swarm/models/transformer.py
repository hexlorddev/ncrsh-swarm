"""
SwarmTransformer: Neural network model optimized for distributed training
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for SwarmTransformer"""
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    bias: bool = True
    layer_norm_epsilon: float = 1e-5
    
    # Swarm-specific optimizations
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    optimize_for_distributed: bool = True


class MultiHeadAttention(nn.Module):
    """Multi-head attention optimized for distributed training"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        assert self.head_dim * config.num_heads == config.hidden_size
        
        # Linear projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            )
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence, channels
        
        # Calculate Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x) 
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
            
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = 4 * config.hidden_size  # Standard 4x expansion
        
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = F.gelu(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class SwarmTransformer(nn.Module):
    """
    Transformer model optimized for distributed swarm training
    
    Features:
    - Gradient checkpointing for memory efficiency
    - Mixed precision training support
    - Optimized parameter layouts for communication
    - Built-in model synchronization utilities
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying (share embeddings with output layer)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections
        for name, param in self.named_parameters():
            if name.endswith('out_proj.weight') or name.endswith('down_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))
                
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
            
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.size()
        
        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = self.embedding_dropout(token_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            if self.config.enable_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, attention_mask)
            else:
                x = block(x, attention_mask)
                
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1
            )
            
        return logits, loss
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text autoregressively"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                logits, _ = self.forward(input_ids)
                
            # Get logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
                
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
                
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for sequence length limit
            if input_ids.size(1) >= self.config.max_seq_len:
                break
                
        return input_ids
        
    def get_parameter_groups(self) -> list:
        """Get parameter groups for optimized distributed training"""
        # Separate weight decay and no weight decay parameters
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(keyword in name for keyword in ['bias', 'ln', 'layernorm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                    
        return [
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
    def get_flops_per_token(self, seq_len: int) -> int:
        """Calculate FLOPs per token for performance monitoring"""
        N = self.get_num_params()
        L = self.config.num_layers
        H = self.config.hidden_size
        Q = self.config.num_heads
        T = seq_len
        
        # Attention FLOPs
        attn_flops = L * (
            4 * H * T +  # QKV projections
            2 * H * T * T / Q +  # Attention computation
            2 * H * T  # Output projection
        )
        
        # MLP FLOPs
        mlp_flops = L * 8 * H * T  # 4H intermediate size
        
        # Embedding and output FLOPs
        embed_flops = 2 * H * T
        
        return int(attn_flops + mlp_flops + embed_flops)
        
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
        
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
        
    def estimate_memory_usage(self, batch_size: int, seq_len: int) -> dict:
        """Estimate memory usage for given batch size and sequence length"""
        # Model parameters
        model_memory = self.get_model_size_mb()
        
        # Activations (rough estimate)
        activation_memory = (
            batch_size * seq_len * self.config.hidden_size * 
            self.config.num_layers * 4  # 4 bytes per float32
        ) / (1024 * 1024)
        
        # Gradients (same size as parameters)
        gradient_memory = model_memory
        
        # Optimizer state (AdamW uses 2x parameter memory)
        optimizer_memory = model_memory * 2
        
        return {
            'model_mb': model_memory,
            'activations_mb': activation_memory,
            'gradients_mb': gradient_memory,
            'optimizer_mb': optimizer_memory,
            'total_mb': model_memory + activation_memory + gradient_memory + optimizer_memory
        }