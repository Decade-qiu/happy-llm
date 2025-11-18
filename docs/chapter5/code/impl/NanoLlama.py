import math
from re import A
from typing import Optional
import warnings
from sympy import Float
import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor, nn

from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "NanoLlama"
    def __init__(
        self,
        dim: int = 1024,
        num_layer: int = 18,
        num_head: int = 16,
        num_kv_head: int = 4,
        vocab_size: int = 6144,
        hidden_dim: int = 0,
        multiple_of: int = 64,
        norm_eps: float = 1e-6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        flash_attn: bool = True,
        **kwargs,
    ):
        self.dim = dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)



class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Args:
        hidden_size (int): Size of the hidden layer.
        eps (float): A small value to prevent division by zero.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # prevent division by zero
        self.eps = eps
        # weight is a learnable parameter
        # we initialize it to ones, to start with no scaling
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor):
        # conserve input dtype to improve performance
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding

    Args:
        dim (int): Dimension of the model.
        max_len (int): Maximum sequence length.
    """
    sin: Tensor
    cos: Tensor

    def __init__(self, dim, max_len=1024):
        super().__init__()
        # (dim/2,)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # (max_len,)
        t = torch.arange(max_len).float()
        # (max_len, dim/2)
        freqs = torch.outer(t, inv_freq)
        # (1, 1, max_len, dim)
        freqs = torch.concat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, q, k):
        # q, k: (batch_size, num_head, seq_len, dim)
        sq, sk = q.shape[-2], k.shape[-2]
        # RoPE: x = x * cos + rotate_half(x) * sin
        # That is: x = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        # x1 is first half of last dim, x2 is second half of last dim
        q_emb = (q * self.cos[:, :, :sq, :]) + (rotate_half(q) * self.sin[:, :, :sq, :])
        k_emb = (k * self.cos[:, :, :sk, :]) + (rotate_half(k) * self.sin[:, :, :sk, :])
        return q_emb, k_emb
    


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat key or value tensor for Grouped Query Attention.

    Args:
        x (Tensor): Input tensor of shape [B, num_kv_head, S, head_dim].
        n_rep (int): Number of repetitions.

    Returns:
        Tensor: Repeated tensor of shape [B, num_head, S, head_dim].
    """
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.size()
    # expand the kv heads
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, S, D)
    return x.reshape(B, H_kv * n_rep, S, D)

class Attention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention and RoPE

    Args:
        args (ModelConfig): Model configuration parameters.
    """
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.dim = args.dim
        self.num_head = args.num_head
        # Grouped Query Attention
        # one kv head used by num_head // num_kv_head query heads
        self.num_kv_head = args.num_kv_head
        assert self.num_head % self.num_kv_head == 0, "num_head must be divisible by num_kv_head"
        # calculate number of repetitions for kv heads
        self.num_rep = self.num_head // self.num_kv_head
        self.head_dim = args.dim // args.num_head

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_len=args.max_seq_len)

        # define Q, K, V, O projection matrices
        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.num_kv_head * self.head_dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.num_kv_head * self.head_dim, bias=False)
        self.W_o = nn.Linear(self.num_head * self.head_dim, args.dim, bias=False)

        # attention and residual dropout layers
        self.dropout = args.dropout
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # check if Flash Attention is supported
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.attn_dropout = nn.Dropout(self.dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        B, S = x.size(0), x.size(1)

        # linear transform the input and get Q, K, V 
        Q: Tensor = self.W_q(x)
        K: Tensor = self.W_k(x)
        V: Tensor = self.W_v(x)

        # reshape Q, K, V for multi-head attention
        # Q: [B, num_head, S, head_dim]
        Q = Q.view(B, S, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        # K, V: [B, num_kv_head, S, head_dim]
        K = K.view(B, S, self.num_kv_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, S, self.num_kv_head, self.head_dim).permute(0, 2, 1, 3)

        # apply RoPE to Q and K
        Q, K = self.rope(Q, K)

        # repeat K and V for Grouped Query Attention
        K = repeat_kv(K, self.num_rep)
        V = repeat_kv(V, self.num_rep)

        if self.flash:
            output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=mask is None
            )
        else:
            attention = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert mask is not None, "mask is required for non-flash attention"
            attention = attention.masked_fill(mask == 0, float('-inf'))
            attention = F.softmax(attention.float(), -1).to(Q.dtype)
            scores = self.attn_dropout(attention)
            output = scores @ V

        # reshape output back to [B, S, num_head * head_dim]
        output = output.permute(0, 2, 1, 3).reshape(B, S, self.num_head * self.head_dim)

        # final linear projection and dropout
        output = self.resid_dropout(self.W_o(output))

        return output

class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) with SiLU activation

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden layer dimension.
        multiple_of (int): Ensures hidden_dim is a multiple of this value.
        dropout (float): Dropout rate.
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # if hidden_dim is not specified, calculate it based on dim and multiple_of
        # origin transformer hidden_dim = 4 * dim, total params = 2 * dim * hidden_dim
        # = 8 * dim^2
        # To maintain similar parameter count, we set hidden_dim = 8/3 * dim
        # then round it to be multiple_of
        # Total params = dim * hidden_dim + hidden_dim * dim + dim * hidden_dim
        # = 3 * dim * hidden_dim
        # 3 * dim * hidden_dim = 8 * dim^2 => hidden_dim = 8/3 * dim
        if hidden_dim == 0:
            hidden_dim = int(8 * dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
    

class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer

    Args:
        layer_id (int): Layer index.
        args (ModelConfig): Model configuration parameters.
    """
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.attention = Attention(args)
        self.mlp = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        h = x + self.attention(self.attention_norm(x), mask)
        out = h + self.mlp(self.ffn_norm(h))
        return out

class NanoLlama(PreTrainedModel):
    """
    NanoLlama Model

    Args:
        args (ModelConfig): Model configuration parameters.
    """

    def __init__(self, args: ModelConfig):
        super().__init__(args)
        self.config = args

        # input embedding layer
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim, args.pad_token_id)
        self.dropout = nn.Dropout(args.dropout)

        # decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(layer_id=i, args=args) for i in range(args.num_layer)
        ])

        # RMSNorm layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # output projection layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # share input and output embeddings
        self.output.weight = self.embed_tokens.weight

        # initialize
        self._no_split_modules = [name for name, _ in self.named_modules()]
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if 'down_proj.weight' in name or 'W_o.weight' in name:
                param.data.normal_(
                    mean=0.0, 
                    std=0.02 / math.sqrt(2 * self.config.num_layer)
                )

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self, 
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            input_ids (Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (Optional[Tensor]): Attention mask of shape (batch_size, seq_len).
            labels (Optional[Tensor]): Target token IDs for computing the loss.
        """

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        return logits

    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Normal text generation function using greedy decoding or sampling.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. 0.0 for greedy decoding.
            top_k: If specified, only consider the top_k tokens for sampling.

        Returns:
            Generated token IDs of shape (batch_size, seq_len + generated_length).
        """
        # cut the context if it's too long
        if input_ids.size(1) > self.config.max_seq_len:
            warnings.warn(f"Input sequence length {input_ids.size(1)} exceeds max_seq_len {self.config.max_seq_len}, truncating.")
            input_ids = input_ids[:, -self.config.max_seq_len:]

        vocab_size = self.config.vocab_size
        eos_id = self.config.eos_token_id if self.config.eos_token_id is not None else vocab_size - 1

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Forward pass to get logits
            logits = self(input_ids)
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            
            if temperature == 0.0:
                # use greedy decoding
                _, next_token = torch.topk(next_token_logits, k=1, dim=-1)
            else:
                # scale logits by temperature
                next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, vocab_size))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # concat the new token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if input_ids.size(0) == 1:
                if next_token.item() == eos_id:
                    break   
            else:
                if (next_token.view(-1) == eos_id).all():
                    break

        return input_ids
    

    @torch.inference_mode()
    def generate_super(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        num_beams: int = 1,
    ):
        """
        Advanced text generation function supporting greedy decoding, sampling, and beam search.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. 0.0 for greedy decoding.
            top_k: If specified, only consider the top_k tokens for sampling.
            do_sample: Whether to use sampling. If False, uses greedy decoding.
            num_beams: Number of beams to use in beam search. If > 1 and do_sample is False, uses beam search.

        Returns:
            Generated token IDs of shape (batch_size, seq_len + generated_length).
        """
        if temperature <= 0:
            temperature = 0.001
        if num_beams < 1:
            num_beams = 1
        if top_k is not None and top_k < 1:
            top_k = None

        if not do_sample and num_beams > 1:
            return self._beam_search(input_ids, max_new_tokens, num_beams, temperature, top_k)

        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            logits = self(idx_cond).logits
            logits = logits[:, -1, :]
            if do_sample:
                next_token = self._random_sample(logits, temperature, top_k)
            else:
                if temperature < 0.1:
                    next_token = self._greedy_decode(logits)
                else:
                    next_token = self._random_sample(logits, temperature, top_k)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if input_ids.size(0) == 1:
                if next_token.item() == self.config.eos_token_id:
                    break   
            else:
                if (next_token.view(-1) == self.config.eos_token_id).all():
                    break

        return input_ids
    
    def _greedy_decode(self, logits: Tensor) -> Tensor:
        """
        Greedy Decoding: Select the token with the highest probability

        Args:
            logits: Model output logits of shape (batch_size, vocab_size)
        """
        _, next_token = torch.topk(logits, k=1, dim=-1)
        return next_token

    def _random_sample(self, logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
        """
        Random Sampling: Sample the next token based on the probability distribution

        Args:
            logits: Model output logits of shape (batch_size, vocab_size)
            temperature: Temperature parameter to control randomness
            top_k: If specified, only consider the top_k tokens for sampling
        """
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def _beam_search(
        self, 
        input_ids: Tensor, 
        max_new_tokens: int, 
        num_beams: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None,
        length_penalty: float = 1.0,
    ) -> Tensor:
        """
        Beam Search Decoding

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            num_beams: Number of beams to use in beam search.
            temperature: Temperature parameter to control randomness.
            top_k: If specified, only consider the top_k tokens for sampling.
            length_penalty: Length penalty factor (>1.0 encourages longer sequences).
        """
        batch_size = input_ids.shape[0]
        eos_id = self.config.eos_token_id if self.config.eos_token_id is not None else self.config.vocab_size - 1

        # (batch_size, seq_len) -> (batch_size * num_beams, seq_len)
        beams = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)
        
        # 初始化 beam scores: (batch_size, num_beams)
        beam_scores = torch.zeros(batch_size, num_beams, device=input_ids.device, dtype=torch.float)
        beam_scores[:, 1:] = float('-inf')
        beam_scores = beam_scores.view(-1)  # (batch_size * num_beams,)

        is_finished = torch.zeros(batch_size * num_beams, device=input_ids.device, dtype=torch.bool)
        
        for step in range(max_new_tokens):
            if beams.size(1) > self.config.max_seq_len:
                beams = beams[:, -self.config.max_seq_len:]
            outputs = self(beams)
            # (batch_size * num_beams, vocab_size)
            logits = outputs.logits[:, -1, :]
            if temperature > 0.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[:, [-1]]] = float('-inf')
            log_probs = F.log_softmax(logits, dim=-1)
            
            # for finished beams, only allow eos_id
            log_probs[is_finished, :] = float('-inf')
            log_probs[is_finished, eos_id] = 0.0
            
            vocab_size = log_probs.size(-1)
            
            # (batch_size, num_beams, vocab_size)
            log_probs = log_probs.view(batch_size, num_beams, vocab_size)
            beam_scores_reshaped = beam_scores.view(batch_size, num_beams)
            next_scores = beam_scores_reshaped.unsqueeze(2) + log_probs
            if length_penalty != 1.0:
                current_length = beams.size(1) - input_ids.size(1) + 1
                next_scores = next_scores / (current_length ** length_penalty)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            # top_scores, top_indices: (batch_size, num_beams)
            top_scores, top_indices = torch.topk(next_scores, k=num_beams, dim=1, largest=True, sorted=True)
            
            beam_indices = top_indices // vocab_size  # (batch_size, num_beams)
            token_indices = top_indices % vocab_size   # (batch_size, num_beams)
            
            batch_offsets = torch.arange(batch_size, device=input_ids.device).unsqueeze(1) * num_beams
            flat_beam_indices = (batch_offsets + beam_indices).view(-1)
            
            beams = torch.cat([
                beams[flat_beam_indices],
                token_indices.view(-1, 1)
            ], dim=1)
            beam_scores = top_scores.view(-1)  # (batch_size * num_beams,)

            is_finished = is_finished[flat_beam_indices] | (token_indices.view(-1) == eos_id)

            is_finished_reshaped = is_finished.view(batch_size, num_beams)
            if is_finished_reshaped.all():
                break

        beam_scores_reshaped = beam_scores.view(batch_size, num_beams)
        best_beam_indices = beam_scores_reshaped.argmax(dim=1)

        batch_offsets = torch.arange(batch_size, device=input_ids.device) * num_beams
        flat_best_indices = batch_offsets + best_beam_indices
        best_beams = beams[flat_best_indices]
        
        return best_beams

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    print(len(tokenizer))
    args = ModelConfig(
        dim=1024,
        num_layer=18,
    )
    model = NanoLlama(args=args)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'LLM总参数量：{num_params / 1e6:.3f} 百万')

    prompt = "you should tell me substring你好呀，今天吃什么呢？你过得怎么样嘞？"
    text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    print(f"Input text: {text}")

    input_id = tokenizer(text).data['input_ids']
    print("input_ids :", input_id)
    print("dcode_str :", tokenizer.decode(input_id))
    for i in input_id:
        print(f"{i}-{tokenizer.decode(i)}")

    X = LongTensor(input_id[:-1]).unsqueeze(0)
    Y = LongTensor(input_id[1:]).unsqueeze(0)
    print("X shape :", X.shape)
    print("Y shape :", Y.shape)

    output = model(X)
    print(output.shape)