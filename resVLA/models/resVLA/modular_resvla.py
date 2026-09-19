import math
from typing import Dict, List, Literal, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_varlen_func = None
    _FLASH_ATTN_AVAILABLE = False
    

# MARK: DCT Block

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * torch.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= torch.sqrt(torch.tensor(N)) * 2
        V[:, 1:] /= torch.sqrt(torch.tensor(N / 2)) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= torch.sqrt(torch.tensor(N)) * 2
        X_v[:, 1:] *= torch.sqrt(torch.tensor(N / 2)) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * torch.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

class BlockDCT(nn.Module):
    """
    Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param dct_type: which dct function in this file to use
    """
    def __init__(self, in_features: int = 16, dct_type: str = 'dct', norm=None):
        super().__init__()
        self.dct_type = dct_type
        self.N = in_features
        self.norm = norm

        mat = self.compute_matrix()
        self.mat = nn.Buffer(mat, persistent=False)

    def compute_matrix(self):
        # initialise using dct function
        I = torch.eye(self.N, dtype=torch.float32)
        if self.dct_type == 'dct':
            mat = dct(I, norm=self.norm)
        elif self.dct_type == 'idct':
            mat = idct(I, norm=self.norm)
        return mat

    def forward(self, x):
        return x @ self.mat

# MARK: Learnable Cutoff Pojector

class LearnableCutoffProjector(nn.Module):
    """
    Projects the learnable cutoff token to a mask over frequency components. The mask has the same number of components as the action horizon, and is used to mask out high-frequency components in the DCT domain, so that the model can focus on low-frequency components for better generalization.
    """
    def __init__(self, num_freq_components: int = 4, hidden_dim: int = 2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_freq_components)
        )

    def forward(self, x, mode="train"):
        """
        Args:
            x: [B, 1, hidden_dim]
        Returns:
            mask: [B, num_freq_components], where the first N components are 1 and the rest are 0, and N is determined by the learnable cutoff token.
        """
        logits = self.projector(x) # (B, 1, num_freq_components)
        if mode == "train":
            one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1) # (B, 1, num_freq_components)
        else:
            one_hot = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).float() # (B, 1, num_freq_components)
        mask = torch.flip(torch.cumsum(torch.flip(one_hot, dims=[-1]), dim=-1), dims=[-1]) # (B, 1, num_freq_components)
        return mask.squeeze(1) # (B, num_freq_components)

# MARK: Anchor Decoder

class AnchorMLP(nn.Module):
    """
    Generate coarse frequency-domain action as anchor
    """
    def __init__(self, action_horizon: int = 16, action_dim: int = 32, hidden_dim: int = 2048):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.mlp = nn.Sequential(
            nn.Flatten(), # (bs, action_horizon, hidden_dim) -> (bs, action_horizon * hidden_dim)
            nn.Linear(action_horizon * hidden_dim, hidden_dim), # (bs, action_horizon * hidden_dim) -> (bs, hidden_dim)
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_horizon * action_dim) # (bs, hidden_dim) -> (bs, action_horizon * action_dim)
        )
    
    def forward(self, x):
        return self.mlp(x).view(-1, self.action_horizon, self.action_dim) # (bs, action_horizon, action_dim)

# MARK: AdaRMS-Zero

class AdaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_eps = norm_eps
        self.linear = nn.Linear(hidden_dim, 3 * hidden_dim)
        # Match AdaLN-Zero's initialization so norm ablations share dynamics.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(variance + self.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected embedding dimension {self.hidden_dim}, got {x.shape[-1]}."
            )
        if cond.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected condition dimension {self.hidden_dim}, got {cond.shape[-1]}."
            )

        modulation = self.linear(cond)
        while modulation.ndim < x.ndim:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)

        dtype = x.dtype
        normalized = self._norm(x)
        normalized = normalized * (1.0 + scale.float()) + shift.float()
        return normalized.to(dtype), gate.to(dtype)

# MARK: Action State Coder

class ActionStateEncoder(nn.Module):
    """
    Encode the action or state of the robot into a latent representation
    """
    def __init__(self, input_dim: int = 2048, output_dim: int = 2048, norm_eps: float = 1e-5):
        super().__init__()
        
        self.W1 = nn.Linear(input_dim, 2 * output_dim) # d -> 2w
        self.W2 = nn.Linear(2 * output_dim, output_dim) # 2w -> w

        self.ada_norm = AdaRMSNorm(
            hidden_dim=output_dim,
            norm_eps=norm_eps,
        )

        self.act_fn = nn.SiLU()

    def forward(self, actions, cond):
        """
        Args:
            actions: [B, T, action_dim]
            cond: [B, hidden_dim] -- embodiment condition
        Returns:
            [B, T, hidden_dim]
        """
        a_emb = self.W2(self.act_fn(self.W1(actions))) # (bs, horizon, action_dim) -> (bs, horizon, 2*hidden_dim) -> (bs, horizon, hidden_dim)

        # The zero-initialized adaptive gate starts this branch as the plain MLP
        # embedding and gradually introduces embodiment/time conditioning.
        normalized, gate = self.ada_norm(a_emb, cond)
        a_emb = a_emb + gate * normalized

        return a_emb


class ActionStateDecoder(nn.Module):
    """
    Decode to action or state from the latent representation
    """
    def __init__(self, input_dim: int = 2048, output_dim: int = 2048, norm_eps: float = 1e-5):
        super().__init__()
        
        self.W1 = nn.Linear(input_dim, 2 * input_dim) # w -> 2w
        self.W2 = nn.Linear(2 * input_dim, output_dim) # 2w -> d

        self.ada_norm = AdaRMSNorm(
            hidden_dim=input_dim,
            norm_eps=norm_eps,
        )

        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, cond):
        """
        Args:
            hidden_states: [B, T, hidden_dim]
            cond: [B, hidden_dim] -- embodiment condition
        Returns:
            [B, T, action_dim]
        """
        normalized, gate = self.ada_norm(hidden_states, cond)
        h = hidden_states + gate * normalized
        h = self.W2(self.act_fn(self.W1(h))) # (bs, horizon, hidden_dim) -> (bs, horizon, 2*hidden_dim) -> (bs, horizon, action_dim)

        return h

# MARK: Diffusion Timestep Encoder

class TimestepEncoder(nn.Module):
    """
    Encode the timestep of the diffusion process into a latent representation
    """
    def __init__(self, hidden_dim: int = 2048, max_period: int = 10000, downscale_freq_shift: int = 1):
        super().__init__()
        
        self.linear1 = nn.Linear(256, 2 * hidden_dim)
        self.linear2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.act_fn = nn.SiLU()
        self.max_period = max_period
        self.downscale_freq_shift = downscale_freq_shift

        exponent = self._init_exponent()
        self.freqs = nn.Buffer(exponent, persistent=False)

    def _init_exponent(self):
        return torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=128) / (128 - self.downscale_freq_shift))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B, 1] or [B, 1, 1] -- timestep of the diffusion process
        Returns:
            [B, hidden_dim]
        """
        if timesteps.ndim == 3:
            timesteps = timesteps.squeeze(-1) # (bs, 1)
        emb = self.freqs * timesteps
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        emb = self.linear2(self.act_fn(self.linear1(emb))) # (bs, 256) -> (bs, 2*hidden_dim) -> (bs, hidden_dim)
        return emb


class EmbodimentEncoder(nn.Module):
    """
    Encode the embodiment condition into a latent representation
    """
    def __init__(self, input_dim: int = 2048, output_dim: int = 2048):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, 2 * output_dim) # d -> 2w
        self.linear2 = nn.Linear(2 * output_dim, output_dim) # 2w -> w
        self.act_fn = nn.SiLU()

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond: [B, hidden_dim] -- embodiment condition
        Returns:
            [B, hidden_dim]
        """
        return self.linear2(self.act_fn(self.linear1(cond))) # (bs, hidden_dim) -> (bs, 2*hidden_dim) -> (bs, hidden_dim)

# MARK: Embodiment Embeddings

class EmbodimentEmbeddings(nn.Module):
    def __init__(self, embodiment_config: List[str], hidden_dim: int = 2048):
        super().__init__()

        self.embodiment_config = embodiment_config

        self.embeddings = nn.Embedding(len(embodiment_config), hidden_dim)

    def forward(self, embodiments: List[str]) -> torch.Tensor:
        """
        Args:
            embodiments: a list of embodiment names
        Returns:
            [B, hidden_dim]
        """
        ids = torch.tensor([self.embodiment_config.index(k) for k in embodiments], device=self.embeddings.weight.device)
        return self.embeddings(ids)
        

# MARK: Position Embeddings

class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding (RoPE).  Only the essentials: no xpos, no natten.

    Partial rotary: applies rotation to the first ``dim`` channels of each head;
    the rest pass through unchanged.  ``dim`` defaults to ``dim_heads // 2``
    (capped at 32) which matches FasterV2's convention.
    """

    def __init__(self, dim: int = 2048, base: int = 10000):
        super().__init__()
        self.config = dict(dim=dim, base=base)
        inv_freq, _ = self.compute_default_rope_parameters(self.config)
        self.rope_type = "default"
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def compute_default_rope_parameters(config, device=None, **kwargs):
        base = config["base"]
        dim = config["dim"]

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return inv_freq.to(device), 1.0

    @torch.no_grad()
    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        key = (seq_len, device, dtype)
        if key not in self._cache:
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
            self._cache[key] = (emb.cos().to(dtype), emb.sin().to(dtype))
        return self._cache[key]

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        return self.get_cos_sin(seq_len, device, dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to the first ``cos.shape[-1]`` dims of q and k."""
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

# MARK: Basic Transformer Blocks

def flash_attn(q, k, v, context_mask=None):
    """Pack valid context tokens; all action queries are valid.

    Inputs use [B, H, L, D]; FlashAttention consumes [total_tokens, H, D].
    context_mask is [B, L_k], with True marking valid keys.
    """
    # FlashAttention only accepts FP16/BF16. Keep an already-supported dtype;
    # otherwise use the BF16 policy configured for this training run.
    flash_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    q = q.to(dtype=flash_dtype)
    k = k.to(dtype=flash_dtype)
    v = v.to(dtype=flash_dtype)

    batch_size, _, query_len, _ = q.shape
    key_len = k.shape[2]
    q = rearrange(q, "b h l d -> (b l) h d").contiguous()
    k = rearrange(k, "b h l d -> (b l) h d").contiguous()
    v = rearrange(v, "b h l d -> (b l) h d").contiguous()
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=q.device, dtype=torch.int32
    ) * query_len

    if context_mask is None:
        cu_seqlens_k = torch.arange(
            batch_size + 1, device=k.device, dtype=torch.int32
        ) * key_len
        max_seqlen_k = key_len
    else:
        indices = context_mask.reshape(-1).nonzero(as_tuple=True)[0]
        k = k.index_select(0, indices)
        v = v.index_select(0, indices)
        lengths = context_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens_k = F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
        max_seqlen_k = int(lengths.max().item())

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        query_len,
        max_seqlen_k,
        dropout_p=0.0,
        causal=False,
    )
    return rearrange(out, "(b l) h d -> b l (h d)", b=batch_size)


def sdpa_attn(q, k, v, context_mask=None):
    attn_mask = context_mask[:, None, None, :] if context_mask is not None else None
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True
    )
    return rearrange(out, "b h l d -> b l (h d)")


def eager_attn(q, k, v, context_mask=None):
    head = q.shape[1]
    kv_head = k.shape[1]

    if head != kv_head:
        repeat = head // kv_head
        k, v = map(lambda t: t.repeat_interleave(repeat, dim=1), (k, v))

    w = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if context_mask is not None:
        w = w.masked_fill(~context_mask[:, None, None, :], float("-inf"))
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(v.dtype)
    out = torch.matmul(w, v)
    return rearrange(out, "b h l d -> b l (h d)")
    

class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        rope_base: int = 10000,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads, got hidden_dim={hidden_dim}, num_head={num_heads}")

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else hidden_dim
        
        self.dim = hidden_dim // num_heads
        self.num_head, self.num_kv_heads = num_heads, num_kv_heads
        self.cross_attention_dim = cross_attention_dim

        self.q = nn.Linear(hidden_dim, self.dim * num_heads, bias=attention_bias)
        self.k = nn.Linear(cross_attention_dim, self.dim * num_kv_heads, bias=attention_bias)
        self.v = nn.Linear(cross_attention_dim, self.dim * num_kv_heads, bias=attention_bias)
        self.o = nn.Linear(self.dim * num_heads, hidden_dim, bias=attention_bias)

        self.q_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(self.dim, elementwise_affine=False)

        self.rope = RotaryEmbedding(self.dim // 2, rope_base)

        if attn_implementation == "flash_attention_2":
            if not _FLASH_ATTN_AVAILABLE:
                raise ImportError("flash_attn is not available. Please install it to use flash attention.")
            self.attn_func = flash_attn
        elif attn_implementation == "sdpa":
            self.attn_func = sdpa_attn
        elif attn_implementation == "eager":
            self.attn_func = eager_attn
        else:
            raise ValueError(f"attn_implementation must be 'flash_attention_2', 'sdpa', or 'eager', got {attn_implementation}")

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input tensor
            context: (B, T, D) optional context tensor for cross-attention.
            context_mask: (B, T_context) bool mask; True marks a valid key.
        """
        context = x if context is None else context

        q = self.q(x)
        k, v = self.k(context), self.v(context)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_head)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_kv_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_kv_heads)

        q, k = self.q_norm(q), self.k_norm(k)
        q_cos, q_sin = self.rope(q.shape[2], q.device, q.dtype)
        k_cos, k_sin = self.rope(k.shape[2], k.device, k.dtype)
        q_cos, q_sin = map(lambda t: rearrange(t, "t d -> 1 1 t d"), (q_cos, q_sin))
        if context_mask is None:
            k_cos, k_sin = map(lambda t: rearrange(t, "t d -> 1 1 t d"), (k_cos, k_sin))
        else:
            # Number valid keys consecutively, including the prepended state.
            positions = (context_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
            k_cos, k_sin = (table[positions].unsqueeze(1) for table in (k_cos, k_sin))
        q, _ = apply_rotary_pos_emb(q, q, q_cos, q_sin)
        k, _ = apply_rotary_pos_emb(k, k, k_cos, k_sin)

        out = self.attn_func(q, k, v, context_mask=context_mask)
        out = self.o(out)

        return out
    

class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()

        self.dim = hidden_dim
        self.inner_dim = hidden_dim * 3 // 8

        self.up = nn.Linear(hidden_dim, self.inner_dim * 2)
        self.down = nn.Linear(self.inner_dim, hidden_dim)
        nn.init.zeros_(self.down.weight)  # zero-init output proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.up(x).chunk(2, dim=-1)
        return self.down(x1 * F.gelu(x2))


# MARK: DiT

class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        rope_base: int = 10000,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn = Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            rope_base=rope_base,
            attn_implementation=attn_implementation
        )

        self.dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(hidden_dim=hidden_dim)

        self.norm1 = AdaRMSNorm(hidden_dim=hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = AdaRMSNorm(hidden_dim=hidden_dim)


    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x, gate = self.norm1(x, temb)
        if context is None:
            x = self.attn(x, context_mask=context_mask)
        else:
            x = self.attn(x, self.norm2(context), context_mask=context_mask)
        x = self.dropout(x)
        x = residual + gate * x

        residual = x
        x, gate = self.norm3(x, temb)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + gate * x

        return x

class DiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        rope_base: int = 10000,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
        dropout: float = 0.0,
        num_layers: int = 28,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                cross_attention_dim=cross_attention_dim,
                attention_bias=attention_bias,
                rope_base=rope_base,
                attn_implementation=attn_implementation,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.shift_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[List[torch.Tensor]] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            # cross-attention
            x = layer(
                x, temb, context[idx] if context is not None else None,
                context_mask=context_mask,
            )

        shift, scale = self.shift_proj(F.silu(temb)).chunk(2, dim=-1)
        out = self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.out_proj(out)

        return out

# MARK: Flow Refiner

class FlowRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        flow_dim: Optional[int] = None,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        rope_base: int = 10000,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
        dropout: float = 0.0,
        num_layers: int = 9,
    ):
        super().__init__()

        flow_dim = flow_dim if flow_dim is not None else hidden_dim
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else hidden_dim

        self.encoder = ActionStateEncoder(input_dim=flow_dim, output_dim=hidden_dim)
        self.decoder = ActionStateDecoder(input_dim=hidden_dim, output_dim=flow_dim)
        self.dit = DiT(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            rope_base=rope_base,
            attn_implementation=attn_implementation,
            dropout=dropout,
            num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        embodiment_cond: torch.Tensor,
        context: Optional[List[torch.Tensor]] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, flow_dim] -- input action/state sequence
            temb: [B, hidden_dim] -- timestep embedding
            embodiment_cond: [B, hidden_dim] -- embodiment condition embedding
            context: List of [B, T, hidden_dim] -- context embeddings for cross-attention
            context_mask: [B, T] bool mask; True marks a valid context token
        Returns:
            [B, T, flow_dim] -- predicted velocity (delta) of the input sequence
        """
        hidden_stats = self.encoder(x, embodiment_cond) # (bs, action_horizon, hidden_dim)
        output = self.dit(hidden_stats, temb, context, context_mask=context_mask) # (bs, action_horizon, hidden_dim)
        velocity = self.decoder(output, embodiment_cond) # (bs, action_horizon, flow_dim)
        return velocity
