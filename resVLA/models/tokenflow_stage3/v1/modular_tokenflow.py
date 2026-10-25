from typing import Dict, Literal, Optional, Tuple

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


class RotaryEmbedding(nn.Module):
    """Partial rotary embeddings, applied to the first half of each attention head."""

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

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
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

        if attn_implementation == "flash_attention_2" and not _FLASH_ATTN_AVAILABLE:
            raise ImportError("flash_attn is not available. Please install it to use flash attention.")
        self.attn_func = {
            "flash_attention_2": flash_attn, "sdpa": sdpa_attn, "eager": eager_attn,
        }[attn_implementation]

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
            # Number valid context keys consecutively, ignoring left padding.
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


class RefinerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.refiner_hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.self_attn = Attention(
            hidden_dim=hidden_size,
            num_heads=config.refiner_num_heads,
            num_kv_heads=config.refiner_num_kv_heads,
            attention_bias=config.attention_bias,
            rope_base=config.rope_base,
            attn_implementation=config.attn_implementation,
        )
        self.cross_attn = Attention(
            hidden_dim=hidden_size,
            num_heads=config.refiner_num_heads,
            num_kv_heads=config.refiner_num_kv_heads,
            cross_attention_dim=config.stage2_config.hidden_size,
            attention_bias=config.attention_bias,
            rope_base=config.rope_base,
            attn_implementation=config.attn_implementation,
        )
        self.ffn = FeedForward(hidden_dim=hidden_size)

    def forward(self, x, context, context_mask):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context, context_mask=context_mask)
        return x + self.ffn(self.norm3(x))


class FlowRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.refiner_hidden_size
        self.up_proj = nn.Linear(2 * config.action_dim, hidden_size)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList([RefinerBlock(config) for _ in range(config.num_refiner_layers)])
        self.down_proj = nn.Linear(hidden_size, config.action_dim)

    def forward(self, actions, time, coarse_action, contexts, context_mask):
        x = self.up_proj(torch.cat([actions, coarse_action], dim=-1).to(self.up_proj.weight.dtype))
        x = x + self.time_embedding(time.to(self.up_proj.weight.dtype))
        for i, block in enumerate(self.blocks):
            x = block(x, contexts[i].to(x.dtype), context_mask)
        return self.down_proj(x)
