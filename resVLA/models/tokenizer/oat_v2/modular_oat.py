import math

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


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        pos_emb = self._init_pos_emb()
        self.pos_emb = nn.Buffer(pos_emb, persistent=False)

    def _init_pos_emb(self):
        max_len = self.max_len
        dim = self.dim
        position = torch.arange(max_len, device="cpu").unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device="cpu", dtype=torch.float32) * -(math.log(10000.0) / dim)
        )

        pos_emb = torch.zeros(max_len, dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        return pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape[0], x.shape[1]
        device = x.device
        dtype = x.dtype
        pos_emb = self.pos_emb[: L, :].to(device=device, dtype=dtype)
        return pos_emb.unsqueeze(0).expand(B, -1, -1) * 0.1


class RMSNorm(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class AdaRMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.norm_eps)
        self.linear = nn.Linear(config.hidden_size, 2 * config.hidden_size)

    def forward(self, x, cond):
        scale, shift = self.linear(cond).float().chunk(2, dim=-1)
        normalized = self.norm(x.float())
        return ((1 + scale) * normalized + shift).to(x.dtype)


class EmbodimentEmbeddings(nn.Module):
    def __init__(self, config, query_length):
        super().__init__()
        self.embodiment_list = config.embodiment_config
        self.query_length = query_length
        self.hidden_size = config.hidden_size
        self.embeddings = nn.Embedding(len(self.embodiment_list), query_length * config.hidden_size)

    def forward(self, embodiments):
        ids = torch.tensor([self.embodiment_list.index(name) for name in embodiments], device=self.embeddings.weight.device)
        return self.embeddings(ids).reshape(-1, self.query_length, self.hidden_size)


def flash_attn(q, k, v, context_mask=None):
    """Pack valid context tokens; all action queries are valid.

    Inputs use [B, H, L, D]; FlashAttention consumes [total_tokens, H, D].
    context_mask is [B, L_k], with True marking valid keys.
    """
    if not _FLASH_ATTN_AVAILABLE:
        raise ImportError("flash_attention_2 requires flash_attn.")
    # FlashAttention only accepts FP16/BF16. Keep an already-supported dtype;
    # otherwise use the BF16 policy configured for this training run.
    original_dtype = q.dtype
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
    out = rearrange(out, "(b l) h d -> b l (h d)", b=batch_size)
    return out.to(dtype=original_dtype)


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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size // config.num_heads
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.cross_attention_dim = config.cross_attention_dim

        self.q = nn.Linear(config.hidden_size, self.dim * self.num_heads, bias=config.attention_bias)
        self.k = nn.Linear(config.cross_attention_dim, self.dim * self.num_kv_heads, bias=config.attention_bias)
        self.v = nn.Linear(config.cross_attention_dim, self.dim * self.num_kv_heads, bias=config.attention_bias)
        self.o = nn.Linear(self.dim * self.num_heads, config.hidden_size, bias=config.attention_bias)

        if config.attn_implementation == "flash_attention_2":
            self.attn_fn = flash_attn
        elif config.attn_implementation == "sdpa":
            self.attn_fn = sdpa_attn
        elif config.attn_implementation == "eager":
            self.attn_fn = eager_attn
        else:
            raise ValueError(f"Unknown attention implementation: {config.attn_implementation}")

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ):
        q, k, v = self.q(x), self.k(context), self.v(context)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_kv_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_kv_heads)

        out = self.attn_fn(q, k, v, context_mask=context_mask)
        out = self.o(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config, self_attention):
        super().__init__()
        self.self_attention = self_attention
        if self_attention:
            self.self_attn = Attention(config)
            self.self_norm = AdaRMSNorm(config)
        self.cross_attn = Attention(config)
        self.query_norm = AdaRMSNorm(config)
        self.context_norm = RMSNorm(config.norm_eps)
        self.ffn_norm = AdaRMSNorm(config)
        self.gate_proj = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=False)
        self.down_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, context, cond):
        if self.self_attention:
            normalized = self.self_norm(x, cond)
            B, L, D = normalized.shape
            # Attention accepts a per-sample key mask, so each query becomes a sample.
            queries = normalized.reshape(B * L, 1, D)
            keys = normalized.unsqueeze(1).expand(B, L, L, D).reshape(B * L, L, D)
            context_mask = torch.ones(L, L, dtype=torch.bool, device=x.device).tril().repeat(B, 1)
            x = x + self.self_attn(queries, keys, context_mask).reshape(B, L, D)
        x = x + self.cross_attn(self.query_norm(x, cond), self.context_norm(context), None)
        normalized = self.ffn_norm(x, cond)
        hidden = self.dropout(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return x + self.dropout(self.down_proj(hidden))


class RegisterQueryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_proj = nn.Linear(config.action_dim, config.hidden_size)
        self.action_pos = SinusoidalPositionalEmbedding(config.hidden_size, config.action_horizon)
        self.register_pos = SinusoidalPositionalEmbedding(config.hidden_size, config.num_tokens)
        self.blocks = nn.ModuleList([TransformerBlock(config, True) for _ in range(config.encoder_depth)])
        self.norm = RMSNorm(config.norm_eps)
        self.head = nn.Linear(config.hidden_size, config.latent_dim)

    def forward(self, actions, cond):
        context = self.action_proj(actions)
        context = context + self.action_pos(context)
        x = cond + self.register_pos(cond)
        for block in self.blocks:
            x = block(x, context, cond)
        return self.head(self.norm(x))


class SinglePassDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tokens = config.num_tokens
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.mask_token = nn.Parameter(torch.empty(config.hidden_size))
        self.latent_pos = SinusoidalPositionalEmbedding(config.hidden_size, config.num_tokens)
        self.action_pos = SinusoidalPositionalEmbedding(config.hidden_size, config.action_horizon)
        self.blocks = nn.ModuleList([TransformerBlock(config, False) for _ in range(config.decoder_depth)])
        self.norm = RMSNorm(config.norm_eps)
        self.head = nn.Linear(config.hidden_size, config.action_dim)

    def forward(self, latents, cond):
        context = self.latent_proj(latents)
        B, L, D = context.shape
        mask_token = self.mask_token.to(context.dtype).view(1, 1, D)
        if self.training and L == self.num_tokens:
            keep_k = 2 ** torch.randint(self.num_tokens.bit_length(), (B, 1), device=context.device)
            mask = torch.arange(L, device=context.device).unsqueeze(0) >= keep_k
            context = torch.where(mask.unsqueeze(-1), mask_token, context)
        context = torch.cat([context, mask_token.expand(B, self.num_tokens - L, D)], dim=1)
        context = context + self.latent_pos(context)
        x = cond + self.action_pos(cond)
        for block in self.blocks:
            x = block(x, context, cond)
        return self.head(self.norm(x))


class FiniteScalarQuantization(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("basis", torch.tensor([1, *levels[:-1]], dtype=torch.long).cumprod(0))

    def forward(self, latents):
        z = latents.float()
        half_width = (self.levels - 1) * 1.001 / 2
        offset = (self.levels % 2 == 0).float() * 0.5
        bounded = torch.tanh(z + torch.atanh(offset / half_width)) * half_width - offset
        rounded = bounded + (bounded.round() - bounded).detach()
        scale = self.levels // 2
        quantized = rounded / scale
        tokens = ((rounded.detach().round().long() + scale) * self.basis).sum(-1)
        return quantized, tokens

    def from_codes(self, tokens):
        values = (tokens.unsqueeze(-1) // self.basis) % self.levels
        scale = self.levels // 2
        return (values - scale).float() / scale
