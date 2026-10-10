import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union

from einops import rearrange
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .configuration_vq_codec import ActionCodecConfig

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

class LayerDCT(nn.Module):
    """
    Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param dct_type: which dct function in this file to use
    """
    def __init__(self, config: ActionCodecConfig, dct_type: str = 'dct'):
        super().__init__()
        self.config = config
        self.dct_type = dct_type
        self.N = config.action_horizon
        self.norm = config.dct_norm_type

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

# MARK: AdaRMSNorm-Zero

class AdaRMSNorm(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.norm_eps = config.norm_eps

        self.linear = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _norm(self, x: torch.Tensor):
        # x: [batch_size, seq_len, hidden_size]
        variance = x.float().square().mean(-1, keepdim=True)
        return x.float() * torch.rsqrt(variance + self.norm_eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        modulation = self.linear(cond)
        while modulation.ndim < x.ndim:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)

        dtype = x.dtype
        normalized = self._norm(x)
        normalized = normalized * (1.0 + scale.float()) + shift.float()

        return normalized.to(dtype), gate.to(dtype)

# MARK: Embeddings (embodiment & part)

class EmbodimentEmbeddings(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.embodiment_list = config.embodiment_config
        self.embeddings = nn.Embedding(len(self.embodiment_list), config.hidden_size * config.num_tokens_per_quantizer)

    def forward(self, embodiments: List[str]) -> torch.Tensor:
        ids = torch.tensor([self.embodiment_list.index(k) for k in embodiments], device=self.embeddings.weight.device)
        return self.embeddings(ids).reshape(-1, self.config.num_tokens_per_quantizer, self.config.hidden_size)


class PartEmbeddings(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(len(config.part_config), config.hidden_size)

    def forward(self, part: str) -> torch.Tensor:
        ids = torch.tensor([self.config.part_config.index(part)], device=self.embeddings.weight.device)
        return self.embeddings(ids)

# MARK: Condition Encoder and FiLM

class ConditionEncoder(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.embodiment_cond_dim = config.num_tokens_per_quantizer * config.hidden_size
        self.part_cond_dim = config.hidden_size

        self.cond_mlp = nn.Sequential(
            nn.Linear(self.embodiment_cond_dim + self.part_cond_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    def forward(self, embodiment_cond: torch.Tensor, part_cond: torch.Tensor) -> torch.Tensor:
        embodiment_cond = embodiment_cond.flatten(1)
        cond = torch.cat([embodiment_cond, part_cond], dim=-1)
        return self.cond_mlp(cond)

class ConditionalFiLM(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.cross_attention_dim
        self.linear = nn.Linear(config.hidden_size, config.cross_attention_dim * 2)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return (1 + scale) * x + shift

# MARK: Position Embeddings

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

# MARK: CNN Adapter

class CNNAdapter(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.out_channels = config.cross_attention_dim

        self.conv = nn.Conv2d(1, self.out_channels, kernel_size=(2, 1), stride=(1, 1), bias=False)
        self.film = ConditionalFiLM(config)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (batch_size, action_horizon, action_dim)
        :param cond: Tensor of shape (batch_size, hidden_size), the condition that mixture of embodiment and part embeddings
        :return features: Tensor of shape (batch_size, (action_horizon - 1) * action_dim, cross_attention_dim)
        """
        x = x.unsqueeze(1)  # (bs, 1, action_horizon, action_dim)
        x = self.conv(x)  # (bs, 32, action_horizon -1, action_dim)
        x = x.flatten(2)  # (bs, 32, (action_horizon - 1) * action_dim)
        x = x.transpose(1, 2)  # (bs, (action_horizon - 1) * action_dim, 32)
        x = self.film(x, cond)

        return F.silu(x)


class CNNDecoder(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.cross_attention_dim

        self.film = ConditionalFiLM(config)
        self.conv = nn.ConvTranspose2d(self.in_channels, 1, kernel_size=(2, 1), stride=(1, 1), bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (batch_size, (action_horizon - 1) * max_part_dim, cross_attention_dim).
        :param cond: Tensor of shape (batch_size, hidden_size).
        :return coefficients: Tensor of shape (batch_size, action_horizon, max_part_dim).
        """
        x = self.film(x, cond) # (bs, (action_horizon - 1) * max_part_dim, cross_attention_dim)
        x = F.silu(x) # (bs, (action_horizon - 1) * max_part_dim, cross_attention_dim)
        x = x.transpose(1, 2).reshape(
            x.shape[0],
            self.in_channels,
            self.config.action_horizon - 1,
            self.config.max_part_dim,
        )
        return self.conv(x).squeeze(1) # (bs, action_horizon, max_part_dim)


# MARK: Attention Op

def flash_attn(q, k, v, context_mask=None):
    """Pack valid context tokens; all action queries are valid.

    Inputs use [B, H, L, D]; FlashAttention consumes [total_tokens, H, D].
    context_mask is [B, L_k], with True marking valid keys.
    """
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

# MARK: Attention Block

class Attention(nn.Module):
    def __init__(self, config: ActionCodecConfig):
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
            if not _FLASH_ATTN_AVAILABLE:
                raise ImportError("flash_attention is not available. Please install it to use flash_attention_2.")
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


class FeedForward(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.hidden_size * 3 // 8

        self.up = nn.Linear(self.hidden_size, self.intermediate_size * 2)
        self.down = nn.Linear(self.intermediate_size, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.up(x).chunk(2, dim=-1)
        return self.down(x1 * F.gelu(x2))


class LayerScale(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.full([config.hidden_size], 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

# MARK: Perceiver Transformer Block

class PerceiverTransformerBlock(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config

        self.attn = Attention(config)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.attn_dropout)
        self.layer_scale = LayerScale(config) if config.use_layer_scale else nn.Identity()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.cross_attention_dim, eps=config.norm_eps)
        self.norm3 = AdaRMSNorm(config)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x, context = self.norm1(x), self.norm2(context)
        x = self.attn(x, context, context_mask)
        x = self.dropout(x)
        x = residual + self.layer_scale(x)

        residual = x
        x, gate= self.norm3(x, cond)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + gate * x

        return x

# MARK: Perceiver Codec

class PerceiverEncoder(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            PerceiverTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.base_query = nn.Parameter(torch.randn(1, config.num_tokens_per_quantizer, config.hidden_size))
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.pos_emb_q = SinusoidalPositionalEmbedding(config.hidden_size, max_len=config.num_tokens_per_quantizer)
        self.pos_emb_kv = SinusoidalPositionalEmbedding(config.cross_attention_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        query = x + self.base_query

        pos_emb_q = self.pos_emb_q(query)
        pos_emb_kv = self.pos_emb_kv(context)

        query = query + pos_emb_q
        context = context + pos_emb_kv

        for layer in self.layers:
            query = layer(query, context, context_mask, cond)

        return self.output_proj(query)


class PerceiverDecoder(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        # Reverse the encoder's query/context widths: patches query latents.
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.cross_attention_dim
        decoder_config.cross_attention_dim = config.hidden_size
        self.config = config
        self.reversed_config = decoder_config

        num_patches = (config.action_horizon - 1) * config.max_part_dim
        self.layers = nn.ModuleList([
            PerceiverTransformerBlock(decoder_config) for _ in range(config.num_layers)
        ])
        self.base_query = nn.Parameter(torch.randn(1, num_patches, decoder_config.hidden_size))
        self.cond_proj = nn.Linear(config.hidden_size, decoder_config.hidden_size)
        self.output_proj = nn.Linear(decoder_config.hidden_size, decoder_config.hidden_size)

        self.pos_emb_q = SinusoidalPositionalEmbedding(decoder_config.hidden_size, max_len=num_patches)
        self.pos_emb_kv = SinusoidalPositionalEmbedding(config.hidden_size, max_len=config.num_tokens_per_quantizer)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (batch_size, num_tokens_per_quantizer, hidden_size).
        :param cond: Tensor of shape (batch_size, hidden_size).
        :return features: Tensor of shape (batch_size, (action_horizon - 1) * max_part_dim, cross_attention_dim).
        """
        query_cond = self.cond_proj(cond) # (bs, hidden_size)

        query = self.base_query.expand(x.shape[0], -1, -1)
        query = query + self.pos_emb_q(query)
        context = x + self.pos_emb_kv(x)

        for layer in self.layers:
            query = layer(query, context, None, query_cond)

        return self.output_proj(query)


# MARK: codebook operations

def _sample_vectors(samples: torch.Tensor, n: int) -> torch.Tensor:
    """Sample codebook seeds from the current batch, in fp32."""
    if samples.shape[0] >= n:
        indices = torch.randperm(samples.shape[0], device=samples.device)[:n]
    else:
        indices = torch.randint(samples.shape[0], (n,), device=samples.device)
    return samples[indices].float()


def _kmeans(samples: torch.Tensor, n_clusters: int, n_iters: int = 10):
    """Lloyd's k-means with L2 assignments and retained empty-cluster centers."""
    samples = samples.float()
    means = _sample_vectors(samples, n_clusters)
    for _ in range(n_iters):
        distances = (
            samples.square().sum(1, keepdim=True)
            - 2 * samples @ means.t()
            + means.square().sum(1).unsqueeze(0)
        )
        buckets = distances.argmin(dim=-1)
        counts = torch.bincount(buckets, minlength=n_clusters)
        new_means = torch.zeros_like(means)
        new_means.scatter_add_(0, buckets[:, None].expand_as(samples), samples)
        new_means = new_means / counts.clamp_min(1).unsqueeze(1)
        means = torch.where((counts == 0).unsqueeze(1), means, new_means)

    distances = (
        samples.square().sum(1, keepdim=True)
        - 2 * samples @ means.t()
        + means.square().sum(1).unsqueeze(0)
    )
    counts = torch.bincount(distances.argmin(dim=-1), minlength=n_clusters).float()
    return means, counts

# MARK: EMA VQ Codebook

class EMAVectorQuantize(nn.Module):
    """L2 vector quantizer with projected EMA codebook; input/output: (B, T, D)."""

    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim
        self.decay = config.ema_decay
        self.epsilon = config.ema_eps
        self.threshold_ema_dead = config.threshold_ema_dead

        self.in_proj = nn.Linear(self.input_dim, self.codebook_dim, bias=False)
        self.out_proj = nn.Linear(self.codebook_dim, self.input_dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

        self.register_buffer("codebook", torch.zeros(self.codebook_size, self.codebook_dim))
        self.register_buffer("embed_avg", torch.zeros(self.codebook_size, self.codebook_dim))
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("inited", torch.tensor(False))
        self.register_buffer("_ema_counts", torch.zeros(self.codebook_size), persistent=False)
        self.register_buffer("_ema_embed_sum", torch.zeros(self.codebook_size, self.codebook_dim), persistent=False)
        self._ema_step = 0

    @torch.no_grad()
    def _init_codebook(self, encodings: torch.Tensor):
        distributed = dist.is_initialized()
        if not distributed or dist.get_rank() == 0:
            means, sizes = _kmeans(encodings, self.codebook_size)
        else:
            means = torch.zeros_like(self.codebook)
            sizes = torch.zeros_like(self.cluster_size)
        if distributed:
            dist.broadcast(means, src=0)
            dist.broadcast(sizes, src=0)
        self.codebook.copy_(means)
        self.embed_avg.copy_(means * sizes.unsqueeze(1))
        self.cluster_size.copy_(sizes)
        self.inited.fill_(True)

    @torch.no_grad()
    def _ema_update(self, encodings: torch.Tensor, onehot: torch.Tensor):
        self._ema_counts.add_(onehot.sum(0))
        self._ema_embed_sum.add_(onehot.t() @ encodings)
        self._ema_step += 1
        if self._ema_step < self.config.ema_update_interval:
            return

        # Preserve the per-micro-batch count scale used by the dead-code threshold.
        counts = self._ema_counts / self.config.ema_update_interval
        embed_sum = self._ema_embed_sum / self.config.ema_update_interval
        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
        self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))
        self._ema_counts.zero_()
        self._ema_embed_sum.zero_()
        self._ema_step = 0

    @torch.no_grad()
    def _replace_dead_codes(self, encodings: torch.Tensor):
        if self.threshold_ema_dead <= 0:
            return
        dead = self.cluster_size < self.threshold_ema_dead
        if not dead.any():
            return
        n_dead = int(dead.sum().item())
        distributed = dist.is_initialized()
        if not distributed or dist.get_rank() == 0:
            samples = _sample_vectors(encodings, n_dead)
        else:
            samples = encodings.new_zeros(n_dead, self.codebook_dim)
        if distributed:
            dist.broadcast(samples, src=0)
        self.codebook[dead] = samples.to(self.codebook.dtype)
        self.cluster_size[dead] = self.threshold_ema_dead
        self.embed_avg[dead] = (samples * self.threshold_ema_dead).to(self.embed_avg.dtype)

    def forward(self, z: torch.Tensor):
        """Return quantized (B,T,D), commitment (B,), codes (B,T), and metrics."""
        orig_dtype = z.dtype
        z_e = self.in_proj(z.float())
        encodings = rearrange(z_e, "b t d -> (b t) d").float()
        if not self.inited.item():
            self._init_codebook(encodings.detach())

        distances = (
            encodings.square().sum(1, keepdim=True)
            - 2 * encodings @ self.codebook.float().t()
            + self.codebook.float().square().sum(1).unsqueeze(0)
        )
        codes = rearrange(distances.argmin(dim=-1), "(b t) -> b t", b=z.shape[0])
        z_q_proj = F.embedding(codes, self.codebook.float())
        commit_loss = F.mse_loss(z_e, z_q_proj.detach(), reduction="none").mean((1, 2))

        if self.training and torch.is_grad_enabled():
            onehot = F.one_hot(codes.reshape(-1), self.codebook_size).float()
            self._ema_update(encodings.detach(), onehot)

        # Measure EMA usage before dead-code replacement resets the counts.
        with torch.no_grad():
            counts = self.cluster_size.float()
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            metrics = {
                "perplexity": (-(probs * probs.log()).sum()).exp(),
                "codebook_usage": (counts >= self.threshold_ema_dead).float().mean(),
            }

        if self.training and torch.is_grad_enabled() and self._ema_step == 0:
            self._replace_dead_codes(encodings.detach())

        z_q = (z_q_proj - z_e).detach() + z_e
        z_q = self.out_proj(z_q)
        return z_q.to(orig_dtype), commit_loss, codes, metrics

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Map (B,T) indices back to quantized latents (B,T,D)."""
        return self.out_proj(F.embedding(codes, self.codebook.float()))

# MARK: RVQ

class ResidualVectorQuantize(nn.Module):
    """Independent EMA codebooks with residual refinement and quantizer dropout."""

    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.n_codebooks = config.num_quantizers
        self.quantizer_dropout = config.quantizer_dropout
        self.quantizers = nn.ModuleList([
            EMAVectorQuantize(config)
            for _ in range(self.n_codebooks)
        ])

    def forward(
        self,
        z: torch.Tensor,
        return_level_data: bool = False,
    ):
        """Return z_q (B,T,D), codes (B,K,T), and scalar loss/codebook metrics.

        With return_level_data, also return each level's codebook-space
        consistency residual (B,T,codebook_dim) and codes. The consistency chain detaches earlier
        quantized outputs; the main residual chain retains their gradients.
        """
        B = z.shape[0]
        if self.training:
            n_q_per_sample = torch.full((B,), self.n_codebooks, device=z.device)
            dropout_mask = torch.rand(B, device=z.device) < self.quantizer_dropout
            rand_n = torch.randint(1, self.n_codebooks + 1, (B,), device=z.device)
            n_q_per_sample[dropout_mask] = rand_n[dropout_mask]
        else:
            n_q_per_sample = torch.full((B,), self.n_codebooks, device=z.device)

        z_q = torch.zeros_like(z)
        residual = z
        all_codes = []
        total_commit_loss = torch.tensor(0.0, device=z.device)
        metrics = {}
        if return_level_data:
            consist_residual = z
            consist_residuals = []
            level_codes = []

        for i, quantizer in enumerate(self.quantizers):
            active = (i < n_q_per_sample).float()
            if return_level_data:
                consist_residuals.append(quantizer.in_proj(consist_residual))

            z_q_i, commit_i, codes_i, quantizer_metrics = quantizer(residual)
            z_q = z_q + z_q_i * active[:, None, None]
            residual = residual - z_q_i
            total_commit_loss = total_commit_loss + (commit_i * active).mean()
            all_codes.append(codes_i)

            metrics.update({f"{key}/quantizer_{i}": value for key, value in quantizer_metrics.items()})

            if return_level_data:
                consist_residual = consist_residual - z_q_i.detach()
                level_codes.append(codes_i)

        codes = torch.stack(all_codes, dim=1)
        metrics["loss/commit_loss"] = total_commit_loss
        if return_level_data:
            return z_q, codes, metrics, consist_residuals, level_codes
        return z_q, codes, metrics

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum decoded codebook vectors from (B,K,T) indices into (B,T,D)."""
        bs, num_quantizers, seq_len = codes.shape
        z_q = torch.zeros(
            bs, seq_len, self.quantizers[0].input_dim,
            device=codes.device, dtype=torch.float32,
        )
        for i in range(num_quantizers):
            z_q = z_q + self.quantizers[i].decode_codes(codes[:, i])
        return z_q
