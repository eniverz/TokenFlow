# OAT

Whole-action OAT tokenizer trained from scratch. The implementation follows the
register-query encoder, FSQ, and prefix-trained single-pass decoder in Praxis.
It does not load Praxis checkpoints.

## Input and architecture

`forward(samples)` reads normalized `samples["actions"]` shaped `[B,H,32]`,
`samples["action_mask"]` shaped `[B,32]`, and `samples["embodiment"]`, a list
of `B` embodiment names. The default LIBERO configuration uses `H=16`.
After normalization, the existing `PadStateAction` dataset transform zero-pads
actions and states to 32 dimensions and creates their boolean validity masks.
LIBERO has 7 valid action dimensions and 8 valid state dimensions. The model
does not consume states. Both reconstruction losses exclude padded dimensions
and average over valid action elements.

The encoder projects actions to hidden embeddings and cross-attends to them
with embodiment-specific register queries. Register self-attention is causal. FSQ quantizes each
register's latent channels into one integer token. The default `[8,8,6,5]`
levels produce a vocabulary of 1920 tokens and 16 tokens per action chunk.

The decoder projects latents into hidden embeddings, replaces a suffix with
a learned mask embedding, and adds latent positions. Action position queries
cross-attend to these embeddings without action self-attention. Both encoder
and decoder use fixed sinusoidal positions and gated MLPs.

Two independent embedding tables map embodiment names to `[B,L,hidden_size]`
encoder conditions and `[B,H,hidden_size]` decoder conditions. Each condition
is used as the initial query (plus positional embeddings) and passed unchanged
to each block's AdaRMSNorm. Query self-attention, cross-attention, and FFN inputs
are modulated with `(1 + scale) * RMSNorm(x) + shift`. There is no residual gate.
Context normalization remains ordinary RMSNorm. Modulation projections
and encoder/decoder output heads are initialized to zero.

`attn_implementation` selects `sdpa` (default), `eager`, or `flash_attention_2`.
The three attention functions, `Attention` class, and
`SinusoidalPositionalEmbedding` class are copied verbatim from vq_actioncodec_v3.
This includes GQA through `num_kv_heads`, `attention_bias`, the absence of QK
normalization, and position embeddings scaled by `0.1` with nonpersistent
buffers. The attention functions themselves are noncausal.

To preserve encoder causality without modifying those definitions, the encoder
block expands each query into a separate batch item and supplies the lower
triangular rows as `context_mask`. This repeats key/value inputs and increases
computation and memory use. Cross-attention passes `context_mask=None`.
FlashAttention uses the original variable-length kernel and key packing, with
the same FP32-to-BF16 conversion and output cast. It requires FlashAttention and
a supported GPU. The original import handling and errors are retained.

`num_tokens` is configured as a power of two. During training, each full-length
decoder call independently samples a prefix length uniformly from powers of
two up to `num_tokens`. A shorter input is padded with mask embeddings and
decoded directly. Prefix decoding always reconstructs all `H` action steps.

## Training

Two branches share the decoder:

- Quantized: encoder → FSQ → decoder, with end-to-end reconstruction gradients.
- Continuous: detached encoder latents → training noise → decoder. Noise has
  a per-sample standard deviation sampled uniformly from
  `[0, continuous_noise_std]`, default `[0,0.03]`.

Both branches train decoder embodiment embeddings. The continuous branch does
not backpropagate into the encoder or its embodiment embeddings.

Both branches use prefix dropout during training. Evaluation disables random
prefix sampling, ordinary dropout, and continuous noise.

The returned dictionary contains scalar tensors `loss`,
`loss/quantized_recons_loss`, and `loss/continuous_recons_loss`. The total is
`1.5 * quantized_mse + 1.0 * continuous_mse` by default. There is no commitment
or consistency loss. Only `loss` retains its gradient in the metrics dictionary.

From the repository root, with the repository dependencies installed:

```bash
python -m resVLA.scripts.train task=oat_stage1 \
  dataset_root=/path/to/libero \
  norm_stats_dir=norm_stats/tokenflow/libero
```

The task uses the existing Trainer and an empty weight-loader list. Checkpoints
are saved through `PreTrainedModel.save_pretrained`.

## Encode and decode

```python
from resVLA.models.tokenizer.oat import OAT

model = OAT.from_pretrained(checkpoint_path).eval()
embodiments = ["franka"] * actions.shape[0]
tokens = model.encode(actions, embodiments)       # [B,16], IDs in [0,1919]
reconstructed = model.decode(tokens, embodiments) # [B,H,A], normalized
coarse = model.decode(tokens[:, :4], embodiments) # Same shape, shorter prefix
```

`encode` expects actions already normalized and padded to 32 dimensions;
`decode` returns all 32 dimensions. Slice the valid dimensions for the target
embodiment before unnormalization.

`encode` and `decode` run without gradients. For differentiable continuous
decoding, call `model.decoder(latents, model.decoder_embeddings(embodiments))`.
`model.encoder(actions, model.encoder_embeddings(embodiments))` returns
unquantized latents `[B,L,4]`. Call `eval()` for deterministic inference.

This codec does not implement the part/RVQ interface expected by the current
TokenFlow Stage 2 wrapper. Stage 2 token templates and codec calls need a
separate integration.
