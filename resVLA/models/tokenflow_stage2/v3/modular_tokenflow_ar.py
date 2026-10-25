import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _ce_tile_stats_kernel(
        logits_ptr, labels_ptr, maxima_ptr, sums_ptr, argmax_ptr, targets_ptr,
        row_stride, label_stride,
        VOCAB_SIZE: tl.constexpr, NUM_TILES: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            logits_ptr + row * row_stride + cols,
            mask=cols < VOCAB_SIZE, other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.max(values, axis=0)
        exp_sum = tl.sum(tl.exp(values - maximum), axis=0)
        # torch.argmax chooses the first index when maxima are tied.
        argmax = tl.min(
            tl.where((cols < VOCAB_SIZE) & (values == maximum), cols, VOCAB_SIZE),
            axis=0,
        )
        offset = row * NUM_TILES + tile
        tl.store(maxima_ptr + offset, maximum)
        tl.store(sums_ptr + offset, exp_sum)
        tl.store(argmax_ptr + offset, argmax)

        if tile == 0:
            target = tl.load(labels_ptr + row * label_stride)
            tl.device_assert((target >= 0) & (target < VOCAB_SIZE), "invalid CE target")
            # Capture targets before the next kernel overwrites logits with gradients.
            target_value = tl.load(
                logits_ptr + row * row_stride + target,
                mask=(target >= 0) & (target < VOCAB_SIZE), other=float("nan"),
            ).to(tl.float32)
            tl.store(targets_ptr + row, target_value)


    @triton.jit
    def _ce_loss_grad_kernel(
        logits_ptr, labels_ptr, maxima_ptr, sums_ptr, argmax_ptr, targets_ptr,
        losses_ptr, correct_ptr, row_stride, label_stride, num_tokens,
        VOCAB_SIZE: tl.constexpr, NUM_TILES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, STATS_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        tiles = tl.arange(0, STATS_SIZE)
        offsets = row * NUM_TILES + tiles
        maxima = tl.load(maxima_ptr + offsets, mask=tiles < NUM_TILES, other=-float("inf"))
        sums = tl.load(sums_ptr + offsets, mask=tiles < NUM_TILES, other=0.0)
        maximum = tl.max(maxima, axis=0)
        exp_sum = tl.sum(sums * tl.exp(maxima - maximum), axis=0)
        target = tl.load(labels_ptr + row * label_stride)

        if tile == 0:
            target_value = tl.load(targets_ptr + row)
            # Subtract before adding log(sum) to preserve precision for large logits.
            loss = (maximum - target_value) + tl.log(exp_sum)
            indices = tl.load(argmax_ptr + offsets, mask=tiles < NUM_TILES, other=VOCAB_SIZE)
            prediction = tl.min(tl.where(maxima == maximum, indices, VOCAB_SIZE), axis=0)
            tl.store(losses_ptr + row, loss)
            tl.store(correct_ptr + row, prediction == target)

        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        ptrs = logits_ptr + row * row_stride + cols
        values = tl.load(ptrs, mask=cols < VOCAB_SIZE, other=-float("inf")).to(tl.float32)
        probabilities = tl.exp(values - maximum) / exp_sum
        gradient = (probabilities - (cols == target).to(tl.float32)) / num_tokens
        tl.store(ptrs, gradient, mask=cols < VOCAB_SIZE)


def _triton_ce_inplace(logits, labels, losses, correct, num_tokens):
    """Write per-token loss/correct and replace contiguous logits by mean-CE gradients."""
    num_rows, vocab_size = logits.shape
    block_size = min(4096, triton.next_power_of_2(vocab_size))
    num_tiles = triton.cdiv(vocab_size, block_size)
    maxima = torch.empty((num_rows, num_tiles), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    argmax = torch.empty((num_rows, num_tiles), device=logits.device, dtype=torch.int32)
    targets = torch.empty(num_rows, device=logits.device, dtype=torch.float32)
    grid = (num_rows, num_tiles)
    # Explicit device selection matters when each DDP rank uses a different GPU.
    with torch.cuda.device(logits.device):
        _ce_tile_stats_kernel[grid](
            logits, labels, maxima, sums, argmax, targets,
            logits.stride(0), labels.stride(0),
            VOCAB_SIZE=vocab_size, NUM_TILES=num_tiles, BLOCK_SIZE=block_size,
            num_warps=4,
        )
        _ce_loss_grad_kernel[grid](
            logits, labels, maxima, sums, argmax, targets, losses, correct,
            logits.stride(0), labels.stride(0), num_tokens,
            VOCAB_SIZE=vocab_size, NUM_TILES=num_tiles, BLOCK_SIZE=block_size,
            STATS_SIZE=triton.next_power_of_2(num_tiles), num_warps=4,
        )


class FusedLinearCE(torch.autograd.Function):
    """Chunked full-vocabulary mean CE and correctness, with first-order gradients.

    Inputs are weight [V, H], hidden_states [N, H], and long labels [N] in
    [0, V). The caller must remove ignored (-100) positions before calling.
    CUDA uses Triton for CE/argmax/dlogits; linear GEMMs stay in PyTorch.
    Requires CUDA tensors and Triton. No full [N, V] activation is retained.
    """

    @staticmethod
    def forward(ctx, weight, hidden_states, labels):
        if not _TRITON_AVAILABLE:
            raise ImportError("FusedLinearCE requires Triton.")
        num_tokens, hidden_size = hidden_states.shape
        chunk_size = max(1, num_tokens * hidden_size // weight.shape[0])
        labels = labels.contiguous()
        losses = hidden_states.new_empty(num_tokens, dtype=torch.float32)
        correct = torch.empty_like(labels, dtype=torch.bool)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_hidden = torch.empty_like(hidden_states)

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            hidden = hidden_states[start:end]
            target = labels[start:end]
            logits = F.linear(hidden, weight)
            _triton_ce_inplace(logits, target, losses[start:end], correct[start:end], num_tokens)
            grad_logits = logits

            grad_hidden[start:end] = grad_logits @ weight
            grad_weight.add_(grad_logits.T @ hidden)
            del logits, grad_logits

        ctx.save_for_backward(grad_weight, grad_hidden)
        ctx.mark_non_differentiable(correct)
        return losses.sum() / num_tokens, correct

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, grad_correct):
        grad_weight, grad_hidden = ctx.saved_tensors
        return grad_weight * grad_loss, grad_hidden * grad_loss, None
