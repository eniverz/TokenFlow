import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, hidden_states, labels):
        num_tokens, hidden_size = hidden_states.shape
        chunk_size = max(1, num_tokens * hidden_size // weight.shape[0])
        loss = hidden_states.new_zeros((), dtype=torch.float32)
        correct = torch.empty_like(labels, dtype=torch.bool)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_hidden = torch.empty_like(hidden_states)

        for start in range(0, num_tokens, chunk_size):
            end = start + chunk_size
            hidden = hidden_states[start:end]
            target = labels[start:end]
            logits = F.linear(hidden, weight)
            scores = logits.float()
            loss += F.cross_entropy(scores, target, reduction="sum")
            correct[start:end] = scores.argmax(-1) == target

            # dCE/dlogits = (softmax(logits) - one_hot(target)) / num_tokens.
            grad_logits = scores.softmax(dim=-1)
            grad_logits[torch.arange(len(target), device=target.device), target] -= 1
            grad_logits = grad_logits.div_(num_tokens).to(logits.dtype)
            grad_hidden[start:end] = grad_logits @ weight
            grad_weight.add_(grad_logits.T @ hidden)
            del logits, scores, grad_logits

        ctx.save_for_backward(grad_weight, grad_hidden)
        ctx.mark_non_differentiable(correct)
        return loss / num_tokens, correct

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, grad_correct):
        grad_weight, grad_hidden = ctx.saved_tensors
        return grad_weight * grad_loss, grad_hidden * grad_loss, None
