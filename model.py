import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inv_rms).to(x.dtype) * self.weight


def apply_rope(q, k):
    """Rotary position embeddings. q/k: [B, heads, T, head_dim]."""
    head_dim = q.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension.")

    positions = torch.arange(q.size(-2), device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000 ** (
            torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    def rotate(x):
        even = x[..., 0::2].float()
        odd = x[..., 1::2].float()
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, window_size=None):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(
            batch, seq_len, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, k)

        if self.window_size is None:
            mask = None
            is_causal = True
        else:
            pos = torch.arange(seq_len, device=x.device)
            q_pos = pos[:, None]
            k_pos = pos[None, :]
            mask = (k_pos <= q_pos) & (
                k_pos >= q_pos - self.window_size + 1
            )
            is_causal = False

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(out)


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    """
    Educational DeepSeek-inspired sparse MoE.
    Uses a learned router, Top-K expert selection, routed SwiGLU experts,
    one always-active shared expert, and an auxiliary balancing loss.
    It is not an exact DeepSeek-V4 implementation.
    """
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.shared_expert = SwiGLUExpert(d_model, d_ff)
        self.last_indices = None
        self.last_weights = None

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        flat_x = x.reshape(-1, d_model)

        router_logits = self.router(flat_x)
        router_probs = torch.softmax(router_logits.float(), dim=-1)

        # A simplified V4-inspired affinity transformation.
        affinity = torch.sqrt(F.softplus(router_logits.float()))
        top_values, top_indices = torch.topk(
            affinity, k=self.top_k, dim=-1
        )
        top_weights = top_values / top_values.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)

        self.last_indices = top_indices.detach()
        self.last_weights = top_weights.detach()

        routed_output = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_rows, slots = torch.where(top_indices == expert_id)
            if token_rows.numel() == 0:
                continue

            expert_input = flat_x.index_select(0, token_rows)
            expert_output = expert(expert_input)
            weights = top_weights[token_rows, slots].unsqueeze(-1)
            routed_output = routed_output.index_add(
                0, token_rows, expert_output * weights
            )

        shared_output = self.shared_expert(flat_x)
        output = routed_output + shared_output

        # Switch-style auxiliary balancing loss.
        assignments = F.one_hot(
            top_indices, num_classes=self.num_experts
        ).float()
        tokens_per_expert = assignments.mean(dim=(0, 1))
        mean_router_prob = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(
            tokens_per_expert * mean_router_prob
        )

        return output.view(batch, seq_len, d_model), aux_loss


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model, n_heads, d_ff, num_experts, top_k,
        dropout=0.0, attention_window=None
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model, n_heads, dropout, window_size=attention_window
        )
        self.moe_norm = RMSNorm(d_model)
        self.moe = SparseMoE(d_model, d_ff, num_experts, top_k)

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        return x + moe_out, aux_loss


class MiniDeepSeekMoE(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=512,
        num_experts=4,
        top_k=2,
        max_seq_len=256,
        sliding_window=64,
        dropout=0.0,
        gradient_checkpointing=True
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for layer_idx in range(n_layers):
            # Alternate local-window and full causal attention.
            window = sliding_window if layer_idx % 2 == 0 else None
            self.blocks.append(
                TransformerBlock(
                    d_model, n_heads, d_ff, num_experts, top_k,
                    dropout=dropout, attention_window=window
                )
            )

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self.embedding_dropout(self.token_embedding(input_ids))
        aux_losses = []

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                def block_forward(hidden, current_block=block):
                    return current_block(hidden)
                x, aux = checkpoint(
                    block_forward, x, use_reentrant=False
                )
            else:
                x, aux = block(x)
            aux_losses.append(aux)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, torch.stack(aux_losses).mean()
