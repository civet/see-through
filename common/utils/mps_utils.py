import gc
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate


class MPSSlicedAttnProcessor2_0:
    """Drop-in replacement for AttnProcessor2_0 that slices large attention matrices on MPS.

    PyTorch's MPS backend does not always use FlashAttention for large multi-frame
    spatial sequences (e.g., 20 frames x 9216 tokens), causing F.scaled_dot_product_attention
    to materialize a multi-gigabyte (B, H, L_q, L_k) buffer and OOM. Slicing over batch
    and attention heads keeps peak memory under a bounded threshold while producing
    mathematically identical results.
    """

    def __init__(self, max_attn_bytes: int = 64 * 1024 * 1024):
        self.max_attn_bytes = max_attn_bytes

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored. Please remove it, "
                "as passing it will raise an error in the future. `scale` should directly be "
                "passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            )
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        per_head_bytes = query.shape[2] * key.shape[2] * query.element_size()
        total_attn_bytes = batch_size * attn.heads * per_head_bytes

        if query.device.type == "mps" and attention_mask is None and total_attn_bytes > self.max_attn_bytes:
            head_chunk = max(1, min(attn.heads, self.max_attn_bytes // max(1, per_head_bytes)))
            out_batches = []
            for b in range(batch_size):
                if head_chunk >= attn.heads:
                    out_batches.append(
                        F.scaled_dot_product_attention(
                            query[b : b + 1],
                            key[b : b + 1],
                            value[b : b + 1],
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=False,
                        )
                    )
                else:
                    out_heads = []
                    for h in range(0, attn.heads, head_chunk):
                        h_end = min(h + head_chunk, attn.heads)
                        out_heads.append(
                            F.scaled_dot_product_attention(
                                query[b : b + 1, h:h_end],
                                key[b : b + 1, h:h_end],
                                value[b : b + 1, h:h_end],
                                attn_mask=None,
                                dropout_p=0.0,
                                is_causal=False,
                            )
                        )
                    out_batches.append(torch.cat(out_heads, dim=1))
            hidden_states = torch.cat(out_batches, dim=0)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def apply_mps_memory_optimizations(pipeline, max_attn_bytes: int = 64 * 1024 * 1024):
    """Install MPS-safe sliced attention and VAE slicing on a diffusers pipeline."""
    if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "set_attn_processor"):
        pipeline.unet.set_attn_processor(MPSSlicedAttnProcessor2_0(max_attn_bytes=max_attn_bytes))
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()


def empty_device_cache():
    """Run Python GC and release cached CUDA or MPS memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
