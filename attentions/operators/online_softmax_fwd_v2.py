from attentions.operators.online_softmax_fwd_v1 import (
    online_softmax_attn_fwd_kernel,
    online_softmax_attention_forward_v1,
)


def online_softmax_attention_forward_v2(
    q,
    k,
    v,
    *,
    causal,
    scale=None,
    window_size=0,
    block_qm=16,
    block_kn=32,
    output_dtype=None,
    num_warps=None,
    num_stages=None,
):
    return online_softmax_attention_forward_v1(
        q,
        k,
        v,
        causal=causal,
        scale=scale,
        window_size=window_size,
        block_qm=block_qm,
        block_kn=block_kn,
        output_dtype=output_dtype,
        num_warps=num_warps,
        num_stages=num_stages,
    )
