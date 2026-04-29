from attentions.operators.online_softmax_fwd_v1 import (
    online_softmax_attn_fwd_kernel,
    online_softmax_attention_forward_v1,
)


def online_softmax_attention_forward_v2(*args, **kwargs):
    return online_softmax_attention_forward_v1(*args, **kwargs)
