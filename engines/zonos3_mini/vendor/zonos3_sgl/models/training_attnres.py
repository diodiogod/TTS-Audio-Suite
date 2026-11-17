"""Reproduce the trainer's fused two-phase AttnRes execution."""

from . import training_attnres_kernels as ar


def forward_layers(model, x):
    layers = model.layers.op_list
    block_size = model.config.block_attnres_block_size // 2
    ctx = ar.FusedBlockAttnResContext((len(layers) + block_size - 1) // block_size + 1, x, gradient_dtype=x.dtype)
    router_states = None
    for start in range(0, len(layers), block_size):
        end = min(start + block_size, len(layers))
        queries = model.block_attnres_queries[2 * start : 2 * end]
        outputs, lses = ar.shared_flash_phase1_split_norm(
            ctx, x, queries, layers[start].attention_norm.weight, model.config.rms_norm_eps
        )
        for i in range(start, end):
            layer = layers[i]
            j = 2 * (i - start)
            inp = (
                outputs[j]
                if i == start
                else ar.fused_phase2_norm(
                    x, queries[j], outputs[j], lses[j], layer.attention_norm.weight, model.config.rms_norm_eps
                )
            )
            a = layer.attention.forward(inp)
            h = a if i == start else x + a
            inp = ar.fused_phase2_norm(
                h, queries[j + 1], outputs[j + 1], lses[j + 1], layer.ffn_norm.weight, model.config.rms_norm_eps
            )
            if layer.is_moe:
                ff, router_states = layer.feed_forward.forward(inp, router_states)
            else:
                ff = layer.feed_forward.forward(inp)
                router_states = None
            x = h + ff
    outputs, _ = ar.shared_flash_phase1_norm(
        ctx, x, model.block_attnres_queries[-1:].contiguous(), model.out_norm.weight, model.config.rms_norm_eps
    )
    return outputs[0]


def forward_layers_full(model, x, state=None):
    """Reproduce the trainer's fused Full AttnRes schedule (LMTransformer._forward_full_attnres with the Triton bank).
    ``state`` reuses a persistent bank of the right shape (reset here) instead of allocating one per forward."""
    layers = model.layers.op_list
    eps = model.config.rms_norm_eps
    queries = model.block_attnres_queries
    if state is None:
        ctx = ar.FusedBlockAttnResContext(2 * len(layers), x, gradient_dtype=x.dtype)
    else:
        ctx = state
        ctx.completed_count = 0
        ctx.call_count = 0
        ctx._routing_mode = None

    def route(source, query, append=True):
        return ar.fused_full_attnres(ctx, source, query, eps, append)

    delta = model.steer_delta(x.shape[0])
    source = model.steer_bank_source(0, x, delta)
    router_states = None
    for i, layer in enumerate(layers):
        attention_out = model.steer_bank_source(
            2 * i + 1,
            layer.attention.forward(layer.attention_norm.forward(route(source, queries[2 * i]), None)[0]),
            delta,
        )
        ffn_input = layer.ffn_norm.forward(route(attention_out, queries[2 * i + 1]), None)[0]
        if layer.is_moe:
            ffn_out, router_states = layer.feed_forward.forward(ffn_input, router_states)
        else:
            ffn_out = layer.feed_forward.forward(ffn_input)
            router_states = None
        source = model.steer_bank_source(2 * i + 2, ffn_out, delta)
    h = route(source, queries[-1], append=False)
    return model.out_norm.forward(h, None)[0]
