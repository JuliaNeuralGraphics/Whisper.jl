# This is sad :(

# TODO remove key and check it is empty at the end
function load_state!(model::WHISPER, state)
    load_state!(model.encoder, state)
    load_state!(model.decoder, state)
    @assert isempty(state)
end

function load_state!(decoder::TextDecoder, state)
    pre_key = "decoder."

    load_state!(decoder.token_embedding,
        pop!(state, pre_key * "token_embedding.weight"))
    load_state!(decoder.positional_embedding,
        pop!(state, pre_key * "positional_embedding"))

    for (i, block) in enumerate(decoder.blocks)
        load_state!(block, state, pre_key * "blocks.$(i - 1).")
    end

    load_state!(decoder.ln,
        pop!(state, pre_key * "ln.weight"),
        pop!(state, pre_key * "ln.bias"))
end

function load_state!(encoder::AudioEncoder, state)
    pre_key = "encoder."

    load_state!(encoder.conv1,
        pop!(state, pre_key * "conv1.weight"),
        pop!(state, pre_key * "conv1.bias"))

    load_state!(encoder.conv2,
        pop!(state, pre_key * "conv2.weight"),
        pop!(state, pre_key * "conv2.bias"))

    for (i, block) in enumerate(encoder.blocks)
        load_state!(block, state, pre_key * "blocks.$(i - 1).")
    end

    load_state!(encoder.ln_post,
        pop!(state, pre_key * "ln_post.weight"),
        pop!(state, pre_key * "ln_post.bias"))

    load_state!(encoder.positional_embedding,
        pop!(state, pre_key * "positional_embedding"))
end

function load_state!(block::ResidualAttentionBlock, state, pre_key::String)
    load_state!(block.attn, state, pre_key * "attn.")
    load_state!(block.attn_ln,
        pop!(state, pre_key * "attn_ln.weight"),
        pop!(state, pre_key * "attn_ln.bias"))

    isnothing(block.cross_attn) ||
        load_state!(block.cross_attn, state, pre_key * "cross_attn.")
    isnothing(block.cross_attn_ln) || load_state!(block.cross_attn_ln,
        pop!(state, pre_key * "cross_attn_ln.weight"),
        pop!(state, pre_key * "cross_attn_ln.bias"))

    load_state!(block.mlp[1],
        pop!(state, pre_key * "mlp.0.weight"),
        pop!(state, pre_key * "mlp.0.bias"))
    load_state!(block.mlp[2],
        pop!(state, pre_key * "mlp.2.weight"), # `2` because of GELU
        pop!(state, pre_key * "mlp.2.bias"))
    load_state!(block.mlp_ln,
        pop!(state, pre_key * "mlp_ln.weight"),
        pop!(state, pre_key * "mlp_ln.bias"))
end

function load_state!(mha::MultiHeadAttention, state, pre_key::String)
    load_state!(mha.query,
        pop!(state, pre_key * "query.weight"),
        pop!(state, pre_key * "query.bias"))
    load_state!(mha.key, pop!(state, pre_key * "key.weight"), nothing)
    load_state!(mha.value,
        pop!(state, pre_key * "value.weight"),
        pop!(state, pre_key * "value.bias"))
    load_state!(mha.out,
        pop!(state, pre_key * "out.weight"),
        pop!(state, pre_key * "out.bias"))
end

function load_state!(layer::Flux.Conv, weight, bias)
    # BCH -> HCB & flip kernel from cross-correlation to convolution.
    copy!(layer.weight, permutedims(weight, (3, 2, 1))[end:-1:1, :, :])
    copy!(layer.bias, bias)
end

function load_state!(layer::Flux.Dense, weight, bias)
    copy!(layer.weight, weight)
    isnothing(bias) || copy!(layer.bias, bias)
end

function load_state!(ln::Flux.LayerNorm, scale, bias)
    copy!(ln.diag.scale, scale)
    copy!(ln.diag.bias, bias)
end

function load_state!(emb::Flux.Embedding, state)
    copy!(emb.weight, transpose(state))
end
