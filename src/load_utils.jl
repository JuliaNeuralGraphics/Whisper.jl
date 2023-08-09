const _MODELS = Dict{String, String}(
    "tiny.en" => "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny" => "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en" => "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base" => "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en" => "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small" => "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en" => "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium" => "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1" => "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2" => "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large" => "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
)

function load_state(model_name::String)
    weights_url = get(_MODELS, model_name, nothing)
    weights_url ≡ nothing && error("""
        Unsupported model name: `$model_name`.
        Supported models: `$(keys(_MODELS))`.
    """)

    cache_dir = _cache_dir()
    @info "Cache directory for weights: $cache_dir."

    weights_file = joinpath(cache_dir, model_name * ".pt")
    if !isfile(weights_file)
        @info "Downloading weights: $weights_url"
        Downloads.download(weights_url, weights_file)
    end
    Pickle.Torch.THload(weights_file)
end

# This is sad :(

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
