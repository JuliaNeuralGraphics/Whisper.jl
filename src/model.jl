struct MultiHeadAttention{Q, K}
    query::Q
    key::K
    value::Q
    out::Q

    n_head::Int
end
Flux.@functor MultiHeadAttention

function MultiHeadAttention(n_state::Int, n_head::Int)
    query = Dense(n_state => n_state)
    key = Dense(n_state => n_state; bias=false)
    value = Dense(n_state => n_state)
    out = Dense(n_state => n_state)
    MultiHeadAttention(query, key, value, out, n_head)
end

function (mha::MultiHeadAttention)(
    x::T, xa::Maybe{T} = nothing; mask::Maybe{M} = nothing,
) where {
    T <: AbstractArray{<: Real, 3},
    M <: AbstractMatrix{Bool},
}
    q = mha.query(x)
    k = mha.key(isnothing(xa) ? x : xa)
    v = mha.value(isnothing(xa) ? x : xa)

    # TODO reshape mask
    ω, _ = dot_product_attention(q, k, v; mask, nheads=mha.n_head)
    mha.out(ω)
end

struct ResidualAttentionBlock{A1, L1, A2, L2, M, L3}
    attn::A1
    attn_ln::L1

    cross_attn::A2
    cross_attn_ln::L2

    mlp::M
    mlp_ln::L3
end
Flux.@functor ResidualAttentionBlock

function ResidualAttentionBlock(n_state::Int, n_head::Int; cross_attention::Bool = false)
    attn = MultiHeadAttention(n_state, n_head)
    attn_ln = LayerNorm(n_state)

    cross_attn = cross_attention ? MultiHeadAttention(n_state, n_head) : nothing
    cross_attn_ln = cross_attention ? LayerNorm(n_state) : nothing

    mlp = Chain(Dense(n_state => n_state * 4, gelu), Dense(n_state * 4 => n_state))
    mlp_ln = LayerNorm(n_state)

    ResidualAttentionBlock(attn, attn_ln, cross_attn, cross_attn_ln, mlp, mlp_ln)
end

function (res::ResidualAttentionBlock)(
    x::T, xa::Maybe{T} = nothing; mask::Maybe{M} = nothing,
) where {
    T <: AbstractArray{<: Real, 3},
    M <: AbstractMatrix{Bool},
}
    x = x .+ res.attn(res.attn_ln(x); mask)
    if !isnothing(res.cross_attn)
        x = x .+ res.cross_attn(res.cross_attn_ln(x), xa)
    end
    x .+ res.mlp(res.mlp_ln(x))
end

struct AudioEncoder{C1, C2, B, L, E}
    conv1::C1
    conv2::C2
    blocks::B
    ln_post::L
    positional_embedding::E
end
Flux.@functor AudioEncoder

function AudioEncoder(;
    n_mels::Int, n_audio_state::Int,
    n_audio_layer::Int, n_audio_head::Int, n_audio_ctx::Int,
)
    conv1 = Conv((3,), n_mels => n_audio_state, gelu; pad=1)
    conv2 = Conv((3,), n_audio_state => n_audio_state, gelu; pad=1, stride=2)
    blocks = Chain([
        ResidualAttentionBlock(n_audio_state, n_audio_head)
        for _ in 1:n_audio_layer]...)
    ln_post = LayerNorm(n_audio_state)
    positional_embedding = Embedding(n_audio_ctx => n_audio_state)

    AudioEncoder(conv1, conv2, blocks, ln_post, positional_embedding)
end

function (enc::AudioEncoder)(x::T) where T <: AbstractArray{<: Real, 3}
    x = x |> enc.conv1 |> enc.conv2
    x = permutedims(x, (2, 1, 3)) # Swap `W` & `audio_state`.
    emb = enc.positional_embedding(1:size(x, 2)) # TODO check embedding is not too small
    x = x .+ emb
    x |> enc.blocks |> enc.ln_post
end

struct TextDecoder{T, P, B, L}
    token_embedding::T
    positional_embedding::P
    blocks::B
    ln::L
end
Flux.@functor TextDecoder

get_backend(dec::TextDecoder) =
    typeof(dec.positional_embedding.weight) <: Array ? cpu : gpu

function TextDecoder(;
    n_vocab::Int, n_text_state::Int,
    n_text_layer::Int, n_text_head::Int, n_text_ctx::Int,
)
    token_embedding = Embedding(n_vocab => n_text_state)
    positional_embedding = Embedding(n_text_ctx => n_text_state)
    blocks = Chain([
        ResidualAttentionBlock(n_text_state, n_text_head; cross_attention=true)
        for _ in 1:n_text_layer]...)
    ln = LayerNorm(n_text_state)
    TextDecoder(token_embedding, positional_embedding, blocks, ln)
end

function (dec::TextDecoder)(x::I, xa::T) where {
    I <: AbstractMatrix{<: Int32},
    T <: AbstractArray{<: Real, 3},
}
    t = dec.token_embedding(x) .+ dec.positional_embedding(1:size(x, 1))
    mask = make_causal_mask(x; dims=1) |> get_backend(dec)
    for block in dec.blocks
        t = block(t, xa; mask)
    end
    t = dec.ln(t)
    transpose(dec.token_embedding.weight) ⊠ t # Reversed
end

struct WHISPER{E, D}
    encoder::E
    decoder::D
end
Flux.@functor WHISPER

function WHISPER(; encoder_kwargs, decoder_kwargs)
    WHISPER(AudioEncoder(; encoder_kwargs...), TextDecoder(; decoder_kwargs...))
end
