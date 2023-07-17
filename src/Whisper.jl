module Whisper

const Maybe{T} = Union{Nothing, T}

import Downloads
import Pickle
import Base64

using AbstractFFTs
using FFTW
using FLAC
using FileIO
using Flux
using Printf
using OrderedCollections
using Statistics

function (ln::Flux.LayerNorm)(x::AbstractArray)
    ϵ = convert(float(eltype(x)), ln.ϵ)
    μ, σ² = _normalize(x; dims=1:length(ln.size))
    y = ln.diag((x .- μ) .* inv.(sqrt.(σ² .+ ϵ)))
    return y
end

function _normalize(x::AbstractArray{Float16}; dims)
    x_fp32 = Float32.(x)
    μ, σ² = _normalize(x_fp32; dims)
    m, v = Float16.(μ), Float16.(σ²)
    return m, v
end

function _normalize(x; dims)
    μ = mean(x; dims)
    σ² = var(x; dims, mean=μ, corrected=false)
    μ, σ²
end

include("tokenizer.jl")
include("audio.jl")
include("model.jl")
include("load_utils.jl")

"""
TODO
- compare tiktoken with simple bpe
- get_encoding
- re-encode using ffmpeg:
    ffmpeg -i in.flac -ac 1 -ar 16000 out.flac
"""

function transcribe(file_path::String = "/home/pxl-th/Downloads/out.flac")
    weights_url = "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt"
    weights_file = "./whisper-tiny.en.pt"
    if !isfile(weights_file)
        @info "Downloading weights: $weights_url"
        Downloads.download(weights_url, weights_file)
    end

    state = Pickle.Torch.THload(weights_file)

    dims = state["dims"]
    encoder_kwargs = (;
        n_mels=dims["n_mels"],
        n_audio_state=dims["n_audio_state"], n_audio_layer=dims["n_audio_layer"],
        n_audio_head=dims["n_audio_head"], n_audio_ctx=dims["n_audio_ctx"])
    decoder_kwargs = (;
        n_vocab=dims["n_vocab"],
        n_text_state=dims["n_text_state"], n_text_layer=dims["n_text_layer"],
        n_text_head=dims["n_text_head"], n_text_ctx=dims["n_text_ctx"])

    tokens_file = "./gpt2.tiktoken"
    ranks, special_tokens, n_vocab = prep_ranks(tokens_file)
    tokenizer = BPE(ranks; special_tokens)

    model = WHISPER(; encoder_kwargs, decoder_kwargs)
    load_state!(model, state["model_state_dict"])

    waveform, sample_rate::Int = load(file_path)
    @assert size(waveform, 2) == 1 # Mono
    @assert sample_rate == 16000

    lst = [tokenizer("<|startoftranscript|>")] # 0-based idx.

    log_spec = prep_audio(waveform, sample_rate)
    enc = model.encoder(log_spec)

    # Decode one token at a time.
    dec_str = ""
    for i in 1:100
        ctx = Int32.(reshape(vcat(lst...), :, 1)) .+ Int32(1) # 1-based idx.
        dec = model.decoder(ctx, enc)
        idx = argmax(dec[:, end, 1]) # `end` take last seq elem; `1` - no batches rn
        push!(lst, [idx - 1])

        dec_str = decode(tokenizer, vcat(lst...))
        occursin("<|endoftext|>", dec_str) && break
    end
    println(dec_str)
    return
end

function mmm()
    tokens_file = "./gpt2.tiktoken"
    ranks, special_tokens, n_vocab = prep_ranks(tokens_file)
    enc = BPE(ranks; special_tokens)

    # @show enc("<|startoftranscript|><|notimestamps|>")
    # @show decode(enc, enc("<|startoftranscript|>hello"))
    return
end

end
