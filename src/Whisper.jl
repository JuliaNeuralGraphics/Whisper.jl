module Whisper

const Maybe{T} = Union{Nothing, T}

import Downloads
import Pickle
import Base64

using AMDGPU
using AbstractFFTs
using FFTW
using FLAC
using FileIO
using Flux
using Printf
using OrderedCollections
using Statistics

include("tokenizer.jl")
include("audio.jl")
include("model.jl")
include("load_utils.jl")

function WHISPER(model_name::String)
    state = load_state(model_name)

    dims = state["dims"]
    encoder_kwargs = (;
        n_mels=dims["n_mels"],
        n_audio_state=dims["n_audio_state"], n_audio_layer=dims["n_audio_layer"],
        n_audio_head=dims["n_audio_head"], n_audio_ctx=dims["n_audio_ctx"])
    decoder_kwargs = (;
        n_vocab=dims["n_vocab"],
        n_text_state=dims["n_text_state"], n_text_layer=dims["n_text_layer"],
        n_text_head=dims["n_text_head"], n_text_ctx=dims["n_text_ctx"])

    model = WHISPER(; encoder_kwargs, decoder_kwargs)
    load_state!(model, state["model_state_dict"])
    model
end

"""
TODO
- save to .srt file
- multilingual
- re-encode using ffmpeg:
    ffmpeg -i in.flac -ac 1 -ar 16000 out.flac
- detect language
- streaming
"""

function transcribe(file_path::String)
    @info "Running on $(Flux.GPU_BACKEND) GPU backend."

    device = gpu
    precision = f16
    model = WHISPER("tiny.en") |> precision |> device

    tokens_file = joinpath(pkgdir(Whisper), "data", "gpt2.tiktoken")
    ranks, special_tokens, n_vocab = prep_ranks(tokens_file)
    tokenizer = BPE(ranks; special_tokens)

    eot_id = tokenizer.special_tokens["<|endoftext|>"]

    waveform, sample_rate::Int = load(file_path)
    @assert size(waveform, 2) == 1 # Mono
    @assert sample_rate == SAMPLE_RATE

    log_spec = prep_audio(waveform, sample_rate) |> precision
    content_frames = size(log_spec, 1)

    all_tokens = Vector{Int64}[]

    seek = 1
    i = 1
    while seek < content_frames
        i == 10 && break

        time_offset = (seek - 1) * HOPSIZE / SAMPLE_RATE
        segment = log_spec[seek:min(seek + (N_FRAMES - 1), content_frames), :, :]

        lst = [tokenizer("<|startoftranscript|>")] # 0-based idx.

        enc = model.encoder(segment |> device)
        while true
            ctx = (Int32.(reshape(vcat(lst...), :, 1)) .+ Int32(1)) |> device # 1-based idx.

            dec = model.decoder(ctx, enc)
            idx = argmax(dec[:, end, 1])
            push!(lst, [idx - 1])

            (idx - 1) == eot_id && break
        end
        push!(all_tokens, vcat(lst...))

        seek += N_FRAMES
        i += 1
    end

    for tokens in all_tokens
        println(decode(tokenizer, tokens; include_specials=false))
        println()
    end
    return
end

end
