module Whisper

const Maybe{T} = Union{Nothing, T}

import Downloads
import Pickle
import Base64

using AMDGPU
using AbstractFFTs
using Distributions
using FFTW
using FLAC
using FileIO
using Flux
using LinearAlgebra
using Printf
using ProgressMeter
using OrderedCollections
using Statistics

get_pb(n, desc::String) = Progress(
    n; desc, dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:white)

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

struct SRTEntry
    start_time::Float32
    end_time::Float32
    text::String
end

function SRTEntries(tokenizer::BPE, all_tokens::Vector{Vector{Int64}})
    srts = SRTEntry[]

    sot_id = tokenizer.special_tokens["<|startoftranscript|>"]
    eot_id = tokenizer.special_tokens["<|endoftext|>"]
    not_id = tokenizer.special_tokens["<|notimestamps|>"]

    time_offset = 0f0

    for tokens in all_tokens
        start_time, end_time = 0f0, 0f0
        start_idx, end_idx = 0, 0

        in_timestamp = false

        for (i, token) in enumerate(tokens)
            (token == sot_id || token == eot_id || token == not_id) && continue

            is_timestamp = token > not_id
            is_timestamp || continue

            in_timestamp ⊻= true

            if in_timestamp
                v = decode(tokenizer, token)
                start_time = parse(Float32, strip(v, ('<', '>', '|')))
                start_idx = i + 1
            else
                end_idx = i - 1
                text_segment = decode(tokenizer, @view(tokens[start_idx:end_idx]))

                v = decode(tokenizer, token)
                end_time = parse(Float32, strip(v, ('<', '>', '|')))
                push!(srts, SRTEntry(
                    start_time + time_offset,
                    end_time + time_offset,
                    text_segment))
            end
        end
        time_offset += 30f0
    end
    srts
end

function save(entries::Vector{SRTEntry}, filename::String)
    function format_timestamp(timestamp::Float32)
        hours, r = divrem(timestamp, 60 * 60)
        minutes, r = divrem(r, 60)
        seconds = floor(r)
        milliseconds = floor(Int, (r - seconds) * 1e3)
        "$(@sprintf("%02d", hours)):$(@sprintf("%02d", minutes)):$(@sprintf("%02d", seconds)),$(@sprintf("%03d", milliseconds))"
    end

    io = open(filename, "w")
    for (i, entry) in enumerate(entries)
        start_time = format_timestamp(entry.start_time)
        end_time = format_timestamp(entry.end_time)
        write(io, "$i\n")
        write(io, "$start_time --> $end_time\n")
        write(io, entry.text, "\n\n")
    end
    close(io)
    return
end

function transcribe(file_path::String, srt_path::String)
    @info "Running on $(Flux.GPU_BACKEND) GPU backend."

    device = gpu
    precision = f16
    model = WHISPER("tiny.en") |> precision |> device

    tokens_file = joinpath(pkgdir(Whisper), "data", "gpt2.tiktoken")
    ranks, special_tokens, n_vocab = prep_ranks(tokens_file)
    tokenizer = BPE(ranks; special_tokens)

    waveform, sample_rate::Int = load(file_path)
    @assert size(waveform, 2) == 1 # Mono
    @assert sample_rate == SAMPLE_RATE

    log_spec = prep_audio(waveform, sample_rate) |> precision
    content_frames = size(log_spec, 1)

    all_tokens = Vector{Int64}[]
    eot_id = tokenizer.special_tokens["<|endoftext|>"]

    # TODO decoding options
    temperature = 0.0f0
    sample_length = 448 ÷ 2
    max_context_size = 64

    n_frames = cld(content_frames, N_FRAMES)
    bar = get_pb(n_frames, "Transcribing: ")

    seek = 1
    i = 1
    while seek < content_frames
        segment = log_spec[seek:min(seek + (N_FRAMES - 1), content_frames), :, :]

        lst = [tokenizer("<|startoftranscript|>")] # 0-based idx.

        enc = model.encoder(segment |> device)
        for i in sample_length
            lst_start = max(1, length(lst) - max_context_size + 1)
            lst_view = @view(lst[lst_start:end])
            ctx = (Int32.(reshape(vcat(lst_view...), :, 1)) .+ Int32(1)) |> device # 1-based idx.

            dec = model.decoder(ctx, enc)
            # TODO decode, if collapse mode, increase temperature -> repeat whole loop for segment
            idx = if temperature ≈ 0
                argmax(dec[:, end, 1])
            else
                logits = Float32.(Array(@view(dec[:, end, 1])))
                p = softmax(logits ./ temperature)
                rand(Categorical(p))
            end
            push!(lst, [idx - 1])

            (idx - 1) == eot_id && break
        end
        push!(lst, [eot_id])
        push!(all_tokens, vcat(lst...))

        seek += N_FRAMES
        i += 1
        next!(bar)
    end

    srts = SRTEntries(tokenizer, all_tokens)
    save(srts, srt_path)
    return
end

end
