module Whisper

const Maybe{T} = Union{Nothing, T}

import Downloads
import Pickle
import Base64

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

function _cache_dir()
    cdr = joinpath(homedir(), ".cache", "Whisper.jl")
    isdir(cdr) || mkdir(cdr)
    cdr
end

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
- support translation
- detect language
- check for collapse mode, then increase temperature and re-run on segment
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
            token == sot_id && continue

            is_timestamp = token ≥ not_id || (token == eot_id && in_timestamp)
            is_timestamp || continue

            # If we reach the end, but no timestamp.
            token == eot_id && !is_timestamp && continue
            in_timestamp ⊻= true

            if in_timestamp
                start_time = if token == not_id
                    0f0
                else
                    v = decode(tokenizer, token)
                    parse(Float32, strip(v, ('<', '>', '|')))
                end
                start_idx = i + 1
            else
                end_idx = i - 1
                text_segment = decode(tokenizer, @view(tokens[start_idx:end_idx]))

                end_time = if token ≤ not_id
                    30f0
                else
                    v = decode(tokenizer, token)
                    parse(Float32, strip(v, ('<', '>', '|')))
                end
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

function transcribe(
    file_path::String, srt_path::String;
    model_name::String = "tiny.en", language::Maybe{String} = nothing,
    dev = gpu, precision = f16,
)
    multilingual = !endswith(model_name, ".en")

    if (language ≢ nothing && language != "english") && multilingual
        error("""
            Speicifed language `$language`, but model `$model_name` supports only English.
            Try dropping `.en` part from the model name.
            Here's the list of all supported models:
            $(keys(_MODELS)).
        """)
    end

    @info "Running on `$dev` device at `$precision` precision."

    if endswith(file_path, ".flac")
        waveform, sample_rate::Int = load(file_path)
        if size(waveform, 2) != 1 || sample_rate != SAMPLE_RATE
            waveform, sample_rate = convert_audio(file_path)
        end
    else
        waveform, sample_rate = convert_audio(file_path)
    end

    log_spec = prep_audio(waveform, sample_rate,
        n_mels=model_name == "large-v3" ? N_MELS_V3 : N_MELS) |> precision
    content_frames = size(log_spec, 1)

    model = WHISPER(model_name) |> precision |> dev
    tokenizer = BPE(; language, multilingual)

    n_frames = cld(content_frames, N_FRAMES)
    bar = get_pb(n_frames, "Transcribing: ")

    all_tokens = Vector{Int64}[]
    seek = 1
    while seek < content_frames
        segment = log_spec[seek:min(seek + (N_FRAMES - 1), content_frames), :, :]
        tokens = decode(model, tokenizer, dev(segment))
        push!(all_tokens, tokens)

        seek += N_FRAMES
        next!(bar)
    end

    @info "Saving results to: `$srt_path`."
    srts = SRTEntries(tokenizer, all_tokens)
    save(srts, srt_path)
    return
end

function decode(
    model::WHISPER, tokenizer::BPE, segment;
    sample_length::Int = decoder_ctx_size(model) ÷ 2,
    max_context_size = sample_length,
    temperature::Float32 = 0f0,
)
    dev = get_backend(model)

    eot_id = tokenizer.special_tokens["<|endoftext|>"]
    transcribe_id = tokenizer.special_tokens["<|transcribe|>"]
    tokens = Int32[tokenizer("<|startoftranscript|>")...] # 0-based idx.

    # If specified, append language & task tokens.
    if !isempty(tokenizer.language)
        push!(tokens, tokens[1] + tokenizer.language_idx)
        push!(tokens, transcribe_id)
    end

    enc = model.encoder(segment)
    for i in 1:sample_length
        tokens_start = max(1, length(tokens) - max_context_size + 1)
        tokens_view = @view(tokens[tokens_start:end])
        ctx = (reshape(tokens_view, :, 1) .+ Int32(1)) |> dev # 1-based idx.

        dec = model.decoder(ctx, enc)
        idx = if temperature ≈ 0f0
            argmax(@view(dec[:, end, 1]))
        else
            logits = @view(dec[:, end, 1])
            eltype(logits) == Float32 || (logits = Float32.(logits);)
            logits .*= (1f0 / temperature)

            p = softmax(logits) |> cpu
            rand(Categorical(p))
        end
        push!(tokens, idx - 1)

        (idx - 1) == eot_id && break
    end
    push!(tokens, eot_id)

    tokens
end

end
