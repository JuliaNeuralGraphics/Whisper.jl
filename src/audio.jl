function prep_audio(
    waveform::Matrix{Float32}, sample_rate::Integer;
    n_fft::Integer = 400, hopsize::Integer = 160, n_mels::Int = 80,
)
    freqs = stft(waveform; n_fft, hopsize)
    magnitudes = abs.(freqs[:, 1:(end - 1), :]).^2

    filters = mel(Float32(sample_rate); n_fft, n_mels) |> transpose
    mel_spec = filters ⊠ magnitudes

    log_spec = log10.(max.(mel_spec, 1f-10))
    log_spec = max.(maximum(log_spec) - 8f0, log_spec)
    permutedims((log_spec .+ 4f0) ./ 4f0, (2, 1, 3))
end

function hanning(n::Integer)
    scale = π / Float32(n)
    [sin(scale * k)^2 for k in UnitRange{Float32}(0, n - 1)]
end

"""
Short-time Fourier transform (STFT).

The STFT represents a signal in the time-frequency domain by
computing discrete Fourier transforms (DFT) over short overlapping windows.
"""
function stft(
    x::Matrix{Float32}; n_fft::Int, hopsize::Int,
    center::Bool = true, padf::Function = pad_reflect,
)
    window = hanning(n_fft)
    start, extra = 1, 0

    # TODO if not center, create 0-sized x_frames_pre, x_frames_post.
    if center
        # How many frames depend on left padding?
        start_k = ceil(Int, n_fft ÷ 2 / hopsize)
        # What's the first frame that depends on extra right padding?
        tail_k = (size(x, 1) + n_fft ÷ 2 - n_fft) ÷ hopsize + 1

        if tail_k ≤ start_k
            # Tail & head overlap.
            padding = (n_fft ÷ 2, n_fft ÷ 2)
            x = padf(x, padding)
        else
            # Padding on each part separately.
            # Middle of the signal.
            start = start_k * hopsize - n_fft ÷ 2 + 1
            padding = (n_fft ÷ 2, 0)

            slice = 1:((start_k - 1) * hopsize - n_fft ÷ 2 + n_fft + 1)
            x_pre = padf(x[slice, :], padding)
            x_frames_pre = splitframes(x_pre; framelen=n_fft, hopsize)
            # Trim to exact number of frames.
            x_frames_pre = x_frames_pre[:, 1:start_k, :]

            # How many extra frames we have from the head?
            extra = size(x_frames_pre, 2)

            # Determine if we have any frames that will fit inside the tail pad.
            if (tail_k * hopsize - n_fft ÷ 2 + n_fft) ≤ (size(x, 1) + n_fft ÷ 2)
                padding = (0, n_fft ÷ 2)
                s = tail_k * hopsize - n_fft ÷ 2 + 1
                x_post = padf(x[s:end, :], padding)
                x_frames_post = splitframes(x_post; framelen=n_fft, hopsize)
                extra += size(x_frames_post, 2)
            else
                x_frames_post = similar(x, 0, 0, 0)
            end
        end
    end

    x_frames = splitframes(x[start:end, :]; framelen=n_fft, hopsize)

    n_channels = size(x, 2)
    n_frames = size(x_frames, 2) + extra
    n_bins = 1 + n_fft ÷ 2
    y = Array{ComplexF32}(undef, n_bins, n_frames, n_channels)
    # Process pre & post.
    if center && extra > 0
        y[:, 1:size(x_frames_pre, 2), :] = rfft(x_frames_pre .* window, 1)
        if size(x_frames_post, 2) > 0
            y[:, end - size(x_frames_post, 2) + 1:end, :] = rfft(x_frames_post .* window, 1)
        end
    end
    # Process the rest.
    y_offset = size(x_frames_pre, 2)
    for i in 1:size(x_frames, 2)
        y[:, i + y_offset, :] = rfft(x_frames[:, i, :] .* window, 1)
    end
    return y
end

function stft(x; framelen::Int, hopsize::Int, window = hanning(framelen))
    frames = splitframes(x; framelen, hopsize)

    freqbins = framelen ÷ 2 + 1
    n_frames, n_channels = size(frames, 2), size(x, 2)

    spectrogram = Array{ComplexF32}(undef, freqbins, n_frames, n_channels)
    for ch in 1:n_channels, i in 1:n_frames
        spectrogram[:, i, ch] = rfft(frames[:, i, ch] .* window)
    end
    spectrogram
end

# splitframes performs overlapping frame splitting.
function splitframes(x::Matrix{Float32}; framelen::Int, hopsize::Int)
    n_frames = countframes(x, framelen, hopsize)
    n_channels = size(x, 2)

    frames = Array{eltype(x)}(undef, framelen, n_frames, n_channels)
    for ch in 1:n_channels, i in 1:n_frames
        s = (i - 1) * hopsize + 1
        e = (i - 1) * hopsize + framelen
        frames[:, i, ch] = x[s:e, ch]
    end
    frames
end

# countframes returns the number of frames that will be processed.
function countframes(x, framelen::Int, hopsize::Int)
    (size(x, 1) - framelen) ÷ hopsize + 1
end

function hz2mel(ω::Float32; htk::Bool = false)
    htk && return 2595f0 * log10(1f0 + ω / 700f0)

    fmin = 0f0
    fsp = 200f0 / 3f0
    mels = (ω - fmin) /fsp

    min_log_ω = 1000f0
    if ω ≥ min_log_ω
        min_log_mel = (min_log_ω - fmin) / fsp
        logstep = log(6.4f0) / 27f0
        mels = min_log_mel + log(ω / min_log_ω) / logstep
    end
    mels
end

function mel2hz(mels::Float32; htk::Bool = false)
    htk && return 700f0 * (10f0^(mels / 2595f0) - 1f0)

    fmin = 0f0
    fsp = 200f0 / 3f0
    ω = fmin + fsp * mels

    min_log_ω = 1000f0
    min_log_mel = (min_log_ω - fmin) / fsp
    if mels ≥ min_log_mel
        logstep = log(6.4f0) / 27f0
        ω = min_log_ω * exp(logstep * (mels - min_log_mel))
    end
    ω
end

function mel(
    sample_rate::Float32 = 16000f0; n_fft::Int = 400, n_mels::Int = 80,
    fmin::Float32 = 0f0, fmax::Float32 = sample_rate ÷ 2,
)
    fft_ω = fftfreq(n_fft, sample_rate)[1:201]
    fft_ω[end] *= -1f0 # Flip negative sign.

    mel_ω = range(hz2mel(fmin), hz2mel(fmax), n_mels + 2) .|> mel2hz
    Δ = @view(mel_ω[2:end]) .- @view(mel_ω[1:end - 1])

    ramps = Matrix{Float32}(undef, length(fft_ω), length(mel_ω))
    for i in 1:length(fft_ω)
        ftω = fft_ω[i]
        for j in 1:length(mel_ω)
            ramps[i, j] = mel_ω[j] - ftω
        end
    end

    ω = zeros(Float32, 1 + n_fft ÷ 2, n_mels)
    for i in 1:n_mels
        lower = -ramps[:, i] ./ Δ[i]
        upper = ramps[:, i + 2] ./ Δ[i + 1]
        ω[:, i] = max.(0, min.(lower, upper))
    end

    # Normalize.
    enorm = 2f0 ./ (mel_ω[3:end] .- mel_ω[1:end - 2])
    ω .*= reshape(enorm, 1, :)

    # TODO check empty channels
    ω
end
