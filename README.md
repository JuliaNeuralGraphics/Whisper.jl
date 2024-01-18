# Whisper.jl

Port of OpenAI's [whisper](https://github.com/openai/whisper) model.

## Installation

Clone the repository and instantiate it.

## Usage

1. Specify GPU backend in `LocalPreferences.toml` file (either `AMDGPU` or `CUDA`) if using GPU for inference.
2. Run the model:
```julia
julia> using AMDGPU # If using AMDGPU for inference.
julia> using CUDA   # If using CUDA for inference.
 
julia> using Whisper, Flux

# GPU inference at FP16 precision.
julia> Whisper.transcribe(
    "./input.flac", "./output.srt";
    model_name="tiny.en", dev=gpu, precision=f16)

# CPU inference.
julia> Whisper.transcribe(
    "./input.flac", "./output.srt";
    model_name="tiny.en", dev=cpu, precision=f32)
```

**Multilingual support**

To perform transcribtion from non-English language,
specify `language` argument (optional) and drop `.en` from the model name.

```julia
julia> Whisper.transcribe(
    "ukrainian-sample.flac", "./output.srt";
    model_name="medium", language="ukrainian", dev=cpu, precision=f32)
```

To see what languages are supported, execute:
```julia
julia> values(Whisper.LANGUAGES)
```

## Details

- Supported input file: `.flac` with 1 channel and 16k sample rate.
- Other input files are converted to it using `ffmpeg` which must be installed on your system and accessible from PATH.

## TODO

- Beam search decoder.
- Streaming support.
