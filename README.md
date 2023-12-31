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

## Details

- Supported input file: `.flac` with 1 channel and 16k sample rate.
- Other input files are converted to it using `ffmpeg` which must be installed on your system and accessible from PATH.
- Only english models are supported right now.
- Supported model names: `tiny.en`, `base.en`, `small.en`, `medium.en`.

## TODO

- Beam search decoder.
- Multilingual support.
- Streaming support.
