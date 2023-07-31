const LANGUAGES::OrderedDict{String, String} = OrderedDict{String, String}(
    "en" => "english",
    "zh" => "chinese",
    "de" => "german",
    "es" => "spanish",
    "ru" => "russian",
    "ko" => "korean",
    "fr" => "french",
    "ja" => "japanese",
    "pt" => "portuguese",
    "tr" => "turkish",
    "pl" => "polish",
    "ca" => "catalan",
    "nl" => "dutch",
    "ar" => "arabic",
    "sv" => "swedish",
    "it" => "italian",
    "id" => "indonesian",
    "hi" => "hindi",
    "fi" => "finnish",
    "vi" => "vietnamese",
    "he" => "hebrew",
    "uk" => "ukrainian",
    "el" => "greek",
    "ms" => "malay",
    "cs" => "czech",
    "ro" => "romanian",
    "da" => "danish",
    "hu" => "hungarian",
    "ta" => "tamil",
    "no" => "norwegian",
    "th" => "thai",
    "ur" => "urdu",
    "hr" => "croatian",
    "bg" => "bulgarian",
    "lt" => "lithuanian",
    "la" => "latin",
    "mi" => "maori",
    "ml" => "malayalam",
    "cy" => "welsh",
    "sk" => "slovak",
    "te" => "telugu",
    "fa" => "persian",
    "lv" => "latvian",
    "bn" => "bengali",
    "sr" => "serbian",
    "az" => "azerbaijani",
    "sl" => "slovenian",
    "kn" => "kannada",
    "et" => "estonian",
    "mk" => "macedonian",
    "br" => "breton",
    "eu" => "basque",
    "is" => "icelandic",
    "hy" => "armenian",
    "ne" => "nepali",
    "mn" => "mongolian",
    "bs" => "bosnian",
    "kk" => "kazakh",
    "sq" => "albanian",
    "sw" => "swahili",
    "gl" => "galician",
    "mr" => "marathi",
    "pa" => "punjabi",
    "si" => "sinhala",
    "km" => "khmer",
    "sn" => "shona",
    "yo" => "yoruba",
    "so" => "somali",
    "af" => "afrikaans",
    "oc" => "occitan",
    "ka" => "georgian",
    "be" => "belarusian",
    "tg" => "tajik",
    "sd" => "sindhi",
    "gu" => "gujarati",
    "am" => "amharic",
    "yi" => "yiddish",
    "lo" => "lao",
    "uz" => "uzbek",
    "fo" => "faroese",
    "ht" => "haitian creole",
    "ps" => "pashto",
    "tk" => "turkmen",
    "nn" => "nynorsk",
    "mt" => "maltese",
    "sa" => "sanskrit",
    "lb" => "luxembourgish",
    "my" => "myanmar",
    "bo" => "tibetan",
    "tl" => "tagalog",
    "mg" => "malagasy",
    "as" => "assamese",
    "tt" => "tatar",
    "haw" => "hawaiian",
    "ln" => "lingala",
    "ha" => "hausa",
    "ba" => "bashkir",
    "jw" => "javanese",
    "su" => "sundanese",
)

const SPECIALS::Vector{String} = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    ["<|$lang|>" for lang in keys(LANGUAGES)]...,
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
    ["<|$(@sprintf("%.2f", i * 0.02))|>" for i in 0:1500]...,
]

function prep_ranks(tokens_file::String)
    ranks = Dict(
        Base64.base64decode(token) => parse(Int, rank) for (token, rank) in
            (split(line) for line in readlines(tokens_file)))
    n_vocab = length(ranks)
    special_tokens = Dict(zip(SPECIALS, n_vocab:(n_vocab + length(SPECIALS) - 1)))
    n_vocab += length(special_tokens)
    return ranks, special_tokens, n_vocab
end

struct BPE
    pattern::Regex
    mergeable_ranks::Dict{Vector{UInt8}, Int64}
    decoder::Dict{Int64, Vector{UInt8}}
    special_tokens::Dict{String, Int64}
    special_decoder::Dict{Int64, String}
end

function BPE(
    mergeable_ranks::Dict{Vector{UInt8}, Int64};
    special_tokens::Dict{String, Int64},
    pattern::Regex = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
)
    decoder = Dict{Int64, Vector{UInt8}}(
        rank => byte for (byte, rank) in mergeable_ranks)
    special_decoder = Dict{Int64, String}(
        rank => token for (token, rank) in special_tokens)
    BPE(pattern, mergeable_ranks, decoder, special_tokens, special_decoder)
end

function (enc::BPE)(text::String)
    bpe_tokens = Int64[]

    in_token = false
    special_token = ""

    for match in eachmatch(enc.pattern, text)
        # Detect special tokens.
        if in_token
            special_token = special_token * match.match
            if match.match == "|>"
                in_token = false
                push!(bpe_tokens, enc.special_tokens[special_token])
            end
            continue
        elseif !in_token && match.match == "<|"
            in_token = true
            special_token = "$(match.match)"
            continue
        end

        append!(bpe_tokens, bpe_encode(enc, codeunits(match.match)))
    end
    bpe_tokens
end

function bpe_encode(enc::BPE, x::Base.CodeUnits{UInt8})
    parts = [[xi] for xi in x]
    while true
        # Iterate over all pairs and find the pair we want to merge the most.
        min_idx, min_rank = -1, -1
        for (i, pair) in enumerate(zip(@view(parts[1:end - 1]), @view(parts[2:end])))
            rank = get(enc.mergeable_ranks, [pair[1]..., pair[2]...], nothing)
            if !isnothing(rank) && (min_rank == -1 || rank < min_rank)
                min_idx, min_rank = i, rank
            end
        end
        # If there were no pairs we could merge, we're done.
        min_rank == -1 && break
        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = [
            parts[1:min_idx - 1]...,
            [parts[min_idx]..., parts[min_idx + 1]...],
            parts[min_idx + 2:end]...]
    end
    [enc.mergeable_ranks[p] for p in parts]
end

function decode(enc::BPE, tokens::Vector{Int}; include_specials::Bool = true)
    bytes = UInt8[]
    for t in tokens
        v = get(enc.decoder, t, nothing)
        if isnothing(v)
            include_specials && append!(bytes, codeunits(enc.special_decoder[t]))
        else
            append!(bytes, v)
        end
    end
    String(bytes)
end

# function clean_text(text::String)
#     text = lowercase(text)
#     text = strip(text)
#     text = replace(text, r"\s+" => " ")
#     strip(text)
# end
