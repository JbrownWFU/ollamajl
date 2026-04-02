module Ollamajl

using HTTP, JSON3, StructTypes

export OllamaInstance, GenerateOptions, initInstance, generate, chat

struct OllamaInstance
    url::String
    messages::Vector{Dict{String, Any}}
end

function initInstance(url::String)
    return OllamaInstance(url, Dict{String, Any}[])
end

Base.@kwdef struct GenerateOptions
    model::String
    prompt::Any = nothing
    stream::Union{Bool, Nothing} = nothing
    format::Union{String, Nothing} = nothing
    suffix::Union{String, Nothing} = nothing
end

StructTypes.StructType(::Type{GenerateOptions}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GenerateOptions}) = true

function generate(ollama::OllamaInstance, opts::GenerateOptions)
    url = ollama.url * "/api/generate"
    body = JSON3.write(opts)
    req = HTTP.request("POST", url, [], body)

    return req
end

# Convenience with kwargs
function generate(ollama::OllamaInstance; kwargs...)
    opts = GenerateOptions(; kwargs...)
    return generate(ollama, opts)
end

function chat(ollama::OllamaInstance, opts::GenerateOptions)
    url = ollama.url * "/api/chat"
    body = JSON3.write(opts)
    req = HTTP.request("POST", url, [], body)

    return req
end

# Convenience with kwargs
function chat(ollama::OllamaInstance; kwargs...)
    opts = GenerateOptions(; kwargs...)
    return generate(ollama, opts)
end

end