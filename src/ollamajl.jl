module Ollamajl

using HTTP, JSON3

#=
TODO
- Implement generate endpoint
    - Built structs for requests
- Implement generate response struct
=#

# Ollama API wrapper for Julia
# Based on official Ollama Python library
# https://github.com/ollama/ollama-python

# For more information see Ollama API documentation
# https://docs.ollama.com/api/introduction

# ***** 
# Structs
# ***** 

abstract type AbstractOllamaClient end

abstract type AbstractRequest end
abstract type AbstractStreamableRequest <: AbstractRequest end
# /generate
abstract type AbstractGenerateRequest <: AbstractStreamableRequest end

# Ollama client
Base.@kwdef struct Client <: AbstractOllamaClient 
    host::String
    headers::Dict{String, String}
    timeout::Union{Int64, Nothing}
end

function Client(;
        host::Union{String, Nothing} = "http://127.0.0.1:11434",
        headers::Union{AbstractDict, Nothing} = Dict{String, String}(),
        timeout = nothing
)
    # Default headers
    finalHeaders = Dict{String, String}(
        "Content-Type" => "application/json",
        "Accept" => "application/json",
        "User-Agent" => "ollama-julia/0.1.0"
    )
    
    # Merge user headers if any are passed
    if !isnothing(headers)
        for (k,v) in headers
            finalHeaders[string(k) |> lowercase] = string(v)
        end
    end
    
    return Client(host, finalHeaders, timeout)
end

# Default client so the user doesn't have to instantiate a client for one shots
const DEFAULT_OLLAMA_CLIENT = Client()

@Base.kwdef struct GenerateRequest <: AbstractGenerateRequest
    model::String
end

# ***** 
# Endpoints
# *****


end # module