module Ollamajl

include("utils.jl")

using HTTP, JSON3, StructTypes, BufferedStreams

export pprint,
    Client,
    generate,
    chat,
    Message,
    embed

#=
TODO
- [x] Implement streaming for all request types
- Implement base64 encoding for images
- Build tools objects
- Implement julia func -> tool utility
- Ensure if a custom return schema is passed that it is able to be returned
- Fix OllamaError exception handling
- Fix Options and parameters in GenerateRequest
- Handle streamable vs non-streamable responses in GenerateResponse
- Implement tool_calls in Message struct
=#

# Ollama API wrapper for Julia
# Based on official Ollama Python library
# https://github.com/ollama/ollama-python

# For more information see Ollama API documentation
# https://docs.ollama.com/api/introduction

# ***** 
# Structs & Exceptions
# ***** 

abstract type AbstractOllamaClient end

abstract type AbstractRequest end

abstract type AbstractResponse end

# TODO fix
struct OllamaError <: Exception
    status::Int16
    method::String
    target::String
    error::String
end

function _error(err::HTTP.Exceptions.StatusError)
    body = String(err.response.body) |> 
    JSON3.read

    return OllamaError(
        err.status,
        err.method,
        err.target,
        body
    )
end

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

# Options parameter object
struct Top_logprobs
    token::String
    logprob::Float64
    bytes::Union{Nothing, Vector{Int}}
end

StructTypes.StructType(::Type{Top_logprobs}) = StructTypes.Struct()

struct Logprobs 
    token::String
    logprob::Float64
    bytes::Union{Nothing, Vector{Int}}
    top_logprobs::Union{Nothing, Vector{Top_logprobs}}
end

StructTypes.StructType(::Type{Logprobs}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Logprobs}) = true

@Base.kwdef struct Options 
    seed::Union{Nothing, Int}=nothing
    temperature::Union{Nothing, Float64}=nothing
    top_k::Union{Nothing, Int}=nothing
    top_p::Union{Nothing, Float64}=nothing
    min_p::Union{Nothing, Float64}=nothing
    stop::Union{Nothing, String}=nothing
    num_ctx::Union{Nothing, Int}=nothing
    num_predict::Union{Nothing, Int}=nothing
end

StructTypes.StructType(::Type{Options}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Options}) = true

@Base.kwdef struct GenerateRequest <: AbstractRequest
    client::Client=DEFAULT_OLLAMA_CLIENT
    model::String
    prompt::Union{Nothing, String}=nothing
    suffix::Union{Nothing, String}=nothing
    # TODO fix options
    options::Union{Nothing, Options}=nothing
    images::Union{Nothing, Vector{String}}=nothing
    # TODO support abstractDict / JSON schema for format
    format::Union{Nothing, String, AbstractDict}=nothing
    system::Union{Nothing, String}=nothing
    stream::Bool=true
    think::Union{Nothing, Bool, String}=nothing
    raw::Union{Nothing, Bool}=nothing
    keep_alive::Union{Nothing, String}=nothing
    # TODO add Options
    logprobs::Union{Nothing, Bool}=nothing
    top_logprobs::Union{Nothing, Int}=nothing
end

StructTypes.StructType(::Type{GenerateRequest}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GenerateRequest}) = true

# TODO Deal with streamable vs non streamable
@Base.kwdef struct GenerateResponse <: AbstractResponse
    model::String
    created_at::String
    response::String
    # Not all models have thinking
    thinking::Union{Nothing, String} = nothing
    done::Bool
    done_reason::Union{Nothing, String} = nothing
    context::Union{Nothing, Vector{Int}} = nothing
    total_duration::Union{Nothing, Int} = nothing
    load_duration::Union{Nothing, Int} = nothing
    prompt_eval_count::Union{Nothing, Int} = nothing
    prompt_eval_duration::Union{Nothing, Int} = nothing
    eval_count::Union{Nothing, Int} = nothing
    eval_duration::Union{Nothing, Int} = nothing
    # Logprobs is an array of objects
    logprobs::Union{Nothing, Vector{Logprobs}} = nothing
end

# JSON3 serialization
StructTypes.StructType(::Type{GenerateResponse}) = StructTypes.Struct()

# Messages / Chat endpoint
@Base.kwdef struct Message
    role::String
    content::String
    thinking::Union{Nothing, String} = nothing
    # TODO implement tool_calls vector / object
    
    Message() = new(role, content, nothing)
    Message(role, content) = new(role, content, nothing)
    Message(role, content, thinking) = new(role, content, thinking)
end
 
StructTypes.StructType(::Type{Message}) = StructTypes.Struct()

@Base.kwdef struct ChatRequest <: AbstractRequest
    model::String
    # User can pass a dict or a built messages object
    messages::Union{AbstractDict, Vector{Message}}
    # TODO build tools object and support abstractDict
    format::Union{Nothing, String, AbstractDict}=nothing
    options::Union{Nothing, Options}=nothing
    think::Union{Nothing, Bool, String}=nothing
    stream::Bool=true
    keep_alive::Union{Nothing, String}=nothing
    logprobs::Union{Nothing, Bool}=nothing
    top_logprobs::Union{Nothing, Int}=nothing
end

StructTypes.StructType(::Type{ChatRequest}) = StructTypes.Struct()
StructTypes.omitempties(::Type{ChatRequest}) = true

@Base.kwdef struct ChatResponse <: AbstractResponse
    model::String
    created_at::String
    message::Message
    # Not all models have thinking
    done::Bool
    done_reason::Union{Nothing, String} = nothing
    total_duration::Union{Nothing, Int} = nothing
    load_duration::Union{Nothing, Int} = nothing
    prompt_eval_count::Union{Nothing, Int} = nothing
    prompt_eval_duration::Union{Nothing, Int} = nothing
    eval_count::Union{Nothing, Int} = nothing
    eval_duration::Union{Nothing, Int} = nothing
    # Logprobs is an array of objects
    logprobs::Union{Nothing, Vector{Logprobs}} = nothing
end

StructTypes.StructType(::Type{ChatResponse}) = StructTypes.Struct()

@Base.kwdef struct EmbedRequest <: AbstractRequest
    model::String
    input::Union{String, Vector{String}}
    truncate::Bool=true
    dimensions::Union{Nothing, Int}=nothing
    options::Union{Nothing, Options}=nothing
    keep_alive::Union{Nothing, String}=nothing
    top_logprobs::Union{Nothing, Int}=nothing
end

StructTypes.StructType(::Type{EmbedRequest}) = StructTypes.Struct()
StructTypes.omitempties(::Type{EmbedRequest}) = true

@Base.kwdef struct EmbedResponse <: AbstractResponse
    model::String
    embeddings::Union{Nothing, Vector{Vector{Float64}}} = nothing
    total_duration::Union{Nothing, Int} = nothing
    load_duration::Union{Nothing, Int} = nothing
    prompt_eval_count::Union{Nothing, Int} = nothing
end

StructTypes.StructType(::Type{EmbedResponse}) = StructTypes.Struct()

# Helpers for request routing
# Route returned object based on request typing
responseType(::GenerateRequest) = GenerateResponse
responseType(::ChatRequest) = ChatResponse
responseType(::EmbedRequest) = EmbedResponse

# Get endpoint
endpoint(::GenerateRequest) = "/api/generate"
endpoint(::ChatRequest) = "/api/chat"
endpoint(::EmbedRequest) = "/api/embed"

# ***** 
# Endpoints
# *****

# Generates a response for the provided prompt.
# https://docs.ollama.com/api/generate
function generate(;
    client::Client=DEFAULT_OLLAMA_CLIENT,
    model::String,
    prompt::Union{Nothing, String} = nothing,
    suffix::Union{Nothing, String} = nothing,
    # TODO add support for Base64 encoded images
    # Allow input of paths?
    images::Union{Nothing, Vector{String}} = nothing,
    format::Union{Nothing, String} = nothing,
    system::Union{Nothing, String} = nothing,
    stream::Bool=true,
    think::Union{Nothing, Bool, String} = nothing,
    raw::Union{Nothing, Bool} = nothing,
    keep_alive::Union{Nothing, String} = nothing,
    options::Union{Nothing, Options} = nothing,
    logprobs::Union{Nothing, Bool} = nothing,
    top_logprobs::Union{Nothing, Int} = nothing
)
    # Build request object
    req = GenerateRequest(
        model=model,
        prompt=prompt,
        suffix=suffix,
        images=images,
        format=format,
        system=system,
        stream=stream,
        think=think,
        raw=raw,
        keep_alive=keep_alive,
        options=options,
        logprobs=logprobs,
        top_logprobs=top_logprobs
    )
    
    # Send request to internal router
    return _request(client, req)
end

# Generate the next chat message in a conversation between a user and an assistant
# https://docs.ollama.com/api/chat
function chat(;
    client::Client=DEFAULT_OLLAMA_CLIENT,
    model::String,
    messages::Union{AbstractDict, Vector{Message}},
    # TODO add tools object
    format::Union{Nothing, String} = nothing,
    stream::Bool=true,
    think::Union{Nothing, Bool, String} = nothing,
    keep_alive::Union{Nothing, String} = nothing,
    options::Union{Nothing, Options} = nothing,
    logprobs::Union{Nothing, Bool} = nothing,
    top_logprobs::Union{Nothing, Int} = nothing
)
    req = ChatRequest(
        model=model,
        messages=messages,
        format=format,
        stream=stream,
        think=think,
        keep_alive=keep_alive,
        options=options,
        logprobs=logprobs,
        top_logprobs=top_logprobs
    )
    
    return _request(client, req)
end

# Creates vector embeddings representing the input text
# https://docs.ollama.com/api/embed

function embed(;
    client::Client=DEFAULT_OLLAMA_CLIENT,
    model::String,
    input::Union{String, Vector{String}},
    truncate::Bool=true,
    dimensions::Union{Nothing, Int} = nothing,
    options::Union{Nothing, Options} = nothing,
    keep_alive::Union{Nothing, String} = nothing,
    top_logprobs::Union{Nothing, Int} = nothing,
)

    req = EmbedRequest(
        model=model,
        input=input,
        truncate=truncate,
        dimensions=dimensions,
        options=options,
        keep_alive=keep_alive,
        top_logprobs=top_logprobs
    )
    
    return _request(client, req)
end

# Internal request endpoint
function _request(client::Client, req::AbstractRequest)
    url = client.host * endpoint(req)
    body = JSON3.write(req)
    
    streaming = hasproperty(req, :stream) && req.stream

    try
        if streaming
            return Channel{responseType(req)}() do channel
                HTTP.open("POST", url, client.headers) do stream
                    write(stream, body)
                    HTTP.closewrite(stream)
                    HTTP.startread(stream)
                    try
                        bufferedStream = BufferedInputStream(stream)
                        for line in eachline(bufferedStream)
                            if !isempty(line)
                                push!(channel, JSON3.read(line, responseType(req)))
                            end
                        end
                    catch err
                        if !(err isa EOFError)
                            rethrow(err)
                        end
                    end
                end
            end
        else
            resp = HTTP.post(
                url,
                client.headers,
                body
            ) 
            
            return JSON3.read(resp.body, responseType(req))
        end
    catch err
        @error err exception=(err, catch_backtrace())
        return nothing
    end
end

# ***** 
# Helpers
# *****

# Overload show for printing
function Base.show(io::IO, resp::GenerateResponse)
    vals = ["$field = $(isdefined(resp, field) ? getfield(resp, field) : "#undef")" for field in fieldnames(typeof(resp))]
    
    output = "GenerateResponse($(join(vals, ", ")))"
    
    print(io, output)
end

function Base.show(io::IO, resp::ChatResponse)
    vals = ["$field = $(isdefined(resp, field) ? getfield(resp, field) : "#undef")" for field in fieldnames(typeof(resp))]
    
    output = "ChatResponse($(join(vals, ", ")))"
    
    print(io, output)
end

# Pretty print response
function pprint(resp::GenerateResponse) 
    printstyled("*** $(resp.created_at) ***\n$(resp.model)\n$(resp.response)"; color=:cyan)
end

function pprint(resp::ChatResponse) 
    printstyled("*** $(resp.created_at) ***\n$(resp.model)\n[$(resp.message.role)]: $(resp.message.content)"; color=:cyan)
end



end # module