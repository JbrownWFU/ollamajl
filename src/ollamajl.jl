module Ollamajl

include("utils.jl")

using HTTP, JSON3, StructTypes, BufferedStreams
using .Utils

export pprint,
    Client,
    generate,
    chat,
    Message,
    embed
#=
TODO
- [x] Implement streaming for all request types
- Fix Client instantiation
- Build tools objects
- Implement tool_calls in Message struct
- Implement julia func -> tool utility
- Ensure if a custom return schema is passed that it is able to be returned
- Fix OllamaError exception handling
- Fix Options and parameters in GenerateRequest
=#

# Jenson Brown - 05/2026

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
"""
    OllamaError(status, method, target, error)

Exception thrown when the Ollama API returns an error status.

# Fields
- `status::Int`: HTTP status code (e.g., 404, 500).
- `method::String`: HTTP method used (e.g., "POST").
- `target::String`: The API endpoint target.
- `error::String`: The error message returned by Ollama.
"""
struct OllamaError <: Exception
    status::Int
    method::String
    target::String
    error::String
end

function Base.showerror(io::IO, e::OllamaError)
    print(io, "OllamaError($(e.status)): $(e.error) [$(e.method) $(e.target)]")
end

function _error(err::HTTP.Exceptions.StatusError)
    body_str = String(err.response.body)
    
    parsed_error = try
        string(JSON3.read(body_str)[:error])
    catch
        body_str
    end

    return OllamaError(
        err.status,
        err.method,
        err.target,
        parsed_error
    )
end

"""
    Client(; host="http://127.0.0.1:11434", headers=nothing, timeout=nothing)

Represents a connection client to an Ollama server.

If `headers` are provided, they will overwrite the default headers.

# Arguments
- `host::String`: The base URL of the Ollama API.
- `headers::Vector{Pair{String, String}}`: Custom HTTP headers to include in requests.
- `timeout::Union{Int64, Nothing}`: Timeout in seconds for the requests.

# Examples
```julia-repl
julia> client = Client(host="http://127.0.0.1:11434")
Client("http://127.0.0.1:11434", ["Content-Type" => "application/json", ...], nothing)
```
"""
Base.@kwdef struct Client <: AbstractOllamaClient 
    host::String
    # headers::Dict{String, String}
    headers::Vector{Pair{String, String}}
    timeout::Union{Int64, Nothing}=nothing
end

function Client(;
        host::Union{String, Nothing} = "http://127.0.0.1:11434",
        headers::Union{Vector{Pair{String, String}}, Nothing} = nothing,
        timeout = nothing
)
    finalHeaders = [
        "Content-Type" => "application/json",
        "Accept" => "application/json",
        "User-Agent" => "ollama-julia/0.1.0"
    ]
    
    if !isnothing(headers)
        # Convert default to Dict to easily overwrite existing keys, then back to Vector
        header_dict = Dict(finalHeaders)
        for (k, v) in headers
            header_dict[k] = v
        end
        finalHeaders = [k => v for (k, v) in header_dict]
    end
    
    return Client(host, finalHeaders, timeout)
end

# Default client so the user doesn't have to instantiate a client for one shots
const DEFAULT_OLLAMA_CLIENT = Client()

# Tool Calling Structs
@Base.kwdef struct ToolFunction
    name::String
    description::String
    parameters::Dict{String, Any}
end

StructTypes.StructType(::Type{ToolFunction}) = StructTypes.Struct()

@Base.kwdef struct Tool
    type::String = "function"
    func::ToolFunction
end

StructTypes.StructType(::Type{Tool}) = StructTypes.Struct()
StructTypes.names(::Type{Tool}) = ((:func, :function),)

@Base.kwdef struct ToolCallFunction
    name::String
    arguments::Dict{String, Any}
end

StructTypes.StructType(::Type{ToolCallFunction}) = StructTypes.Struct()

@Base.kwdef struct ToolCall
    func::ToolCallFunction
end

StructTypes.StructType(::Type{ToolCall}) = StructTypes.Struct()
StructTypes.names(::Type{ToolCall}) = ((:func, :function),)

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

"""
    Options(; kwargs...)

Detailed generation options for Ollama models.

# Arguments
- `seed::Int`: Sets the random number seed to use for generation.
- `temperature::Float64`: The temperature of the model. Increasing the temperature will make the model answer more creatively.
- `top_k::Int`: Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers.
- `top_p::Float64`: Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text.
- `min_p::Float64`: Alternative to top_p.
- `stop::String`: Sets the stop sequences to use.
- `num_ctx::Int`: Sets the size of the context window used to generate the next token.
- `num_predict::Int`: Maximum number of tokens to predict when generating text.
"""
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
    logprobs::Union{Nothing, Bool}=nothing
    top_logprobs::Union{Nothing, Int}=nothing
end

StructTypes.StructType(::Type{GenerateRequest}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GenerateRequest}) = true
StructTypes.excludes(::Type{GenerateRequest}) = (:client,)

# TODO Deal with streamable vs non streamable
"""
    GenerateResponse

The response returned by the `generate` endpoint.

# Fields
- `model::String`: The model name.
- `created_at::String`: Timestamp of the response generation.
- `response::String`: The generated text response.
- `thinking::Union{Nothing, String}`: The reasoning/thinking process, if available.
- `done::Bool`: Whether the generation is complete.
- `done_reason::Union{Nothing, String}`: The reason generation stopped.
- `context::Union{Nothing, Vector{Int}}`: An encoding of the conversation used in this response.
- `total_duration::Int`: Time spent generating the response (nanoseconds).
- `load_duration::Int`: Time spent in nanoseconds loading the model.
- `prompt_eval_count::Int`: Number of tokens in the prompt.
- `prompt_eval_duration::Int`: Time spent in nanoseconds evaluating the prompt.
- `eval_count::Int`: Number of tokens in the response.
- `eval_duration::Int`: Time in nanoseconds spent generating the response.
- `logprobs::Union{Nothing, Vector{Logprobs}}`: Log probability of each token, if requested.
"""
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

"""
    Message(role, content[, images, tool_calls, thinking])

Represents a single message in a chat conversation.

# Arguments
- `role::String`: The role of the message creator ("user", "assistant", or "system").
- `content::String`: The text content of the message.
- `images::Vector{String}`: Optional list of base64 encoded images or file paths.
- `tool_calls::Vector{ToolCall}`: Optional list of tool calls made by the model.
- `thinking::String`: Optional thinking/reasoning string for supported models.

# Examples
```julia-repl
julia> msg = Message("user", "Hello!")
Message("user", "Hello!", nothing, nothing, nothing)
```
"""
@Base.kwdef struct Message
    role::String
    content::String = ""
    images::Union{Nothing, Vector{String}} = nothing
    tool_calls::Union{Nothing, Vector{ToolCall}} = nothing
    thinking::Union{Nothing, String} = nothing
end

# Outer constructors for convenience
Message(role::String, content::String) = Message(role=role, content=content)
Message(role::String, content::String, thinking::String) = Message(role=role, content=content, thinking=thinking)
Message(role::String, content::String, images::Vector{String}) = Message(role=role, content=content, images=parseImage.(images))
 
StructTypes.StructType(::Type{Message}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Message}) = true

@Base.kwdef struct ChatRequest <: AbstractRequest
    client::Client=DEFAULT_OLLAMA_CLIENT
    model::String
    # User can pass a dict or a built messages object
    messages::Union{AbstractDict, Vector{Message}}
    tools::Union{Nothing, Vector}=nothing
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
StructTypes.excludes(::Type{ChatRequest}) = (:client,)

"""
    ChatResponse

The response returned by the `chat` endpoint.

# Fields
- `model::String`: The model name.
- `created_at::String`: Timestamp of the response generation.
- `message::Message`: The chat message generated by the model.
- `done::Bool`: Whether the generation is complete.
- `done_reason::Union{Nothing, String}`: The reason generation stopped.
- `total_duration::Int`: Time spent generating the response (nanoseconds).
- `load_duration::Int`: Time spent in nanoseconds loading the model.
- `prompt_eval_count::Int`: Number of tokens in the prompt.
- `prompt_eval_duration::Int`: Time spent in nanoseconds evaluating the prompt.
- `eval_count::Int`: Number of tokens in the response.
- `eval_duration::Int`: Time in nanoseconds spent generating the response.
- `logprobs::Union{Nothing, Vector{Logprobs}}`: Log probability of each token, if requested.
"""
@Base.kwdef struct ChatResponse <: AbstractResponse
    model::String
    created_at::String
    message::Message
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
    client::Client=DEFAULT_OLLAMA_CLIENT
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
StructTypes.excludes(::Type{EmbedRequest}) = (:client,)

"""
    EmbedResponse

The response returned by the `embed` endpoint.

# Fields
- `model::String`: The model name.
- `embeddings::Vector{Vector{Float64}}`: The generated vector embeddings.
- `total_duration::Int`: Total time spent (nanoseconds).
- `load_duration::Int`: Time spent in nanoseconds loading the model.
- `prompt_eval_count::Int`: Number of tokens in the prompt.
"""
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

# Internal helper to merge keyword arguments into Options struct
function _mergeOptions(options::Union{Nothing, Options}; seed=nothing, temperature=nothing, top_k=nothing, top_p=nothing, min_p=nothing, stop=nothing, num_ctx=nothing, num_predict=nothing)
    if any(!isnothing, (seed, temperature, top_k, top_p, min_p, stop, num_ctx, num_predict))
        if isnothing(options)
            return Options(seed=seed, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, stop=stop, num_ctx=num_ctx, num_predict=num_predict)
        else
            return Options(
                seed = isnothing(seed) ? options.seed : seed,
                temperature = isnothing(temperature) ? options.temperature : temperature,
                top_k = isnothing(top_k) ? options.top_k : top_k,
                top_p = isnothing(top_p) ? options.top_p : top_p,
                min_p = isnothing(min_p) ? options.min_p : min_p,
                stop = isnothing(stop) ? options.stop : stop,
                num_ctx = isnothing(num_ctx) ? options.num_ctx : num_ctx,
                num_predict = isnothing(num_predict) ? options.num_predict : num_predict
            )
        end
    end
    return options
end

"""
    generate(; model, prompt=nothing, stream=true, kwargs...)

Generate a response for a given prompt using the specified model.

If `stream` is `true`, returns a `Channel{GenerateResponse}` that yields responses as they are generated. Otherwise, returns a single `GenerateResponse`.

# Arguments
- `client::Client`: The client instance (defaults to `DEFAULT_OLLAMA_CLIENT`).
- `model::String`: The name of the model to use (e.g., "llama3").
- `prompt::String`: The prompt to generate a response for.
- `system::String`: The system message to use.
- `template::String`: The prompt template to use.
- `stream::Bool`: Whether to stream the response. Defaults to `true`.
- `think::Union{Nothing, Bool, String}`: Control the thinking process output.
- `raw::Bool`: If `true`, no formatting will be applied to the prompt.
- `keep_alive::String`: Controls how long the model will stay loaded into memory.
- `options::Options`: Detailed generation options like `temperature`, `seed`, etc.

# Examples
```julia-repl
julia> response = generate(model="llama3", prompt="Why is the sky blue?", stream=false)
GenerateResponse(...)
```
"""
function generate(;
    client::Client=DEFAULT_OLLAMA_CLIENT,
    model::String,
    prompt::Union{Nothing, String} = nothing,
    suffix::Union{Nothing, String} = nothing,
    images::Union{Nothing, Vector{String}} = nothing,
    format::Union{Nothing, String} = nothing,
    system::Union{Nothing, String} = nothing,
    stream::Bool=true,
    think::Union{Nothing, Bool, String} = nothing,
    raw::Union{Nothing, Bool} = nothing,
    keep_alive::Union{Nothing, String} = nothing,
    options::Union{Nothing, Options} = nothing,
    logprobs::Union{Nothing, Bool} = nothing,
    top_logprobs::Union{Nothing, Int} = nothing,
    seed::Union{Nothing, Int}=nothing,
    temperature::Union{Nothing, Float64}=nothing,
    top_k::Union{Nothing, Int}=nothing,
    top_p::Union{Nothing, Float64}=nothing,
    min_p::Union{Nothing, Float64}=nothing,
    stop::Union{Nothing, String}=nothing,
    num_ctx::Union{Nothing, Int}=nothing,
    num_predict::Union{Nothing, Int}=nothing
)
    # Process images if any are passed
    processedImages = isnothing(images) ? nothing : parseImage.(images)
    
    # Merge direct option kwargs
    mergedOptions = _mergeOptions(
        options, 
        seed=seed, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        min_p=min_p, 
        stop=stop, 
        num_ctx=num_ctx, 
        num_predict=num_predict
    )

    # Build request object
    req = GenerateRequest(
        client=client,
        model=model,
        prompt=prompt,
        suffix=suffix,
        images=processedImages,
        format=format,
        system=system,
        stream=stream,
        think=think,
        raw=raw,
        keep_alive=keep_alive,
        options=mergedOptions,
        logprobs=logprobs,
        top_logprobs=top_logprobs
    )
    
    # Send request to internal router
    return _request(client, req)
end

"""
    chat(; model, messages, stream=true, kwargs...)

Generate the next chat message in a conversation.

If `stream` is `true`, returns a `Channel{ChatResponse}` that yields responses as they are generated. Otherwise, returns a single `ChatResponse`.

# Arguments
- `client::Client`: The client instance (defaults to `DEFAULT_OLLAMA_CLIENT`).
- `model::String`: The name of the model to use.
- `messages::Union{AbstractDict, Vector{Message}}`: The list of messages in the conversation.
- `tools::Vector`: List of tools the model may use.
- `format::String`: The format to return the response in (e.g., "json").
- `stream::Bool`: Whether to stream the response. Defaults to `true`.
- `keep_alive::String`: Controls how long the model will stay loaded into memory.

# Examples
```julia-repl
julia> messages = [Message("user", "Hello!")]
julia> response = chat(model="llama3", messages=messages, stream=false)
ChatResponse(...)
```
"""
function chat(;
    client::Client=DEFAULT_OLLAMA_CLIENT,
    model::String,
    messages::Union{AbstractDict, Vector{Message}},
    tools::Union{Nothing, Vector} = nothing,
    format::Union{Nothing, String} = nothing,
    stream::Bool=true,
    think::Union{Nothing, Bool, String} = nothing,
    keep_alive::Union{Nothing, String} = nothing,
    options::Union{Nothing, Options} = nothing,
    logprobs::Union{Nothing, Bool} = nothing,
    top_logprobs::Union{Nothing, Int} = nothing,
    seed::Union{Nothing, Int}=nothing,
    temperature::Union{Nothing, Float64}=nothing,
    top_k::Union{Nothing, Int}=nothing,
    top_p::Union{Nothing, Float64}=nothing,
    min_p::Union{Nothing, Float64}=nothing,
    stop::Union{Nothing, String}=nothing,
    num_ctx::Union{Nothing, Int}=nothing,
    num_predict::Union{Nothing, Int}=nothing
)
    # Process images in messages
    processedMessages = if messages isa Vector{Message}
        [Message(
            role=m.role,
            content=m.content,
            images=isnothing(m.images) ? nothing : parseImage.(m.images),
            tool_calls=m.tool_calls,
            thinking=m.thinking
        ) for m in messages]
    else
        messages
    end
    
    # Process tools
    processedTools = if !isnothing(tools)
        # Parse functions to Dicts or Tools, leave Dicts as is
        Any[t isa Tool ? t : buildTool(t) for t in tools]
    else
        nothing
    end
    
    mergedOptions = _mergeOptions(
        options, 
        seed=seed, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        min_p=min_p, 
        stop=stop, 
        num_ctx=num_ctx, 
        num_predict=num_predict
    )

    req = ChatRequest(
        client=client,
        model=model,
        messages=processedMessages,
        tools=processedTools,
        format=format,
        stream=stream,
        think=think,
        keep_alive=keep_alive,
        options=mergedOptions,
        logprobs=logprobs,
        top_logprobs=top_logprobs
    )
    
    return _request(client, req)
end

"""
    embed(; model, input, kwargs...)

Generate vector embeddings for the provided input.

# Arguments
- `client::Client`: The client instance (defaults to `DEFAULT_OLLAMA_CLIENT`).
- `model::String`: The name of the model to use.
- `input::Union{String, Vector{String}}`: The input text or list of texts to embed.
- `truncate::Bool`: Whether to truncate the input to fit the model's context window.
- `dimensions::Int`: The number of dimensions for the embedding.

# Examples
```julia-repl
julia> resp = embed(model="all-minilm", input="The sky is blue")
EmbedResponse(...)
```
"""
function embed(;
    client::Client=DEFAULT_OLLAMA_CLIENT,
    model::String,
    input::Union{String, Vector{String}},
    truncate::Bool=true,
    dimensions::Union{Nothing, Int} = nothing,
    options::Union{Nothing, Options} = nothing,
    keep_alive::Union{Nothing, String} = nothing,
    top_logprobs::Union{Nothing, Int} = nothing,
    seed::Union{Nothing, Int}=nothing,
    temperature::Union{Nothing, Float64}=nothing,
    top_k::Union{Nothing, Int}=nothing,
    top_p::Union{Nothing, Float64}=nothing,
    min_p::Union{Nothing, Float64}=nothing,
    stop::Union{Nothing, String}=nothing,
    num_ctx::Union{Nothing, Int}=nothing,
    num_predict::Union{Nothing, Int}=nothing
)
    
    mergedOptions = _mergeOptions(
        options, 
        seed=seed, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        min_p=min_p, 
        stop=stop, 
        num_ctx=num_ctx, 
        num_predict=num_predict
    )

    req = EmbedRequest(
        client=client,
        model=model,
        input=input,
        truncate=truncate,
        dimensions=dimensions,
        options=mergedOptions,
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
        if err isa HTTP.Exceptions.StatusError
            throw(_error(err))
        else
            rethrow(err)
        end
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

"""
    pprint(resp)

Pretty-print an Ollama response object with color formatting.

# Examples
```julia-repl
julia> resp = chat(model="llama3", messages=[Message("user", "Hi")], stream=false)
julia> pprint(resp)
*** 2026-05-05T12:00:00Z ***
llama3
[assistant]: Hello! How can I help you today?
```
"""
function pprint(resp::GenerateResponse) 
    println("─"^40)
    printstyled("Model: ", resp.model, "\n", color=:cyan, bold=true)
    printstyled("Time:  ", resp.created_at, "\n", color=:light_black)
    println("─"^40)
    println(resp.response)
    println("─"^40)
end

function pprint(resp::ChatResponse) 
    println("─"^40)
    printstyled("Model: ", resp.model, "\n", color=:cyan, bold=true)
    printstyled("Time:  ", resp.created_at, "\n", color=:light_black)
    println("─"^40)
    
    role = resp.message.role

    printstyled("$role:\n", color=:blue, bold=true)
    println(resp.message.content)
    println("─"^40)
end



end # module