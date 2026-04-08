module Ollamajl

using HTTP, JSON3, StructTypes

export OllamaInstance, Options, initInstance, generate, chat, embed, tags, Tool, ToolFunction, Message

struct OllamaInstance
    url::String
    # messages::Vector{Dict{String, Any}}

    function OllamaInstance(url::String)
        # return new(url, Dict{String, Any}[])
        return new(url)
    end
end

# Options objects
# ------------

# Sub options object for requests
Base.@kwdef struct SubOptions
    seed::Union{Int, Nothing} = nothing
    temperature::Union{Float64, Nothing} = nothing
    top_k::Union{Int, Nothing} = nothing
    top_p::Union{Float64, Nothing} = nothing
    min_p::Union{Float64, Nothing} = nothing
    stop::Union{String, Nothing} = nothing
    num_ctx::Union{Int, Nothing} = nothing
    num_predict::Union{Int, Nothing} = nothing
end

StructTypes.StructType(::Type{SubOptions}) = StructTypes.Struct()
StructTypes.omitempties(::Type{SubOptions}) = true

# Tool Structs
# ------------

# Tool object function parameter
Base.@kwdef struct ToolFunction
    name::String
    parameters::Dict{String, Any}
    description::Union{String, Nothing} = nothing
end

StructTypes.StructType(::Type{ToolFunction}) = StructTypes.Struct()

# Tool object
Base.@kwdef struct Tool
    type::String = "function"
    func::ToolFunction
end

StructTypes.StructType(::Type{Tool}) = StructTypes.Struct()
StructTypes.names(::Type{Tool}) = ((:func, :function),)

# Ollama tool call 
Base.@kwdef struct ToolCallFunction
    name::String
    arguments::Dict{String, Any}
end

StructTypes.StructType(::Type{ToolCallFunction}) = StructTypes.Struct()

Base.@kwdef struct ToolCall
    func::ToolCallFunction
end

StructTypes.StructType(::Type{ToolCall}) = StructTypes.Struct()
StructTypes.names(::Type{ToolCall}) = ((:func, :function),)

# Message object
# ------------

# Message object
Base.@kwdef mutable struct Message
    role::String
    content::String
    images::Union{Vector{String}, Nothing} = nothing
    tool_calls::Union{Vector{ToolCall}, Nothing} = nothing
end

# Messages constructor for JSON serialization
function Messages(messages::Union{Dict, NamedTuple})
    msgs = JSON3.read(JSON.write(messages), Message) 
    return msgs
end

# Universal options struct
Base.@kwdef mutable struct Options
    # Global
    model::String
    stream::Union{Bool, Nothing} = nothing
    format::Any = nothing
    # TODO support for dict, JSON input
    options::Union{SubOptions, Nothing} = nothing
    # options::Union{JSON3.Object, Nothing} = nothing
    system::Union{String, Nothing} = nothing
    think::Union{Union{Bool, String}, Nothing} = nothing
    keep_alive::Union{String, Nothing} = nothing
    logprobs::Union{Bool, Nothing} = nothing
    top_logprobs::Union{Int, Nothing} = nothing
    # Generate
    prompt::Any = nothing
    suffix::Union{String, Nothing} = nothing
    images::Union{Vector{String}, Nothing} = nothing
    # Chat
    # TODO support for dict, JSON input
    messages::Union{AbstractVector, Nothing} = nothing
    # TODO support for dict, JSON input
    tools::Union{AbstractVector, Nothing} = nothing
    # Embed
    input::Union{String, Nothing} = nothing
    truncate::Union{Bool, Nothing} = nothing
    dimensions::Union{Int, Nothing} = nothing
end

StructTypes.StructType(::Type{Options}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Options}) = true

# Messaging endpoints
# -----

# Generates a response for the provided prompt
function generate(ollama::OllamaInstance, opts::Options)
    url = ollama.url * "/api/generate"
    body = JSON3.write(opts)
    req = HTTP.request("POST", url, [], body)

    return req
end

# Kwarg handler
generate(ollama; kwargs...) = generate(ollama, Options(; kwargs...))

# Convenience with kwargs
# function generate(ollama::OllamaInstance; kwargs...)
#     opts = Options(; kwargs...)
#     return generate(ollama, opts)
# end

# Generate the next chat message in a conversation between a user and an assistant
function chat(ollama::OllamaInstance, opts::Options)
    url = ollama.url * "/api/chat"
    body = JSON3.write(opts)
    req = HTTP.request("POST", url, [], body)

    return req
end

# Kwarg handler
chat(ollama; kwargs...) = chat(ollama, Options(; kwargs...))

# function chat(ollama::OllamaInstance; kwargs...)
#     opts = Options(; kwargs...)
#     return chat(ollama, opts)
# end

# Creates vector embeddings representing the input text
function embed(ollama::OllamaInstance, opts::Options)
    url = ollama.url * "/api/embed"
    body = JSON3.write(opts)
    req = HTTP.request("POST", url, [], body)

    return req
end

function embed(ollama::OllamaInstance; kwargs...)
    opts = Options(; kwargs...)
    return embed(ollama, opts)
end

# Server endpoints
# -----

# Fetch a list of models and their details 
function tags(ollama::OllamaInstance)
    url = ollama.url * "/api/tags"
    req = HTTP.request("GET", url, [], body)

    return req
end

# Retrieve a list of models that are currently running
function ps(ollama::OllamaInstance)
    url = ollama.url * "/api/ps"
    req = HTTP.request("GET", url, [], body)

    return req
end

end # module