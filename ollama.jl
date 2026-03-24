module Ollama 

using HTTP, JSON3

export OllamaInstance

struct OllamaInstance
    url::String
    model::String
    messages::Vector{Dict}
end

function initInstance(url::String, model::String)
    messages = []
    return OllamaInstance(
        url,
        model,
        messages
    )
end

# Generate a response
function generate(ollama::OllamaInstance, message::String; stream=false)
    printstyled("=> $message\n"; color=:yellow)

    url = ollama.url * "/api/generate"
    
    body = Dict(
        "model" => ollama.model,
        "prompt" => message,
        "stream" => stream
    ) |> JSON3.write
    
    req = HTTP.request("POST", url, [], body)
    resp = String(req.body) |> JSON3.read

    printstyled("<= $(resp.response)\n"; color=:green)
    
    return resp
end
    
# Chat with message history
function chat(ollama::OllamaInstance, message::String; stream=false)
    printstyled("=> $message\n"; color=:yellow)

    url = ollama.url * "/api/chat"
    
    # Update messsage history
    push!(ollama.messages, Dict("role" => "user", "content" => message))
    
    body = Dict(
        "model" => ollama.model,
        "messages" => ollama.messages,
        "stream" => stream
    ) |> JSON3.write
    
    req = HTTP.request("POST", url, [], body)
    resp = String(req.body) |> JSON3.read
    
    # Push response to conversation history
    push!(ollama.messages, Dict("role" => resp.message.role, "content" => resp.message.content))
    
    printstyled("<= $(resp.message.content)\n"; color=:green)
    
    return resp
end

function clearChat(ollama::OllamaInstance)
    resize!(ollama.messages, 0)
end

function printChat(ollama::OllamaInstance)
    for msg in ollama.messages
        if msg["role"] == "user"
            printstyled("<> $(msg["content"])\n"; color=:yellow)
        else
            printstyled("<> $(msg["content"])\n"; color=:green)
        end
    end
end


end