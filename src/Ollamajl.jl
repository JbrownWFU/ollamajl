module Ollamajl

# PUBLIC
# Ollama julia package
# Currently supports chat, generate endpoints with limited arguments (TODO)
# Chat uses a message history while generate returns a single completion.
# Next: Adding embeddings endpoints for RAG usage

using HTTP, JSON3

export OllamaInstance, generate, chat, clearChat, printChat

struct OllamaInstance
    url::String
    model::String
    messages::Vector{Dict}
end

function initInstance(url::String; model::String)
    messages = []
    return OllamaInstance(
        url,
        model,
        messages
    )
end

# Generate a response
function generate(ollama::OllamaInstance, message::String; stream=false, format="")
    printstyled("=> $message\n"; color=:yellow)

    url = ollama.url * "/api/generate"
    
    body = Dict(
        "model" => ollama.model,
        "prompt" => message,
        "stream" => stream
    )
    
    if format != ""
        body["format"] = format
    end
    
    req = HTTP.request("POST", url, [], JSON3.write(body))
    resp = String(req.body) |> JSON3.read

    printstyled("<= $(resp.response)\n"; color=:green)
    
    return resp
end
    
# Chat with message history
function chat(ollama::OllamaInstance, message::String; stream=false, format="")
    printstyled("=> $message\n"; color=:yellow)

    url = ollama.url * "/api/chat"
    
    # Update messsage history
    push!(ollama.messages, Dict("role" => "user", "content" => message))
    
    body = Dict(
        "model" => ollama.model,
        "messages" => ollama.messages,
        "stream" => stream
    ) 
    
    if format != ""
        body["format"] = format
    end
    
    req = HTTP.request("POST", url, [], JSON3.write(body))
    resp = String(req.body) |> JSON3.read
    
    # Push response to conversation history
    push!(ollama.messages, Dict("role" => resp.message.role, "content" => resp.message.content))
    
    printstyled("<= $(resp.message.content)\n"; color=:green)
    
    return resp
end

# Helpers

# Clear ollama chat messsage history
function clearChat(ollama::OllamaInstance)
    resize!(ollama.messages, 0)
end

# Pretty print ollama chat messsage history
function printChat(ollama::OllamaInstance)
    for msg in ollama.messages
        if msg["role"] == "user"
            printstyled("<> $(msg["content"])\n"; color=:yellow)
        else
            printstyled("<> $(msg["content"])\n"; color=:green)
        end
    end
end

# Set first message system prompt for conversations 
function setSysPrompt(ollama::OllamaInstance, prompt::String)
    ollama.messages = insert!(ollama.messages, 1, prompt)
end

end