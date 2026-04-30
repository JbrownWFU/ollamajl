# Ollama Python Client Architecture & Julia Implementation Guide

This document outlines the architectural patterns and best practices observed in the official `ollama-python` package, providing a blueprint for building a similar client in Julia.

## 1. The "Front Door" (Public API)
In `ollama/__init__.py`, the library is exposed in two ways:
- **Object-Oriented:** Users can instantiate `Client()` or `AsyncClient()`.
- **Functional:** They export a default singleton instance of `Client`, allowing users to simply call `ollama.chat()` without managing client instances.

**Julia Implementation:**
You can achieve this by exporting a default constant `const DEFAULT_CLIENT = Client()` and defining top-level functions that delegate to it by default.

## 2. Client Architecture
The implementation uses a clear hierarchy to manage HTTP logic:
- **`BaseClient`**: Handles shared logic like host parsing (checking `OLLAMA_HOST`), setting default headers (User-Agent, Content-Type), and managing the underlying HTTP session (using `httpx`).
- **`Client` / `AsyncClient`**: These are thin wrappers around the base client that differentiate between synchronous and asynchronous behavior.
- **The `_request` Method**: This is the core communication method that:
    1. Sends the HTTP request.
    2. Handles errors (e.g., connection issues or 4xx/5xx responses).
    3. If `stream=True`, returns a generator (iterator) that yields response objects line-by-line.
    4. If `stream=False`, returns a single parsed response object.

## 3. Data Modeling (Types)
The library uses **Pydantic v2** (`ollama/_types.py`) for robust data validation and serialization:
- **Request Models**: They use inheritance (e.g., `BaseRequest` -> `GenerateRequest`) to share common fields like `model` or `options`.
- **Response Models**: Every API response is mapped to a class, providing autocomplete and type safety.
- **Subscriptable Models**: Models implement custom dictionary-like behavior (`model['field']`), allowing them to act as both objects and dictionaries based on user preference.

**Julia Implementation:**
Use `Base.@kwdef struct` for your types to allow keyword argument constructors. Pair this with `StructTypes.jl` and `JSON3.jl` for seamless and highly performant JSON mapping.

## 4. Advanced "Magic" (Utilities)
In `ollama/_utils.py`, a standout feature is `convert_function_to_tool`:
- It uses Python's `inspect` module to parse a function's signature and docstrings.
- It automatically generates the JSON schema required for Ollama's "tool calling" feature.

**Julia Implementation:**
You could leverage Julia's powerful introspection (like `methods(f)` or `Base.Method`) and Docstrings (`Base.Docs.doc`), or even write a macro to automatically generate tool schemas from Julia functions.

## 5. Streaming
- Streaming is handled by reading the HTTP response stream line-by-line.
- Each line is a JSON object, which is parsed into a specific response model and yielded to the caller.

**Julia Implementation:**
Use `Channels` or `Task` based streaming paired with HTTP.jl's streaming capabilities for an idiomatic way to handle continuous streams of data.

---

## Recommended Julia Scaffold

```julia
module Ollama

using HTTP
using JSON3
using StructTypes

# --- 1. Types ---
Base.@kwdef struct ChatMessage
    role::String
    content::String
end
StructTypes.StructType(::Type{ChatMessage}) = StructTypes.Struct()

Base.@kwdef struct ChatResponse
    model::String
    message::ChatMessage
    done::Bool
end
StructTypes.StructType(::Type{ChatResponse}) = StructTypes.Struct()


# --- 2. Client ---
mutable struct Client
    base_url::String
    timeout::Int
end

# Default values mirroring Python's logic
Client(; base_url=get(ENV, "OLLAMA_HOST", "http://127.0.0.1:11434"), timeout=120) = Client(base_url, timeout)
const DEFAULT_CLIENT = Client()


# --- 3. Request Handler ---
function request(c::Client, path::String, body::Dict; stream::Bool=false)
    url = "\$(c.base_url)\$path"
    headers = ["Content-Type" => "application/json"]
    
    # Example non-streaming handler
    if !stream
        res = HTTP.post(url, headers, JSON3.write(body))
        return JSON3.read(res.body, ChatResponse) # Assuming chat for example
    else
        # Implement streaming using HTTP.open and a Channel
        # ...
    end
end


# --- 4. Public API ---
function chat(model::String, messages::Vector{Dict}; client::Client=DEFAULT_CLIENT, stream::Bool=false, kwargs...)
    body = Dict(
        "model" => model,
        "messages" => messages,
        "stream" => stream,
        kwargs...
    )
    return request(client, "/api/chat", body; stream=stream)
end

export chat, Client

end # module Ollama
```
