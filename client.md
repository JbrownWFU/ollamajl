# Julia Ollama Client: Implementation Guide

This document summarizes the architectural patterns found in the official `ollama-python` library and maps them to an idiomatic implementation strategy for Julia.

## 1. Type System Architecture

### Python Implementation (`_types.py`)
The Python library uses **Pydantic** with a deep inheritance hierarchy to share common fields across different API endpoints:

*   **Foundation:** `SubscriptableBaseModel` provides dictionary-like access to object fields.
*   **Requests:** `BaseRequest` (model) → `BaseStreamableRequest` (stream) → `BaseGenerateRequest` (options, format, keep_alive) → `ChatRequest`/`GenerateRequest`.
*   **Responses:** `BaseGenerateResponse` contains all performance metrics (duration, token counts) and is inherited by `ChatResponse` and `GenerateResponse`.

### Julia Mapping Strategy
Since Julia does not support concrete inheritance (one struct inheriting fields from another), the following approach is recommended:

*   **Abstract Hierarchy:** Use abstract types for dispatching.
    ```julia
    abstract type AbstractOllamaRequest end
    abstract type AbstractStreamableRequest <: AbstractOllamaRequest end
    ```
*   **Field Injection (Macros):** To avoid manual repetition of common fields (like performance metrics in responses), use a macro.
    ```julia
    macro response_metrics()
        quote
            model::Union{String, Nothing}
            total_duration::Union{Int, Nothing}
            load_duration::Union{Int, Nothing}
            # ... etc
        end
    end

    struct ChatResponse
        @response_metrics()
        message::Message
    end
    ```

---

## 2. JSON Serialization & Validation

### Python Implementation
Pydantic handles serialization via `model_dump(exclude_none=True)`. This is critical because the Ollama server expects optional fields to be omitted rather than sent as `null`.

### Julia Mapping Strategy
Use **`JSON3.jl`** and **`StructTypes.jl`**.
*   Define `StructTypes.StructType(::Type{T}) = StructTypes.Struct()` for each type.
*   To emulate `exclude_none`, use `JSON3.write` with appropriate filtering or ensure `Nothing` fields are handled by the serialization layer.

---

## 3. Client Implementation (`_client.py`)

### Python Implementation
Python implements two distinct classes: `Client` (synchronous using `httpx.Client`) and `AsyncClient` (asynchronous using `httpx.AsyncClient`).

*   **`_request` Abstraction:** Both clients use a private `_request` method that branches logic based on the `stream` boolean.
*   **Streaming:** Returns an `Iterator` (sync) or `AsyncIterator` (async) that yields parsed objects line-by-line.

### Julia Mapping Strategy
In Julia, a single client can handle both synchronous and asynchronous workflows using **Tasks** and **Channels**.

*   **Network Layer:** Use **`HTTP.jl`**.
*   **Streaming:** Return a `Channel{T}`. This allows the user to iterate over the response as it arrives.
    ```julia
    function chat(client::OllamaClient, req::ChatRequest)
        if req.stream
            return Channel{ChatResponse}() do ch
                HTTP.open("POST", "$(client.url)/api/chat", ...) do stream
                    for line in eachline(stream)
                        put!(ch, JSON3.read(line, ChatResponse))
                    end
                end
            end
        else
            resp = HTTP.post("...")
            return JSON3.read(resp.body, ChatResponse)
        end
    end
    ```

---

## 4. Helper Logic & Multiple Dispatch

### Python Implementation
Python uses explicit helper functions like `_copy_images` and `_copy_tools` to handle various input types (e.g., converting a file path string into a Base64 encoded `Image` object).

### Julia Mapping Strategy
Use **Multiple Dispatch** to handle input variety gracefully.
```julia
# Handle raw bytes
Image(data::Vector{UInt8}) = Image(base64encode(data))

# Handle file paths
function Image(path::String)
    if isfile(path)
        return Image(base64encode(read(path)))
    end
    return Image(path) # Assume it's already base64
end
```

## 5. Recommended Package Stack
To replicate the functionality of the Python library, the following Julia packages are recommended:

1.  **`HTTP.jl`**: Core networking and streaming.
2.  **`JSON3.jl`**: High-performance JSON parsing.
3.  **`StructTypes.jl`**: Mapping JSON to Julia Structs.
4.  **`Base64.jl`**: Encoding images for multimodal models.
5.  **`URIs.jl`**: Robust parsing of the `OLLAMA_HOST`.
