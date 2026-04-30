# Ollamajl Options Refactor Notes

## The Problem

The monolithic `Options` struct unions three different request shapes into one. Every call site is implicitly lying about which fields are valid — fields like `prompt` are meaningless in a chat request, and `messages` is meaningless in a generate request. These aren't variations of one thing, they're distinct request types that happen to share some fields.

```julia
# Current: one struct rules them all, most fields are Nothing at any given time
Base.@kwdef mutable struct Options
    model::String
    prompt::Any = nothing        # generate only
    messages::Union{...} = nothing  # chat only
    input::Union{...} = nothing     # embed only
    tools::Union{...} = nothing     # chat only
    # ...
end
```

---

## The Approach: Abstract Supertype + Concrete Request Types

Define an abstract supertype and a concrete struct per endpoint. Shared fields are repeated across structs — this is intentional (see below).

```julia
abstract type OllamaRequest end

Base.@kwdef struct GenerateRequest <: OllamaRequest
    model::String
    prompt::String
    stream::Union{Bool, Nothing} = nothing
    options::Union{SubOptions, Nothing} = nothing
    system::Union{String, Nothing} = nothing
    images::Union{Vector{String}, Nothing} = nothing
    suffix::Union{String, Nothing} = nothing
    keep_alive::Union{String, Nothing} = nothing
end

Base.@kwdef struct ChatRequest <: OllamaRequest
    model::String
    messages::Vector{Message}
    stream::Union{Bool, Nothing} = nothing
    options::Union{SubOptions, Nothing} = nothing
    system::Union{String, Nothing} = nothing
    tools::Union{Vector{Tool}, Nothing} = nothing
    keep_alive::Union{String, Nothing} = nothing
end

Base.@kwdef struct EmbedRequest <: OllamaRequest
    model::String
    input::String
    stream::Union{Bool, Nothing} = nothing
    options::Union{SubOptions, Nothing} = nothing
    truncate::Union{Bool, Nothing} = nothing
    dimensions::Union{Int, Nothing} = nothing
    keep_alive::Union{String, Nothing} = nothing
end
```

---

## Why the Repetition is Fine

The repeated fields (`model`, `stream`, `keep_alive`, etc.) look redundant but they're documenting something true: each request type *independently* supports those fields. If Ollama ever adds a field to `generate` that doesn't apply to `chat`, you'd be glad you weren't sharing storage.

The alternative — sharing storage through composition with a nested `CommonOptions` — trades this honesty for DRY, then pays for it in serialization complexity (you'd need to flatten `common` back out before sending JSON, since Ollama doesn't expect a nested key). For a 4–5 field overlap, repetition is the better deal.

---

## Serialization Just Works

Because there's no nesting, `JSON3.write` produces exactly what Ollama expects with no custom serializer needed:

```julia
req = GenerateRequest(model="llama3", prompt="hello")
JSON3.write(req)
# {"model":"llama3","prompt":"hello"}
```

---

## Dispatch Replaces the if/else

### Endpoint routing

```julia
_endpoint(::GenerateRequest) = "/api/generate"
_endpoint(::ChatRequest)     = "/api/chat"
_endpoint(::EmbedRequest)    = "/api/embed"
```

These are standard single-line Julia functions with anonymous (unnamed) arguments. The `::GenerateRequest` without a variable name tells Julia: *"dispatch on this type, but I don't need the value in the function body."* Julia picks the right method based on the runtime type of the argument — no `if/else` or `switch` needed.

Equivalent verbose form:
```julia
function _endpoint(req::GenerateRequest)
    return "/api/generate"
end
```

### Shared post logic

```julia
function _post(ollama::OllamaInstance, req::OllamaRequest)
    url  = ollama.url * _endpoint(req)
    body = JSON3.write(req)
    return HTTP.request("POST", url, [], body)
end

generate(ollama::OllamaInstance, req::GenerateRequest) = _post(ollama, req)
chat(ollama::OllamaInstance, req::ChatRequest)         = _post(ollama, req)
embed(ollama::OllamaInstance, req::EmbedRequest)       = _post(ollama, req)
```

Passing a `ChatRequest` to `generate` is now a method dispatch error, not a silent bug where `messages` gets ignored.

---

## Keeping the Kwarg Interface

The ergonomic kwarg interface is preserved as a thin shim on top of the typed structs:

```julia
function generate(ollama::OllamaInstance; model, prompt, kwargs...)
    req = GenerateRequest(; model, prompt, kwargs...)
    return generate(ollama, req)
end

function chat(ollama::OllamaInstance; model, messages, kwargs...)
    req = ChatRequest(; model, messages, kwargs...)
    return chat(ollama, req)
end
```

The "real" API is typed. Kwargs are sugar on top. Callers who want type safety use the struct directly; callers who want brevity use kwargs.

---

## Operating on Any Request Type

The abstract supertype also enables shared *behaviour* without shared *storage*. Functions that operate on any request type can be written once:

```julia
# Works for any request type (using Accessors.jl)
streamed(req::OllamaRequest) = @set req.stream = true
with_model(req::OllamaRequest, m::String) = @set req.model = m

# Usage — type is preserved
req = ChatRequest(model="llama3", messages=[...])
req = streamed(req)  # still a ChatRequest, stream=true
```

Note: if you need to *read* a shared field generically (outside of Accessors.jl), Julia structs don't have inherited fields so you need explicit accessors:

```julia
model(req::OllamaRequest)  = req.model
stream(req::OllamaRequest) = req.stream
```

For a write-and-serialize workflow like this one, you'll rarely need this — each call site already knows its concrete type.

---

## Summary

| | Monolithic `Options` | Abstract supertype |
|---|---|---|
| Invalid fields | Silently `Nothing` | Compile-time dispatch error |
| Serialization | Works | Works (no nesting to flatten) |
| Shared behaviour | One struct | Methods on `OllamaRequest` |
| Field clarity | All fields always present | Each struct only has relevant fields |
| Repetition | None | ~5 shared fields repeated |
