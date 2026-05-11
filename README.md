# Ollamajl.jl 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia Version](https://img.shields.io/badge/julia-v1.6+-blue.svg)](https://julialang.org)

**The most idiomatic way to use Ollama in Julia.** 

`Ollamajl.jl` brings the simplicity of the official [Ollama Python library](https://github.com/ollama/ollama-python) to the Julia ecosystem, while leveraging Julia's powerful type system and meta-programming for a seamless experience.

For more information on the underlying endpoints and capabilities, please refer to the [official Ollama API documentation](https://docs.ollama.com/api/introduction).

---

## Features ✨

- **Zero Configuration:** Works out of the box with your local Ollama instance.
- **Streaming by Default:** Built for responsiveness—get tokens as they're generated.
- **Auto-Magic Tool Calling:** Pass any Julia function directly; we'll handle the JSON schema generation using reflection and docstrings.
- **Effortless Multimodal:** Pass local file paths directly to vision models like `llava`.
- **Type Safe:** Fully typed responses powered by `StructTypes` and `JSON3`.

## Installation

```julia
using Pkg
# Add via URL until registered
Pkg.add(url="https://github.com/JbrownWFU/Ollamajl.jl")
```

## Quick Start 🚀

No need to set up a client or configure headers—just start chatting.

```julia
using Ollamajl

# 1. Streaming Chat (The Default)
# Requests stream by default for that "live" feel.
ms = [Message("user", "Write a 5-line poem about Julia's speed.")]
for chunk in chat(model="llama3", messages=ms)
    print(chunk.message.content)
end

# 2. Simple Non-streaming Generate
# Just set stream=false for one-shot responses.
resp = generate(model="llama3", prompt="Why is Julia great for AI?", stream=false)
println(resp.response)
```

## Advanced Usage 🛠️

### Automatic Tool Calling
Stop writing JSON schemas by hand. `Ollamajl` introspects your Julia functions, extracts their documentation, and builds the schema for you.

```julia
"""
Add two numbers together. This docstring is sent to the model!
"""
function add_numbers(a::Int, b::Int)
    return a + b
end

resp = chat(
    model="llama3.1", 
    messages=[Message("user", "What is 15 + 27?")],
    tools=[add_numbers],
    stream=false
)

if !isnothing(resp.message.tool_calls)
    # The model "called" your function!
    call = resp.message.tool_calls[1]
    println("Model wants: ", call.func.name, " with args ", call.func.arguments)
end
```

### Vision (Multimodal) 👁️
Pass local file paths, base64 strings, or Data URIs. We handle the encoding automatically.

```julia
resp = generate(
    model="llava", 
    prompt="Describe this image.", 
    images=["resources/cat.jpg"], 
    stream=false
)
```

### Custom Clients & Options
Need to connect to a remote server or tweak model parameters?

```julia
# Connect to a remote Ollama server
remote_client = Client(host="http://gpu-server:11434", timeout=120)

resp = generate(
    client=remote_client,
    model="gemma2",
    prompt="Explain quantum entanglement",
    temperature=0.7, # Keyword arguments pass directly to model options
    num_ctx=4096
)
```

## API Reference 📖

- `chat(...)`: Conversation endpoint (returns `Channel` if streaming).
- `generate(...)`: Completion endpoint (returns `Channel` if streaming).
- `embed(...)`: Generate vector embeddings for search and RAG.
- `pprint(resp)`: Pretty prints responses with some terminal color.
- `Message(role, content, [images])`: The building block for chat history.

## License
MIT
