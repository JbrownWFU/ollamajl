# Ollamajl.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight and flexible Julia wrapper for the [Ollama](https://ollama.com/) API. `Ollamajl.jl` provides a clean, idiomatic interface for interacting with local LLMs, supporting everything from simple text generation to complex multi-turn chats and vector embeddings.

## Features

- **Intuitive API**: Use simple functions like `generate`, `chat`, and `embed`.
- **Multiple Interaction Styles**: Choose between concise keyword arguments or structured `Options` objects.
- **Rich Type System**: Leverages Julia structs (`Message`, `Options`, `Tool`) for clear data modeling and type safety.
- **Stateful Chats**: Easily manage conversation history using mutable options.
- **Extensible**: Includes initial support for tool calling and advanced generation parameters via `SubOptions`.

## Ollama API Documentation
see the [Ollama API Docs](https://docs.ollama.com/api/introduction) for more information on the Ollama API, 

## Installation

To install `Ollamajl.jl`, use the Julia package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/San/ollamajl.git")
```

## Quick Start

### 1. Initialize the Instance

First, create an `OllamaInstance` pointing to your local Ollama server:

```julia
using Ollamajl

# Default local Ollama endpoint
o = OllamaInstance("http://127.0.0.1:11434")
```

### 2. Text Generation

Generate responses quickly using keyword arguments:

```julia
resp = generate(o, model="llama3", prompt="Why is Julia great for AI?", stream=false)
println(String(resp.body))
```

Or use a structured `Options` object for more control:

```julia
opts = Options(
    model="gemma2", 
    prompt="Write a haiku about recursion.", 
    stream=false
)
generate(o, opts)
```

### 3. Chat Interface

The `chat` function allows for sophisticated messaging. You can pass raw dictionaries, `Message` structs, or even JSON-parsed data.

```julia
# Using Message structs for clarity
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Hello!")
]

chat(o; model="llama3", messages=messages, stream=false)
```

#### Managing Conversation History
Because the `Options` struct is mutable, you can easily track a conversation:

```julia
using JSON3

opts = Options(
    model="llama3",
    messages=[Message(role="user", content="My name is San.")],
    stream=false
)

# Get response
resp = chat(o, opts)

# Append the LLM's response to your history
llm_msg = JSON3.read(String(resp.body)).message
push!(opts.messages, Message(role=llm_msg.role, content=llm_msg.content))
```

### 4. Vector Embeddings

Generate embeddings for RAG or search tasks:

```julia
# Using kwargs
embed(o; model="mxbai-embed-large", input="Julia is fast.")

# Using Options
opts = Options(model="mxbai-embed-large", input="Julia is fast.")
e = embed(o, opts)
```

## Advanced Usage

### Customizing Generation Parameters
Use `SubOptions` within your `Options` struct to fine-tune the model's behavior:

```julia
sub_opts = SubOptions(temperature=0.7, top_p=0.9, num_ctx=4096)
opts = Options(model="llama3", prompt="Be creative!", options=sub_opts)
generate(o, opts)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
