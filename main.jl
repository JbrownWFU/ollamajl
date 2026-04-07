begin
    using Pkg
    Pkg.activate(".")

    using Revise
    using Ollamajl
    # using .Ollamajl
end

# Ollama instance
o = OllamaInstance("http://127.0.0.1:11434")

# Generate a response
generateOpts = Options(model="gemma4:e2b", prompt="Test message", stream=false)

g = generate(o, generateOpts)

# With kwargs
g2 = generate(o; model="gemma4:e2b", prompt="hello robot")

# Generate the next chat message
messages=[Dict("role" => "user", "content" => "hello robot")]

c2 = chat(o; model="gemma4:e2b", messages=messages, stream=false)

# With kwargs
messages=[Dict("role" => "user", "content" => "hello robot")]

chatOpts = Options(model="gemma4:e2b", messages=messages, stream=false)

c = chat(o, chatOpts)

# Embeddings
embedOpts = Options(model="embeddinggemma", input="why is the sky blue?")
e = embed(o, embedOpts)
e2 = embed(o; model="embeddinggemma", input="why is the sky blue?")
