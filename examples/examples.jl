using Ollamajl, JSON3

# Instantiate an ollama instance
o = OllamaInstance("http://127.0.0.1:11434")

# Generate a response
# ------------

# Using kwargs (simplest)
generate(o, model="gemma4:e2b", prompt="Go gators!", stream=false)

# Using a premade Options object
opts = Options(
    model="gemma4:e2b", 
    prompt="Go gators!", 
    stream=false
    )

generate(o, opts)

# Generate the next chat message
# ------------

# Using kwargs (simplest) by passing a vector of Dicts
chat(o,
    model="gemma4:e2b", 
    messages=[Dict("role" => "user", "content" => "Go gators!")],
    stream=false
)

# Using Message struct
messages = Message(role="user", content="Go gators!")
chat(o; model="gemma4:e2b", messages=[messages], stream=false)

# Using JSON3 to read and pass raw JSON
rawJson = Dict("role" => "user", "content" => "Go gators!") |> JSON3.write
messages = JSON3.read(rawJson)

chat(o,
    model="gemma4:e2b",
    messages = [messages],
    stream=false
)

# Using premade Options object
msgs = Message(role="user", content="Go gators!")
opts = Options(
    model="gemma4:e2b",
    messages=[msgs],
    stream=false
    )

chat(o, opts)

# Message struct can also be passed to Options struct
msgs = Message(
    role="user",
    content="Go gators!"
    )

opts = Options(
    model="gemma4:e2b", 
    messages=[msgs], 
    stream=false
    ) 

resp = chat(o, opts)

# The Options struct is mutable to allow storage of messages from the LLM
llmMsg = String(resp.body) |> JSON3.read
push!(opts.messages, Message(role=llmMsg.message.role, content=llmMsg.message.content))

# Embeddings
# ------------

# Using kwargs
embed(o; model="embeddinggemma", input="why is the sky blue?")

# Using Options struct
opts = Options(model="embeddinggemma", input="why is the sky blue?")
e = embed(o, opts)
