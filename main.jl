begin
    using Pkg
    Pkg.activate(".")

    using Revise
    includet("src/ollamajl.jl")
    using .Ollamajl
    
    using HTTP, JSON3
end

# Optionally define a client
client = Client()

# 1. Non-streaming Generate
resp = generate(model="gemma4:e2b", prompt="What is Julia (programming language) in one sentence?", stream=false)

dog = generate(client=client, model="gemma4:e2b", prompt="What is this?", images=["resources/dog.png"], stream=false)
pprint(dog)

cat = generate(model="gemma4:e2b", prompt="What is this?", images=["resources/cat.jpg"], stream=false)
pprint(cat)

fish = generate(model="gemma4:e2b", prompt="What is this?", images=["resources/fish.webp"], stream=false)
pprint(fish)

# 2. Streaming Generate
for chunk in generate(model="gemma4:e2b", prompt="Write a 3 line poem about Julia (programming language).", stream=true)
    print(chunk.response)
end

# TODO test with image
# With an image

# 3. Non-streaming Chat
ms = [Message("user", "What is 2+2?")]
resp = chat(model="gemma4:e2b", messages=ms, stream=false)

ms = [Message(role="user", content="Rate my dog", images=["resources/dog.png"])]
resp = chat(model="gemma4:e2b", messages=ms, stream=false)

# 4. Streaming Chat
ms = Message("user", "Tell me a short 2 paragaph story about a robot.")
for chunk in chat(model="gemma4:e2b", messages=[ms], stream=true)
    print(chunk.message.content)
end

# 5. Embedding (No streaming support)
emb = Ollamajl.embed(model="embeddinggemma", input="Julia is fast!")
