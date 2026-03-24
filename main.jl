begin
    using Pkg
    Pkg.activate(".")
    
    using HTTP, JSON3
    
    include("ollama.jl")
end

ollama = Ollama.initInstance(
    "http://127.0.0.1:11434", 
    "granite4:3b"
    )

# prompt = "hello robot"
# resp = Ollama.generate(ollama, prompt)

chatPrompt = "whats the status of the hyperdrive"
chatResp = Ollama.chat(ollama, chatPrompt)

Ollama.printChat(ollama)

Ollama.clearChat(ollama)