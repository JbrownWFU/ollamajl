begin 
    using Pkg
    Pkg.activate(".")
    
    using HTTP, JSON3, DotEnv
    
    include("ollama.jl")
end

begin
    template = read("templates/v1.md") |> String
    sysPrompt = read("prompts/readmer.md") |> String
end

ollama = Ollamajl.initInstance(
    "http://192.168.1.165:11434", 
    model="granite4:3b")

code = read("ollama.jl") |> String

query = sysPrompt * "\nCode:\n" * code

gen = Ollamajl.generate(ollama, query)