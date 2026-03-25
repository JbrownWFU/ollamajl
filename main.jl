begin 
    using Pkg
    Pkg.activate(".")
    
    using HTTP, JSON3, DotEnv
    
    include("ollama.jl")
end

# Document maker TODO
# Add input for context, i.e. this is a dev focused package vs user facing application
# Make smart, point to directory and work
# Plan out and think about each file, reference other files
# Generate plan and execute step by step 

begin
    template = read("templates/v1.md") |> String
    sysPrompt = read("prompts/readmer.md") |> String
end

ollama = Ollamajl.initInstance(
    "http://192.168.1.165:11434", 
    model="granite4:3b")

code = read("d.toit") |> String

query = sysPrompt * "\nCode:\n" * code

gen = Ollamajl.generate(ollama, query)

doc = gen.response

open("toit-README.md", "w") do io
    write(io, doc)
end 