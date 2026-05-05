include("src/ollamajl.jl")
using .Ollamajl
using JSON3

# Create a test request
req = Ollamajl.GenerateRequest(model="llama3", prompt="Hello")

println("--- Current JSON Serialization ---")
println(JSON3.write(req))
