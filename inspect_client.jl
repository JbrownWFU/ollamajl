push!(LOAD_PATH, joinpath(pwd(), "src"))
include("src/ollamajl.jl")
using .Ollamajl
c = Client()
println("Type: ", typeof(c))
println("Host: ", c.host)
println("Headers: ", c.headers)
println("Timeout: ", c.timeout)
