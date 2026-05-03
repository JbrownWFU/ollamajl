begin
    using Pkg
    Pkg.activate(".")

    using Revise
    includet("src/ollamajl.jl")
    using .Ollamajl
    
    using HTTP, JSON3
end

client = Ollamajl.Client()