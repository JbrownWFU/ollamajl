begin
    using Pkg
    Pkg.activate(".")

    using Revise
    includet("ollamajl.jl")
    using .Ollamajl
    
    using HTTP, JSON3
end