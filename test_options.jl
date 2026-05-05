push!(LOAD_PATH, joinpath(pwd(), "src"))
include("src/ollamajl.jl")
using .Ollamajl
using Test

@testset "Options Merging API" begin
    # Test `_mergeOptions` internally
    
    # Generate request should pick up `temperature`
    opt1 = Ollamajl._mergeOptions(nothing, temperature=0.8, top_k=40)
    @test opt1 isa Ollamajl.Options
    @test opt1.temperature == 0.8
    @test opt1.top_k == 40
    @test opt1.num_ctx == nothing
    
    # Merging existing Options
    existing = Ollamajl.Options(temperature=0.2, num_ctx=2048)
    opt2 = Ollamajl._mergeOptions(existing, temperature=0.9, top_p=0.5)
    
    # Overwrites temperature
    @test opt2.temperature == 0.9
    # Keeps num_ctx
    @test opt2.num_ctx == 2048
    # Adds top_p
    @test opt2.top_p == 0.5
end
