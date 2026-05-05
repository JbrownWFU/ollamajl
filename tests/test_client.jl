push!(LOAD_PATH, joinpath(pwd(), "src"))
include("src/ollamajl.jl")
using .Ollamajl
using Test

@testset "Client instantiation" begin
    # Default client
    c = Client()
    @test c.host == "http://127.0.0.1:11434"
    @test any(p -> p.first == "User-Agent" && p.second == "ollama-julia/0.1.0", c.headers)
    
    # Custom host
    c2 = Client(host="http://example.com")
    @test c2.host == "http://example.com"
    
    # Custom headers - should merge with defaults?
    # Current implementation replaces them if passed.
    custom_headers = ["X-Custom" => "Value"]
    c3 = Client(headers=custom_headers)
    @test any(p -> p.first == "X-Custom" && p.second == "Value", c3.headers)
    # Defaults should now be merged!
    @test any(p -> p.first == "User-Agent", c3.headers)
end
