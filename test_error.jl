push!(LOAD_PATH, joinpath(pwd(), "src"))
include("src/ollamajl.jl")
using .Ollamajl
using Test

@testset "OllamaError Validation" begin
    # Create a mock error directly to ensure the struct and error formatting work
    err = Ollamajl.OllamaError(404, "POST", "/api/generate", "model 'invalid_model_that_doesnt_exist' not found")
    
    # Test fields
    @test err.status == 404
    @test err.method == "POST"
    @test err.error == "model 'invalid_model_that_doesnt_exist' not found"
    
    # Test Custom Showerror
    io = IOBuffer()
    Base.showerror(io, err)
    err_str = String(take!(io))
    
    @test contains(err_str, "OllamaError(404)")
    @test contains(err_str, "model 'invalid_model_that_doesnt_exist' not found")
end

@testset "Live Error Handling" begin
    # Test that the library actually throws the proper exception
    try 
        Ollamajl.generate(model="invalid_model_that_doesnt_exist"; stream=false) 
        @test false # Should not reach here
    catch e 
        @test e isa Ollamajl.OllamaError
        @test e.status == 404
        @test e.method == "POST"
    end
end

