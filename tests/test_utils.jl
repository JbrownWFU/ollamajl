push!(LOAD_PATH, joinpath(pwd(), "src"))
include("src/ollamajl.jl")
using .Ollamajl
using Test
using Base64

@testset "Utils.parseImage" begin
    # Test base64 string
    b64 = Base64.base64encode("fake image data")
    # PNG magic bytes
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    @test Ollamajl.Utils.parseImage(png_b64) == png_b64
    
    # Test file path (dog.png exists in resources)
    dog_path = "resources/dog.png"
    if isfile(dog_path)
        encoded_dog = Ollamajl.Utils.parseImage(dog_path)
        @test !isempty(encoded_dog)
        @test encoded_dog == Base64.base64encode(read(dog_path))
    end
    
    # Test Data URI
    data_uri = "data:image/png;base64,$png_b64"
    @test Ollamajl.Utils.parseImage(data_uri) == png_b64
    
    # Test invalid input
    @test_throws ArgumentError Ollamajl.Utils.parseImage("not a file and not base64")
end

@testset "Utils.buildTool" begin
    """
    Test function for tool building
    """
    function my_test_func(x::Int, y::Float64, s::String, b::Bool)
        return "$s: $(x + y) ($b)"
    end
    
    tool = Ollamajl.Utils.buildTool(my_test_func)
    @test tool["type"] == "function"
    @test tool["function"]["name"] == "my_test_func"
    @test tool["function"]["description"] == "Test function for tool building"
    
    params = tool["function"]["parameters"]
    @test params["type"] == "object"
    @test "x" in params["required"]
    @test "y" in params["required"]
    @test "s" in params["required"]
    @test "b" in params["required"]
    
    @test params["properties"]["x"]["type"] == "integer"
    @test params["properties"]["y"]["type"] == "number"
    @test params["properties"]["s"]["type"] == "string"
    @test params["properties"]["b"]["type"] == "boolean"
end
