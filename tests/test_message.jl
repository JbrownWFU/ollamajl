push!(LOAD_PATH, joinpath(pwd(), "src"))
include("src/ollamajl.jl")
using .Ollamajl

println("Testing Message constructors...")

try
    println("Attempting keyword constructor: Message(role=\"user\", content=\"hello\")")
    m = Message(role="user", content="hello")
    println("SUCCESS: ", m)
catch e
    println("FAILURE (keyword): ", e)
    # Stacktrace for more detail if needed
    # Base.display_error(e, catch_backtrace())
end

try
    println("\nAttempting positional constructor: Message(\"user\", \"hello\")")
    m = Message("user", "hello")
    println("SUCCESS: ", m)
catch e
    println("FAILURE (positional): ", e)
end

try
    println("\nAttempting default constructor: Message()")
    m = Message()
    println("SUCCESS: ", m)
catch e
    println("FAILURE (default): ", e)
end
