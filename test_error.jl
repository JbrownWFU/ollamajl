include("src/ollamajl.jl")
using .Ollamajl
try 
    Ollamajl.generate(model="invalid_model_that_doesnt_exist"; stream=false) 
catch e 
    println("Caught error: ", typeof(e)) 
    println("Status: ", e.status)
    println("Method: ", e.method)
    println("Error message: ", e.error)
end
Ollamajl.generate(model="invalid_model_that_doesnt_exist"; stream=false) 