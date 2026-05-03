begin
    using Pkg
    Pkg.activate(".")

    using Revise
    includet("src/ollamajl.jl")
    using .Ollamajl
    
    using HTTP, JSON3
end

client = Client()

g = generate(model="granite4:3b", prompt="Hello robot", stream=false)
# pprint(g)

# opts = Ollamajl.Options(temperature=65.0)
# g = generate(model="granite4:3b", prompt="Hello bot", stream=false, options=opts)

# Chat
ms = Ollamajl.Message("user", "hello brobot")
c = Ollamajl.chat(model="granite4:3b", messages=[ms], stream=false)