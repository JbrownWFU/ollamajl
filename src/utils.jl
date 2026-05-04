module Utils
    using Base64
    

# Base64 encode a file (usually images)
function encodeFile(path::String)
    if !isfile(path)
        throw(SystemError("Error reading file:", 2))
    end
    
    return open(path, "r") do io
        base64encode(io)
    end
end

end # module

