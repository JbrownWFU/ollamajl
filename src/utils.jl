module Utils

using Base64

export encodeFile, parseImage, buildTool

"""
    parseImage(str::String)

Parse a user-provided string and determine if it's a base64 encoded image or a file path.
If it's a file path, it reads and encodes the file.

Returns a base64 encoded string of the image data.
"""
function parseImage(str::String)
    # 1. Handle Data URI if present
    # e.g., data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...
    if startswith(str, "data:image/") && contains(str, ";base64,")
        parts = split(str, ";base64,")
        if length(parts) >= 2
            return String(parts[2])
        end
    end

    # 2. Check if it's likely a base64 string already
    # Common image base64 headers:
    # PNG: iVBOR
    # JPEG: /9j/
    # GIF: R0lG
    # WEBP: UklG
    # Note: does not support animated gifs
    if startswith(str, "iVBOR") || startswith(str, "/9j/") || 
       startswith(str, "R0lG") || startswith(str, "UklG")
        return str
    end

    # 3. Check for file path
    # We do this after magic bytes check because a very long base64 string 
    # might accidentally trigger some OS path limit checks or be slow.
    isPath = try
        isfile(str)
    catch
        false
    end

    if isPath
        return encodeFile(str)
    end

    # 4. Fallback: if it's very long, assume it's base64 even if magic bytes didn't match
    # (Though most Ollama-compatible images will match those magic bytes)
    if length(str) > 4096
        return str
    end

    # Otherwise error
    throw(ArgumentError("Invalid image input: String is neither a valid file path nor recognized as base64 encoded image data."))
end

"""
    encodeFile(path::String)

Base64 encode a file (usually images). Throws a `SystemError` if the file cannot be read.
"""
function encodeFile(path::String)
    if !isfile(path)
        throw(SystemError("Error reading file: $path", 2))
    end

    return open(path, "r") do io
        base64encode(io)
    end
end

"""
    buildTool(dict::AbstractDict)

Build a tool from an `AbstractDict` (if the user manually created a schema).
Returns the dictionary directly, which will be serialized by `JSON3`.
"""
function buildTool(dict::AbstractDict)
    return dict
end

"""
    buildTool(f::Function)

Inspect a Julia function and build a JSON schema tool definition for the Ollama API.
It extracts the function name, documentation, and argument types via reflection.
"""
function buildTool(f::Function)
    name = string(nameof(f))
    
    # Extract documentation
    docStr = ""
    # Getting doc string safely (first block)
    try
        docObj = Docs.doc(f)
        if !isnothing(docObj)
            # Take the raw string and strip markdown or just use it directly
            fullDoc = string(docObj)
            # Only use the first paragraph to avoid overwhelming the model with signature
            docStr = strip(split(fullDoc, "\n\n")[1])
            # If doc string is empty or just says "No documentation found.", ignore
            if contains(docStr, "No documentation found")
                docStr = ""
            end
        end
    catch
        # Ignore docs if not available
    end
    
    ms = methods(f)
    if isempty(ms)
        throw(ArgumentError("Function $name has no methods."))
    end
    
    # We use the first method to infer types
    m = first(ms)
    
    properties = Dict{String, Any}()
    required = String[]
    
    # Getting argument names using reflection (requires Julia 1.1+)
    argNames = Base.method_argnames(m)[2:end] # Skip #self#
    argTypes = m.sig.types[2:end]
    
    for (i, argName) in enumerate(argNames)
        argStr = string(argName)
        if isempty(argStr) || startswith(argStr, "#")
            continue
        end
        
        push!(required, argStr)
        T = argTypes[i]
        
        jsonType = if T <: Bool
            "boolean"
        elseif T <: Integer
            "integer"
        elseif T <: AbstractFloat
            "number"
        elseif T <: AbstractString || T === Symbol
            "string"
        elseif T <: AbstractVector || T <: Tuple
            "array"
        else
            "string"
        end
        
        properties[argStr] = Dict{String, Any}(
            "type" => jsonType
        )
    end
    
    parameters = Dict{String, Any}(
        "type" => "object",
        "properties" => properties,
        "required" => required
    )
    
    # Structs for tool will be built by ChatRequest
    return Dict{String, Any}(
        "type" => "function",
        "function" => Dict{String, Any}(
            "name" => name,
            "description" => isempty(docStr) ? "Calls function $name" : docStr,
            "parameters" => parameters
        )
    )
end

end # module

