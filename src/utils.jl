module Utils
    using Base64

export encodeFile, parseImage, build_tool
    
# Parse a user passed string and determine if its a base64 encode image or a file path
# If file path encode image
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
    # Note does not support animated gifs
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

# Base64 encode a file (usually images)
function encodeFile(path::String)
    if !isfile(path)
        throw(SystemError("Error reading file: $path", 2))
    end
    
    return open(path, "r") do io
        base64encode(io)
    end
end

# Build a tool from an AbstractDict (if the user manually created a schema)
function build_tool(dict::AbstractDict)
    # We just return the dict, it will be serialized by JSON3
    return dict
end

# Inspect a Julia function and build a JSON schema tool
function build_tool(f::Function)
    name = string(nameof(f))
    
    # Extract documentation
    doc_str = ""
    # Getting doc string safely (first block)
    try
        doc_obj = Docs.doc(f)
        if !isnothing(doc_obj)
            # Take the raw string and strip markdown or just use it directly
            full_doc = string(doc_obj)
            # Only use the first paragraph to avoid overwhelming the model with signature
            doc_str = strip(split(full_doc, "\n\n")[1])
            # If doc string is empty or just says "No documentation found.", ignore
            if contains(doc_str, "No documentation found")
                doc_str = ""
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
    arg_names = Base.method_argnames(m)[2:end] # Skip #self#
    arg_types = m.sig.types[2:end]
    
    for (i, arg_name) in enumerate(arg_names)
        arg_str = string(arg_name)
        if isempty(arg_str) || startswith(arg_str, "#")
            continue
        end
        
        push!(required, arg_str)
        T = arg_types[i]
        
        json_type = if T <: Bool
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
        
        properties[arg_str] = Dict{String, Any}(
            "type" => json_type
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
            "description" => isempty(doc_str) ? "Calls function $name" : doc_str,
            "parameters" => parameters
        )
    )
end

end # module

