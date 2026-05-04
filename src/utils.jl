module Utils
    using Base64

export encodeFile, parseImage
    
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

end # module

