include("src/ollamajl.jl")
using .Ollamajl
using JSON3

println("=== Testing build_tool (Function Introspection) ===")
"""
Add two numbers
"""
function add_two_numbers(a::Int, b::Int)
    return a + b
end

t = Ollamajl.Utils.build_tool(add_two_numbers)
println(JSON3.write(t))

println("\n=== Testing ChatRequest Serialization ===")
subtract_tool = Dict(
  "type" => "function",
  "function" => Dict(
    "name" => "subtract_two_numbers",
    "description" => "Subtract two numbers",
    "parameters" => Dict(
      "type" => "object",
      "required" => ["a", "b"],
      "properties" => Dict(
        "a" => Dict("type" => "integer", "description" => "The first number"),
        "b" => Dict("type" => "integer", "description" => "The second number")
      )
    )
  )
)

messages = [Message(role="user", content="What is three plus one?")]

# Simulate what `chat()` does internally with tools
processedTools = Any[t isa Ollamajl.Tool ? t : Ollamajl.Utils.build_tool(t) for t in [add_two_numbers, subtract_tool]]

req = Ollamajl.ChatRequest(
    model="llama3.1",
    messages=messages,
    tools=processedTools,
    stream=false
)

json_str = JSON3.write(req)
println(json_str)

println("\n=== Testing Deserialization of ToolCall (Simulating Response) ===")
mock_response_json = """
{
  "model": "llama3.1",
  "created_at": "2026-05-05T00:00:00Z",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "add_two_numbers",
          "arguments": {"a": 3, "b": 1}
        }
      }
    ]
  },
  "done": true
}
"""

resp = JSON3.read(mock_response_json, Ollamajl.ChatResponse)
tool_call = resp.message.tool_calls[1]

println("Parsed ToolCall Function Name: ", tool_call.func.name)
println("Parsed ToolCall Arguments: ", tool_call.func.arguments)
