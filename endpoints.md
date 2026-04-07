# Generate
```
Body
application/json
​
model
string
required

Model name
​
prompt
string

Text for the model to generate a response from
​
suffix
string

Used for fill-in-the-middle models, text that appears after the user prompt and before the model response
​
images
string[]

Base64-encoded images for models that support image input
​
format
string

Structured output format for the model to generate a response from. Supports either the string "json" or a JSON schema object.
​
system
string

System prompt for the model to generate a response from
​
stream
boolean
default:true

When true, returns a stream of partial responses
​
think
boolean

When true, returns separate thinking output in addition to content. Can be a boolean (true/false) or a string ("high", "medium", "low") for supported models.
​
raw
boolean

When true, returns the raw response from the model without any prompt templating
​
keep_alive
string

Model keep-alive duration (for example 5m or 0 to unload immediately)
​
options
object

Runtime options that control text generation

Show child attributes
​
logprobs
boolean

Whether to return log probabilities of the output tokens
​
top_logprobs
integer

Number of most likely tokens to return at each token position when logprobs are enabled
Response

Generation responses
​
model
string

Model name
​
created_at
string

ISO 8601 timestamp of response creation
​
response
string

The model's generated text response
​
thinking
string

The model's generated thinking output
​
done
boolean

Indicates whether generation has finished
​
done_reason
string

Reason the generation stopped
​
total_duration
integer

Time spent generating the response in nanoseconds
​
load_duration
integer

Time spent loading the model in nanoseconds
​
prompt_eval_count
integer

Number of input tokens in the prompt
​
prompt_eval_duration
integer

Time spent evaluating the prompt in nanoseconds
​
eval_count
integer

Number of output tokens generated in the response
​
eval_duration
integer

Time spent generating tokens in nanoseconds
​
logprobs
object[]

Log probability information for the generated tokens when logprobs are enabled

Show child attributes
```