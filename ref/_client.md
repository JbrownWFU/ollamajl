# `ollama/_client.py` - Core Request Logic

## Overall Design
This file is the engine of the library. It handles the low-level HTTP communication with the Ollama server, host parsing, and error handling.

### Key Components
- **`BaseClient`**: An abstract base class that manages the shared configuration for both synchronous and asynchronous clients (headers, base URL, timeouts).
- **`Client` (Sync)**: Uses `httpx.Client` for blocking requests.
- **`AsyncClient` (Async)**: Uses `httpx.AsyncClient` for non-blocking requests.
- **`_request` / `_request_raw`**: Centralized methods for making HTTP calls. `_request` handles the logic of parsing JSON responses and mapping them to Pydantic models.
- **Host Parsing (`_parse_host`)**: A robust utility that handles various host formats (IPv4, IPv6, ports, schemes) and defaults to `http://127.0.0.1:11434`.

## Design Principles for Julia Port

### 1. Abstraction of Concurrency
The library uses inheritance to share 90% of the logic between sync and async clients.
- **Julia Tip**: You might use a common `AbstractOllamaClient` type and then implement specific methods for `SyncClient` and `AsyncClient`, or use a single client with optional `async` flags.

### 2. Streaming as Iterators
Streaming responses are handled by `_request` using `yield` (iterators) in Python. Each line is parsed as an independent JSON object and yielded as a response model.
- **Julia Tip**: Use `Channels` or `Iterators` to handle streamed lines from the HTTP response. This allows the user to process parts of the message as they arrive.

### 3. Dynamic Host Resolution
The `_parse_host` function is essential for robustness. It handles environment variables (`OLLAMA_HOST`) and provides sensible defaults.
- **Julia Tip**: Implement a similar utility function to normalize URLs before making requests.

### 4. Overloaded Signatures
Python uses `@overload` to tell type checkers that the return type changes based on the `stream` parameter.
- **Julia Tip**: In Julia, you can use multiple dispatch! You could have `generate(client, prompt; stream=false)` and `generate(client, prompt; stream=true)` return different types (e.g., a `GenerateResponse` vs. a `Channel{GenerateResponse}`).

### 5. Error Wrapping
The client catches HTTP errors and wraps them in custom exceptions like `ResponseError` and `ConnectionError` with user-friendly messages.
- **Julia Tip**: Define custom `Exception` types in Julia to provide clear feedback when the server is down or returns an error.
