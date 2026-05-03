# `ollama/__init__.py` - Public API Surface

## Overall Design
The `__init__.py` file serves as the gateway to the library. It follows a **facade pattern**, flattening the internal structure to provide a clean, easy-to-use interface for end-users.

### Key Components
- **Public Exports (`__all__`)**: Explicitly defines which classes and types are part of the public API, such as `Client`, `AsyncClient`, and various response types.
- **Singleton Pattern (Default Client)**: It instantiates a default `Client()` named `_client`.
- **Method Aliasing**: It maps methods from the internal `_client` instance (e.g., `_client.chat`) directly to the module level (e.g., `ollama.chat`).

## Design Principles for Julia Port

### 1. Zero-Config Start
The Python implementation allows users to call `ollama.chat()` without manually instantiating a client. 
- **Julia Tip**: Consider providing a default `GlobalClient` or using a similar pattern where a default connection is used if none is provided.

### 2. Flattened Namespace
Internal implementation details (like `_client.py` and `_types.py`) are hidden. Users only interact with the `ollama` module.
- **Julia Tip**: Use your `module` exports strategically to hide internal sub-modules and only expose what's necessary.

### 3. Comprehensive Type Exposure
By exporting response types in `__init__.py`, the library ensures that users can perform type checking or use those types for their own function signatures.
- **Julia Tip**: Export your `struct` types for responses so users can take advantage of Julia's multiple dispatch and type system.

### 4. Sync/Async Symmetry
Both `Client` and `AsyncClient` are exported prominently, acknowledging that different environments (scripts vs. web servers) have different concurrency needs.
- **Julia Tip**: Julia's `Task` and `async/await` model will allow you to mirror this, perhaps using `channel`s for streaming responses.
