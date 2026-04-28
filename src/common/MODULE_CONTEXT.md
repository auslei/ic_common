# Common Module Context

## Rules

- Only add code here if it is used by at least two packages.
- Keep interfaces stable, minimal, and dependency-light.
- No heavy wrappers around stdlib or third-party libraries.

## Utilities

### `unified_llm.py`

`UnifiedChatModel` — LangChain-compatible `BaseChatModel` for all providers (Ollama, OpenAI, DeepSeek, LlamaCpp, LM Studio).

Key features:

- `invoke(prompt, schema=MyPydanticModel, thinking=False)` — structured output and CoT toggle at call time.
- `bind_tools` / `bind` return new instances (fluent API).
- Auto-logs inference time and model load time.

### `llm_utils.py`

Low-level helpers.

- `get_llm(settings: LLMSettings) -> UnifiedChatModel` — primary factory.
- `get_message_content(msg)` — strips `<think>`, `<tool_call>` tags from message content.
- `get_token_count(messages)` — estimates tokens (1 char = 1 token, CJK-safe).
- `unload_ollama_model(model_name)` — frees VRAM for a specific Ollama model.

### `LLMProvider.py`

- `LLMSettings` — runtime params (temperature, context_window, top_p, thinking).
- `LLMProvider` — provider/model identification.
- `LLAMA_CPP_MODEL_PATHS` — local GGUF blob paths.

### `pydantic_helper.py`

- `build_structured_prompt(prompt, schema)` — injects JSON schema instructions into a prompt.
- `parse_structured_output(message, schema, llm)` — parses LLM output into Pydantic with multi-stage recovery (direct → bracket extraction → LLM repair).

### `logger.py`

- `get_logger(name, level, icon, to_file)` — Rich console logger with optional file output.
- Set `LOG_TO_FILE=true` in `.env` to persist logs to `logs/`.

### `job_queue.py`

Shared serialized queue for long-running GPU/CPU ops. Exposes FastAPI routes via `make_router()`.

### `config.py`

Singleton `config` validated via Pydantic from `config.yaml`.

### `models.py`

Shared Pydantic domain models: `ExtractionResult`, `ProjectMetadata`, etc.

### `docker_utils.py`

- `resolve_project_dir(project_name)` — consistent path resolution across host and Docker.

### `ml_utils.py`

- `get_device()` — returns `cuda` / `mps` / `cpu`.
- `get_gpu_usage()` — current GPU memory usage percentage.
- `millisec_to_time(ms)` — formats duration string.

### `document_utils.py`

Markdown-to-PDF conversion via Pandoc.

### `download_utils.py`

YouTube/media downloading with caching via yt-dlp.

### `utils.py` / `file_utils.py`

General filesystem and formatting helpers: `ensure_dir`, `render_markdown_to_html`, `sanitize_project_name`.
