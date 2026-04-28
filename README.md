# Investor-IQ Common Module (ic-common)

This project contains shared utilities and core abstractions for the Investor-IQ ecosystem. It is designed to be a lightweight, stable, and dependency-efficient library used by multiple services (agents, extraction services, etc.).

## Features

- **Unified LLM Interface:** Support for LangChain, Ollama, OpenAI, DeepSeek, and more.
- **Job Queuing:** Robust job management and processing.
- **Structured Extraction:** Tools for extracting structured data from documents.
- **Utilities:** Common helpers for ML, file operations, downloading, and more.

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

### Installation

From the project root:

```bash
cd src/common
```

**Core only:**
```bash
uv sync
```

**With optional features:**
```bash
uv sync --extra all
# Or specifically:
# uv sync --extra ml --extra media --extra pdf
```

### Running Tests

```bash
# Unit tests
uv run pytest

# Manual/Integration tests
uv run pytest src/common/tests/manual
```

## Directory Structure

- `src/common/`: Core source code.
- `src/common/tests/`: Unit and functional tests.
- `src/common/tests/manual/`: Integration tests requiring active LLM providers.

## Development

This project follows the standards defined in `GLOBAL_STANDARDS.md` and `GEMINI.md`.

- **Dependency Management:** Always use `uv`.
- **Formatting:** Follow Black/Ruff standards.
- **Type Safety:** Use Python type hints for all function signatures.
- **Logging:** Use the shared logger from `src/common/logger.py`.

## License

(Specify license here or refer to a LICENSE file)
