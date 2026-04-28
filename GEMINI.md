# Common Module (ic-common)

This project contains shared utilities and core abstractions for the Investor-IQ ecosystem. It is designed to be a lightweight, stable, and dependency-efficient library used by multiple services (agents, extraction services, etc.).

## Project Overview

- **Core Purpose:** Provides a unified interface for LLM providers, job queuing, structured data extraction, and common utility functions (ML, File, Download).
- **Primary Technologies:**
  - **Python:** >= 3.12
  - **LLM Integration:** LangChain (Core, Community, Ollama, OpenAI, DeepSeek)
  - **API/Concurrency:** FastAPI
  - **Data Validation:** Pydantic v2
  - **Package Management:** [uv](https://github.com/astral-sh/uv)

## Building and Running

### Prerequisites
- Python 3.12+
- `uv` (recommended)

### Installation
From the `src/common` directory:

**Core only:**
```bash
uv sync
```

**With optional features:**
```bash
uv sync --extra ml --extra media --extra pdf
# Or all at once
uv sync --extra all
```

### Running Tests
Automated tests are located in `src/common/tests`.
```bash
uv run pytest
```
Manual/Integration tests can be found in `src/common/tests/manual`.

## Development Standards (from GLOBAL_STANDARDS.md)

### Core Behaviour
- Use context7 MCP to retrieve code references; ensure you use the correct version for libraries.
- If anything is unclear or ambiguous, **ask the user** before acting.
- Propose changes first; implement only after explicit user approval.
- Please do not modify/reference source code under z_archive.

### Modules & Architecture
- **Isolation:** Each module must define clear I/O specs and run independently.
- **Commonality:** Shared logic lives in `src/common`.
- **State:** In graph-based workflows, always verify the `State` schema before modifying node logic.
- **Coupling:** High cohesion, low coupling. Interact only via public APIs.
- **File Size:** Avoid large files (>30KB). Refactor into smaller modules or sub-packages if complexity grows, to ensure compatibility with license and obfuscation tools.
- **Packages:** Each package must design clear I/O specs and also webservice intended to be used as a microservice.

### Coding Standards
- **Environment:** Always use `uv` for dependency management (`uv add`, `uv lock`). Ensure `PYTHONPATH` is set to the project root when running scripts or tests.
- **Style:** Simple, concise, readable. Use `snake_case` (vars/functions) and `PascalCase` (classes).
- **Tooling:** Follow project formatting (Black/Ruff). Keep imports alphabetized and minimal.
- **Type Safety:** Use Python type hints for all function signatures. Use Pydantic models (see `src/common/pydantic_helper.py`) for complex data structures and tool arguments.
- **AI Tools:** Use descriptive docstrings for all tool definitions, as these are used as metadata for the LLM.
- **Comments:** Short inline comments only where intent is non-obvious.

### Design Principles
- Prefer straightforward logic over unnecessary wrappers/abstractions.
- Expose functionality via Python APIs, not shell commands unless mandatory.
- **Zero Side Effects:** Functions should be deterministic where possible.

### Error Handling & Logging
- Raise clear, actionable errors; avoid raw tracebacks.
- **Logging:**
    - Use the shared logger from `src/common/logger.py` instead of `print()` or the standard `logging` module for internal execution tracking.
    - Logging (level info) should be done sufficiently before major steps, to make it easier for debugging.

### Testing & Quality
- Update/add tests for every behavioral change.
- **Location:** Place tests within the corresponding module's directory (e.g., `src/[module_name]/tests/`).
- Prefer small, deterministic unit tests.
- Verify changes against existing test suites before declaring "done."

### Workflow
- **Plan:** Explain reasoning and impact before coding.
- **Minimalism:** Modify only what the user requests.
- **Cleanup:** Keep `dev/` and `docs/` updated; remove obsolete files immediately. Update local `MODULE_CONTEXT.md` files within modules when their APIs or responsibilities change.
- **Structure:** Follow conventions: `examples/`, `tests/`, `dev/`, `docs/`, `src/`.

## Directory Structure (Core)
- `src/common/`: Core source code.
- `src/common/tests/`: Unit and functional tests.
- `src/common/tests/manual/`: Integration tests requiring active LLM providers.
