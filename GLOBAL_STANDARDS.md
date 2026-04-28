## Agent Rules (Common Standard)

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
- **Packages:** Each package must desine clear I/O specs and also webservice intended to be used as a microservice. 

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

### Final Rule

Review this file before generating or modifying code to ensure consistency.
