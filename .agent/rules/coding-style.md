---
trigger: glob
globs: "src/**/*.py", "tests/**/*.py"
---

## Python Coding Standards

### Structure & Organization
- **File Size**: Target 200-400 lines; strict max 600 lines. Split large components.
- **Function Size**: Keep functions small and focused (<50 lines).
- **Modularity**: Prefer many small, cohesive files over monolithic ones.
- **Grouping**: Organize by feature domain, not technical layer.
- **Utilities**: Aggressively extract shared logic into utility modules.

### Code Quality
- **Readability**: Prioritize clear internal naming and readability.
- **Complexity**: Avoid deep nesting (>4 levels). Simplify control flow.
- **Immutability**: Prefer immutable data structures; avoid side-effect mutation.
- **Values**: No hardcoded constants; use config or named constants.
- **Logging**: No `print` statements. Use standard logging.

### Naming Conventions
- **Modules/Files**: `snake_case` (e.g., `data_io.py`)
- **Classes**: `PascalCase` (e.g., `Trainer`, `V3`)
- **Functions/Vars**: `snake_case` (e.g., `train_one_epoch`)
- **Constants**: `UPPER_SNAKE`

### Best Practices
- **Typing**: Strict type hints for public APIs. Avoid `Any`.
- **Errors**: Handle specific exceptions. Never use bare `except`.
- **Imports**: Absolute from project root (e.g., `from src.utils import config`). Avoid relative imports.
- **Docs**: Google-style docstrings for non-trivial classes/functions.