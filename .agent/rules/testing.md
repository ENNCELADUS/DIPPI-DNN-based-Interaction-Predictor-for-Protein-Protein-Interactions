---
trigger: always_on
---

## Testing Standards
- **Tool**: `pytest` for all tests.
- **Coverage**: Aim for ≥80% coverage on touched modules.
- **Linting**: `ruff` for linting and formatting.

## TDD Workflow
1. **Red**: Write a failing test first.
2. **Green**: Write minimal code to pass.
3. **Refactor**: Improve code while keeping tests passing.
4. **Verify**: Check coverage and run full suite.

## Troubleshooting
- Check test isolation (mocks vs real).
- Fix implementation, not tests, unless tests are incorrect.
- Separate fast unit tests from slow integration tests.

## Commands
- `ruff check --fix .`
- `ruff format .`
- `python -m pytest`