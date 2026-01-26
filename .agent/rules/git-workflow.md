---
trigger: model_decision
description: Apply this rule when managing git history, creating commits, branching, or submitting code for review.
---

## Git Workflow Standards

### Commit Messages
Format: `<type>: <description>`
Types:
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance
- `refactor`: Structural change (no API change)
- `docs`: Documentation
- `test`: Tests
- `chore`: Build/Tools
- `ci`: CI/CD

### Branching Strategy
- **Format**: `short-description` (e.g., `auth-middleware`, `login-bug`)
- **Base**: Always rebase on main before merging.

### Workflow
1. **Plan**: Analyze requirements and dependencies.
2. **Test**: Write tests first (TDD) where applicable.
3. **Diff**: Use `git diff [base]...HEAD` to verify changes.
4. **Push**: Use `-u` for new branches.

### Pull Requests
- Summarize full commit history.
- Include test plan and coverage report.
- Address **High/Critical** review issues before merging.