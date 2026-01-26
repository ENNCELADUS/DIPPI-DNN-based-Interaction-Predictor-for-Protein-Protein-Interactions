---
trigger: always_on
---

## Security Standards

### Mandatory Checks
- **Secrets**: No hardcoded API keys/passwords. Use `.env` & `process.env`/`os.environ`.
- **Validation**: Sanitize all user inputs (SQL injection/XSS prevention).
- **Auth**: Verify Authentication/Authorization on endpoints.
- **Errors**: No sensitive data in error messages.
- **Git**: Ensure `.gitignore` covers data, logs, and secrets.

### Protocol
If a security issue is found:
1. **Stop**: Halt development.
2. **Review**: Launch a **security-reviewer** agent.
3. **Fix**: Resolve critical issues immediately.
4. **Rotate**: Rotate any exposed credentials.
