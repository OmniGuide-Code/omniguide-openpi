# Agent Rules

- Fail loudly instead of silently tolerating errors.
- Do not add backward-compatibility paths, hidden cleanup, broad try/except handling, or automatic repair unless explicitly requested.
- When a command fails because of stale local state, explain the concrete cause and provide the manual command to fix it.
- Keep release setup code and scripts minimal. Prefer direct, general commands over machine-specific workarounds.
