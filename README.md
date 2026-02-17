# Simple Agentic Loop

Minimal `uv` project using the OpenAI Responses API with exactly one tool: local Python code execution.

## Features

- Conversation loop with persistent context between turns.
- One tool only: `run_code` (runs `python -c ...` in a local subprocess).
- Progress output sections for `reasoning`, `code`, and `result`.
- Wrapped terminal output for cleaner long messages.
- Interrupt support:
  - `Ctrl+C` while typing clears input and returns to prompt.
  - `Ctrl+C` during an active turn interrupts the current response.

## Setup

```bash
uv sync
```

Create `.env` in project root:

```env
OPENAI_API_KEY=your_key_here
```

## Run

```bash
uv run --env-file .env python main.py
```

## Prompt Controls

- `Enter`: send message
- `Ctrl+J`: insert newline (multiline input)
- `quit` or `exit`: leave the app
