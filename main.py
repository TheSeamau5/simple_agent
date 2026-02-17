import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings


def print_section(title: str, content: str) -> None:
    bar = "-" * 72
    print(f"\n{bar}\n{title}\n{bar}")
    print(wrap_for_terminal(content if content else "(no output)"))
    print()


def wrap_for_terminal(text: str) -> str:
    width = max(60, shutil.get_terminal_size(fallback=(100, 24)).columns - 2)
    wrapped_lines: list[str] = []
    for line in (text.splitlines() or [""]):
        if not line:
            wrapped_lines.append("")
            continue
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        wrapped_lines.extend(wrapped or [""])
    return "\n".join(wrapped_lines)


def print_agent_message(text: str) -> None:
    width = max(60, shutil.get_terminal_size(fallback=(100, 24)).columns - 2)
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent="agent> ",
        subsequent_indent=" " * 7,
        break_long_words=False,
        break_on_hyphens=False,
    )
    print(wrapper.fill(text))


def create_response(
    client: OpenAI,
    model: str,
    input_payload,
    previous_response_id: str | None,
    instructions: str,
    run_code_tool: dict,
):
    kwargs = {
        "model": model,
        "input": input_payload,
        "instructions": instructions,
        "tools": [run_code_tool],
        "reasoning": {"summary": "auto"},
        "tool_choice": "auto",
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    return client.responses.create(**kwargs)


def run_code(code: str, timeout_seconds: int = 10) -> str:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout_seconds}s."

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        return stdout
    return stderr or stdout or f"Process exited with code {completed.returncode}."


def execute_run_code_call(call) -> str:
    if getattr(call, "name", "") != "run_code":
        return f"Unknown code tool: {getattr(call, 'name', '')}"

    try:
        arguments = json.loads(getattr(call, "arguments", "{}"))
    except json.JSONDecodeError as error:
        return f"Tool arguments were not valid JSON.\n{error}"

    code = arguments.get("code") if isinstance(arguments, dict) else None
    if not isinstance(code, str):
        return "Tool arguments must include string field `code`."

    print_section("code", code)
    return run_code(code)


def print_reasoning_sections(response) -> None:
    for item in response.output:
        if getattr(item, "type", None) != "reasoning":
            continue

        summaries = [getattr(summary, "text", "") for summary in (getattr(item, "summary", []) or [])]
        summaries = [" ".join(text.replace("**", "").split()) for text in summaries]
        summaries = [text for text in summaries if text]
        if summaries:
            for text in summaries:
                print_section("reasoning", text)
            continue

        for content in (getattr(item, "content", []) or []):
            text = " ".join(getattr(content, "text", "").replace("**", "").split())
            if text:
                print_section("reasoning", text)


def run_agent_turn(client: OpenAI, prompt: str, model: str, previous_response_id: str | None) -> tuple[str, str]:
    run_code_tool = {
        "type": "function",
        "name": "run_code",
        "description": "Run Python code locally. Can read/write files (including /tmp) and return stdout/stderr.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute with `python -c`.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    }

    host_source = Path(__file__).read_text()
    instructions = f"""
You are an agent with one tool, `run_code`, which executes arbitrary Python in a local subprocess.
You can read and write local files from Python code.
Use `/tmp` as a persistent memory store across turns: create it if missing, write notes there, and read from it later when needed.
If a request depends on runtime information, local state, calculations, or stored memory, call `run_code` instead of guessing.
When uncertain, verify with `run_code` rather than fabricating.
Do not claim you lack runtime or filesystem access while `run_code` is available.
You may spawn helper agents by writing new Python files and invoking them from `run_code` (for example with subprocess), then collect and combine their outputs.

Current host file (quoted):
```python
{host_source}
```
""".strip()

    response = create_response(client, model, prompt, previous_response_id, instructions, run_code_tool)

    while True:
        print_reasoning_sections(response)
        calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not calls:
            return response.output_text, response.id

        tool_outputs = []
        for call in calls:
            output = execute_run_code_call(call)
            print_section("result", output)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": getattr(call, "call_id", ""),
                    "output": output,
                }
            )

        response = create_response(client, model, tool_outputs, response.id, instructions, run_code_tool)

def build_prompt_session() -> PromptSession:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    return PromptSession("user> ", multiline=True, key_bindings=bindings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal OpenAI Responses agent with one tool.")
    parser.add_argument("--model", default="gpt-5.2-codex", help="Responses API model.")
    args = parser.parse_args()

    client = OpenAI()
    prompt_session = build_prompt_session()
    previous_response_id: str | None = None

    while True:
        try:
            user_input = prompt_session.prompt()
        except KeyboardInterrupt:
            # Cancel current prompt input and continue; do not exit.
            print()
            continue
        except EOFError:
            print()
            break

        trimmed = user_input.strip()
        if not trimmed:
            continue
        if trimmed.lower() in {"exit", "quit"}:
            break

        try:
            answer, previous_response_id = run_agent_turn(
                client=client,
                prompt=user_input,
                model=args.model,
                previous_response_id=previous_response_id,
            )
        except KeyboardInterrupt:
            # Interrupt in-flight model/tool work and return to prompt.
            print("\n(interrupted current response)")
            continue

        print_agent_message(answer)


if __name__ == "__main__":
    main()
