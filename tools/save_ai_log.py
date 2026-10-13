"""Save the current Claude Code conversation into docs/ai-logs.

Run automatically by the Stop hook in .claude/settings.json after every
assistant reply. Reads the hook payload (JSON) from stdin, copies the raw
transcript (JSONL) and writes a human-readable Markdown version.

Can also be run by hand on any transcript:
    python tools/save_ai_log.py path/to/transcript.jsonl
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "docs" / "ai-logs" / "raw"
READABLE_DIR = ROOT / "docs" / "ai-logs" / "readable"
# App-injected context notes; kept in the raw copy, dropped from the readable one.
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def text_of(content):
    """Flatten a message's content (string or list of blocks) to Markdown."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block["text"])
        elif kind == "tool_use":
            args = json.dumps(block.get("input", {}), ensure_ascii=False, indent=2)
            parts.append(f"**[tool call: {block.get('name')}]**\n```json\n{args}\n```")
        elif kind == "tool_result":
            result = block.get("content")
            if isinstance(result, list):
                result = "\n".join(b.get("text", f"[{b.get('type')}]") for b in result)
            parts.append(f"**[tool result]**\n```\n{str(result)[:4000]}\n```")
        elif kind == "image":
            parts.append("*[image attached]*")
    return "\n\n".join(parts)


def to_markdown(transcript: Path) -> str:
    lines = [f"# AI conversation log: {transcript.stem}\n"]
    for raw in transcript.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("type") not in ("user", "assistant") or entry.get("isMeta"):
            continue
        msg = entry.get("message", {})
        body = REMINDER.sub("", text_of(msg.get("content"))).strip()
        if not body:
            continue
        who = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"## {who} ({entry.get('timestamp', '')})\n\n{body}\n")
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        transcript = Path(sys.argv[1])
    else:
        payload = json.load(sys.stdin)
        transcript = Path(payload["transcript_path"])
    if not transcript.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    READABLE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript, RAW_DIR / transcript.name)
    (READABLE_DIR / f"{transcript.stem}.md").write_text(to_markdown(transcript), encoding="utf-8")


if __name__ == "__main__":
    main()
