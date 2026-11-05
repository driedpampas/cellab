"""Write redacted copies of AI conversation logs into docs/ai-logs/public/ for the repository.

The raw transcripts (docs/ai-logs/raw/) stay local. This script turns a transcript into
the same readable Markdown as save_ai_log.py, then removes personal details:

- e-mail addresses (except the noreply address in commit co-author lines);
- every text listed in docs/ai-logs/redact.txt, one rule per line, written as
      text => replacement
  and matched without regard to case. That file is git-ignored, so the list itself
  is never published.

Usage:
    python tools/publish_ai_logs.py SESSION_ID...   publish these sessions
    python tools/publish_ai_logs.py                 refresh the sessions already published

Read the output before committing it: the script only removes what it is told to.
"""
import re
import sys
from pathlib import Path

from save_ai_log import RAW_DIR, ROOT, to_markdown

PUBLIC_DIR = ROOT / "docs" / "ai-logs" / "public"
RULES_FILE = ROOT / "docs" / "ai-logs" / "redact.txt"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|ro|org|net|edu|io|dev|info|eu|uk|de)\b")
KEEP_EMAILS = {"noreply@anthropic.com"}
NOTE = ("> Published copy. Personal details (names, e-mail addresses, account names) were replaced\n"
        "> with placeholders such as [member 1]; nothing else was changed.\n")


def read_rules() -> list[tuple[re.Pattern, str]]:
    if not RULES_FILE.exists():
        sys.exit(f"{RULES_FILE.relative_to(ROOT)} is missing: list the texts to remove first (see this script's docstring).")
    rules = []
    for line in RULES_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            text, _, replacement = line.partition("=>")
            rules.append((text.strip(), replacement.strip()))
    rules.sort(key=lambda rule: len(rule[0]), reverse=True)   # longest first: full names before parts
    return [(re.compile(re.escape(text), re.IGNORECASE), replacement) for text, replacement in rules]


def redact(text: str, rules) -> str:
    text = EMAIL.sub(lambda m: m.group(0) if m.group(0).lower() in KEEP_EMAILS else "[email]", text)
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


def main():
    rules = read_rules()
    sessions = sys.argv[1:] or sorted(path.stem for path in PUBLIC_DIR.glob("*.md"))
    if not sessions:
        sys.exit("Nothing to publish: give the session id(s) to publish.")
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        transcript = RAW_DIR / f"{session}.jsonl"
        if not transcript.exists():
            sys.exit(f"No raw transcript {transcript.relative_to(ROOT)}")
        title, _, body = to_markdown(transcript).partition("\n")
        text = redact(f"{title}\n\n{NOTE}{body}", rules)
        left = [pattern.pattern for pattern, _ in rules if pattern.search(text)] + EMAIL.findall(text)
        left = [item for item in left if item.lower() not in KEEP_EMAILS]
        if left:
            sys.exit(f"{session}: still contains {left}; nothing written")
        (PUBLIC_DIR / f"{session}.md").write_text(text, encoding="utf-8")
        print(f"{session}: {len(text):,} characters written to {(PUBLIC_DIR / session).relative_to(ROOT)}.md")


if __name__ == "__main__":
    main()
