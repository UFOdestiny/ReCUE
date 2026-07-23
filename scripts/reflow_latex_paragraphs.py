#!/usr/bin/env python3
"""Join hard-wrapped LaTeX prose while preserving structural environments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PROTECTED_ENVIRONMENTS = {
    "CCSXML",
    "algorithm",
    "algorithmic",
    "align",
    "aligned",
    "cases",
    "equation",
    "gather",
    "gathered",
    "lstlisting",
    "split",
    "tabular",
    "verbatim",
}

STANDALONE_COMMANDS = re.compile(
    r"^\\(?:"
    r"appendix|balance|bibliography|bibliographystyle|bottomrule|centering|clearpage|"
    r"cmidrule|documentclass|else|endinput|fi|hfill|input|label|maketitle|midrule|"
    r"newcommand|newpage|pdfminorversion|providecommand|renewcommand|rowcolor|rule|"
    r"scriptsize|setcopyright|setlength|settopmatter|small|toprule|vspace|"
    r"section|subsection|subsubsection|paragraph|title|author|acmConference|acmYear|"
    r"copyrightyear|keywords|ccsdesc|usepackage|AtBeginDocument"
    r")\b"
)

JOINED_BRACE_COMMANDS = re.compile(r"^\\(?:caption|Description)\s*\{")
PROSE_COMMANDS = (
    r"\m",
    r"\pap",
    r"\dte",
    r"\ccf",
    r"\textbf",
    r"\emph",
    r"\texttt",
    r"\cite",
    r"\ref",
    r"\eqref",
    r"\url",
    r"\footnote",
)


def brace_delta(line: str) -> int:
    """Count unescaped braces on one source line."""
    delta = 0
    for match in re.finditer(r"(?<!\\)([{}])", line):
        delta += 1 if match.group(1) == "{" else -1
    return delta


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def reflow(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    protected_stack: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            output.append(" ".join(part.strip() for part in paragraph))
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        begin_match = re.match(r"^\\begin\{([^}]+)\}", stripped)
        end_match = re.match(r"^\\end\{([^}]+)\}", stripped)

        if protected_stack:
            flush_paragraph()
            output.append(line.rstrip())
            if begin_match and begin_match.group(1) in PROTECTED_ENVIRONMENTS:
                protected_stack.append(begin_match.group(1))
            if end_match and end_match.group(1) == protected_stack[-1]:
                protected_stack.pop()
            index += 1
            continue

        if begin_match and begin_match.group(1) in PROTECTED_ENVIRONMENTS:
            flush_paragraph()
            output.append(line.rstrip())
            protected_stack.append(begin_match.group(1))
            index += 1
            continue

        if JOINED_BRACE_COMMANDS.match(stripped):
            flush_paragraph()
            pieces = [stripped]
            depth = brace_delta(stripped)
            while depth > 0 and index + 1 < len(lines):
                index += 1
                continuation = lines[index].strip()
                pieces.append(continuation)
                depth += brace_delta(continuation)
            output.append(" ".join(pieces))
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            output.append("")
            index += 1
            continue

        if stripped.startswith("%"):
            flush_paragraph()
            output.append(line.rstrip())
            index += 1
            continue

        if stripped.startswith(r"\item"):
            flush_paragraph()
            paragraph.append(stripped)
            index += 1
            continue

        is_explicit_break = stripped.endswith(r"\\")
        is_environment_boundary = begin_match is not None or end_match is not None
        is_standalone = STANDALONE_COMMANDS.match(stripped) is not None
        is_unknown_command = stripped.startswith("\\") and not stripped.startswith(PROSE_COMMANDS)

        if is_explicit_break or is_environment_boundary or is_standalone or is_unknown_command:
            flush_paragraph()
            output.append(line.rstrip())
        else:
            paragraph.append(stripped)
        index += 1

    flush_paragraph()
    result = "\n".join(output) + ("\n" if text.endswith("\n") else "")
    if normalize_whitespace(result) != normalize_whitespace(text):
        raise RuntimeError("Reflow changed non-whitespace content")
    return result


def default_files(root: Path) -> list[Path]:
    files = sorted((root / "latex" / "chap").glob("*.tex"))
    files.extend(sorted((root / "latex" / "table").glob("*.tex")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Rewrite files in place")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    files = args.files or default_files(root)
    changed = 0
    removed_lines = 0
    for path in files:
        original = path.read_text()
        formatted = reflow(original)
        if formatted == original:
            continue
        changed += 1
        removed_lines += original.count("\n") - formatted.count("\n")
        print(path.relative_to(root))
        if args.write:
            path.write_text(formatted)
    mode = "rewritten" if args.write else "would rewrite"
    print(f"{mode}: {changed} files; removed hard wraps: {removed_lines}")


if __name__ == "__main__":
    main()
