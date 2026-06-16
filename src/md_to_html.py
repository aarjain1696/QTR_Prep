#!/usr/bin/env python3
"""Convert Markdown files to standalone, nicely-styled HTML with LaTeX math.

Why this exists: the Week-1 write-ups are heavy on LaTeX ($...$, $$...$$), tables,
and blockquotes. This renders them to self-contained HTML you can open in a browser
(math via MathJax CDN), so you don't have to re-generate HTML by hand each time.

Usage
-----
    # defaults to the two Week-1 .md files in ../notebooks
    python src/md_to_html.py

    # or pass any markdown files explicitly
    python src/md_to_html.py notebooks/Week1-Volatility_Surface_Explained.md

    # choose an output directory (default: alongside each source file)
    python src/md_to_html.py docs/*.md --outdir build/html
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown

# Markdown extensions. `pymdownx.arithmatex` (generic mode) emits \( \) / \[ \]
# delimiters for MathJax and is "smart" about $ — it won't mistake "$50 ... $300"
# in prose for inline math, which a naive regex would.
EXTENSIONS = [
    "pymdownx.arithmatex",
    "tables",
    "fenced_code",
    "sane_lists",
    "toc",
    "attr_list",
]
EXTENSION_CONFIGS = {"pymdownx.arithmatex": {"generic": True}}

# Self-contained page: embedded CSS for readability + MathJax from CDN.
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
  :root {{ --fg:#1b1f24; --muted:#57606a; --bg:#ffffff; --soft:#f6f8fa;
           --border:#d0d7de; --accent:#0969da; --code:#eaeef2; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e6edf3; --muted:#9198a1; --bg:#0d1117; --soft:#161b22;
             --border:#30363d; --accent:#4493f8; --code:#1f2630; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
          line-height:1.65; font-size:17px; }}
  main {{ max-width:880px; margin:0 auto; padding:48px 24px 120px; }}
  h1,h2,h3,h4 {{ line-height:1.25; margin-top:1.8em; margin-bottom:.6em; }}
  h1 {{ font-size:2.1em; margin-top:.2em; }}
  h2 {{ font-size:1.5em; padding-bottom:.3em; border-bottom:1px solid var(--border); }}
  h3 {{ font-size:1.2em; }}
  p, li {{ color:var(--fg); }}
  a {{ color:var(--accent); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  hr {{ border:none; border-top:1px solid var(--border); margin:2.4em 0; }}
  blockquote {{ margin:1.2em 0; padding:.4em 1.1em; color:var(--muted);
                border-left:4px solid var(--accent); background:var(--soft); border-radius:6px; }}
  blockquote p:first-child {{ margin-top:0; }} blockquote p:last-child {{ margin-bottom:0; }}
  code {{ background:var(--code); padding:.15em .4em; border-radius:6px;
          font-family:"SF Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.88em; }}
  pre {{ background:var(--soft); border:1px solid var(--border); border-radius:8px;
         padding:14px 16px; overflow-x:auto; }}
  pre code {{ background:none; padding:0; font-size:.86em; }}
  table {{ border-collapse:collapse; width:100%; margin:1.3em 0; font-size:.94em; display:block; overflow-x:auto; }}
  th, td {{ border:1px solid var(--border); padding:8px 12px; text-align:left; vertical-align:top; }}
  thead th {{ background:var(--soft); }}
  tbody tr:nth-child(even) {{ background:var(--soft); }}
  mjx-container {{ overflow-x:auto; overflow-y:hidden; max-width:100%; }}
  .doc-footer {{ margin-top:4em; padding-top:1.2em; border-top:1px solid var(--border);
                 color:var(--muted); font-size:.85em; }}
</style>
</head>
<body>
<main>
{body}
<div class="doc-footer">Rendered from <code>{source}</code> · math by MathJax</div>
</main>
</body>
</html>
"""

DEFAULT_FILES = [
    "notebooks/Week1-Volatility_Surface_Explained.md",
    "notebooks/Week1-Volatility_Surface_Review.md",
]


def _isolate_display_math(text: str) -> str:
    """Surround every $$...$$ block with blank lines.

    In the source, display equations sit on a line that directly touches the
    surrounding prose. Without a blank line the Markdown parser folds them into
    the paragraph and arithmatex renders them *inline* instead of as centered
    *display* math. Promoting them to standalone blocks fixes that. Single-$ inline
    math and literal '$50' prose are untouched (this only matches doubled $$).
    """
    text = re.sub(r"\$\$(.+?)\$\$",
                  lambda m: "\n\n$$" + m.group(1).strip() + "$$\n\n",
                  text, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text)


def _title_from(md_text: str, fallback: str) -> str:
    """Use the first level-1 heading as the page <title>, else the file name."""
    for line in md_text.splitlines():
        m = re.match(r"#\s+(.+)", line)
        if m:
            return re.sub(r"[*`_]", "", m.group(1)).strip()
    return fallback


def convert(md_path: Path, out_path: Path) -> Path:
    md_text = md_path.read_text(encoding="utf-8")
    md_text = _isolate_display_math(md_text)
    # Fresh converter per file so `toc` ids and state don't leak between docs.
    md = markdown.Markdown(extensions=EXTENSIONS, extension_configs=EXTENSION_CONFIGS)
    body = md.convert(md_text)
    html = HTML_TEMPLATE.format(
        title=_title_from(md_text, md_path.stem),
        body=body,
        source=md_path.name,
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Markdown to standalone HTML (with MathJax).")
    ap.add_argument("files", nargs="*", help="Markdown files (default: the two Week-1 write-ups).")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Output directory (default: next to each source file).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    raw = args.files or DEFAULT_FILES
    paths = [Path(f) if Path(f).is_absolute() else repo_root / f for f in raw]

    outdir = Path(args.outdir).resolve() if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for src in paths:
        if not src.exists():
            print(f"  skip (not found): {src}")
            continue
        dst = (outdir / f"{src.stem}.html") if outdir else src.with_suffix(".html")
        convert(src, dst)
        print(f"  {src.name}  ->  {dst.relative_to(repo_root) if dst.is_relative_to(repo_root) else dst}")


if __name__ == "__main__":
    main()
