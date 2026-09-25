"""Render a session report (and, if present, its feedback) as one
self-contained HTML page for a coach - the page that replaced the old
match scoreboard (see src/session/coach_report_template.html).

    python scripts/render_coach_report.py \\
        --report outputs/dingles_serve_volley/session_report.json \\
        --feedback outputs/dingles_serve_volley/session_feedback.json \\
        --out outputs/dingles_serve_volley/coach_report.html

The page embeds the report's data and draws everything client-side, so the
file opens straight from disk. The only external request is Google Fonts,
with system fonts as the fallback.
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "src" / "session" / "coach_report_template.html"


def _embed(data) -> str:
    # "</" inside a <script> block would end it early if any string in the
    # data ever contained "</script>"; escaping the slash is inert in JSON.
    return json.dumps(data).replace("</", "<\\/")


def render(report: dict, feedback, title: str) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    return (
        html.replace("__TITLE__", title)
        .replace("__REPORT__", _embed(report))
        .replace("__FEEDBACK__", _embed(feedback))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--feedback", help="scripts/session_feedback.py output (optional)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", help="Page title (default: from the clip name)")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    feedback = json.loads(Path(args.feedback).read_text()) if args.feedback and Path(args.feedback).exists() else None
    title = args.title or report["clip"]["name"].replace("_", " ").title() + " Report"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(report, feedback, title), encoding="utf-8")
    print(f"Wrote {out}" + ("" if feedback else "  (no feedback section)"))


if __name__ == "__main__":
    main()
