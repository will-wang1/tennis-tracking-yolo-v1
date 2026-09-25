"""Write coaching feedback for a session report by asking Claude to read it
(src/session/feedback.py).

    python scripts/session_feedback.py \\
        --report outputs/dingles_serve_volley/session_report.json \\
        --out outputs/dingles_serve_volley/session_feedback.json

Needs Anthropic API credentials: ANTHROPIC_API_KEY, or `ant auth login`.
Without them it writes the exact prompt to `<out>.prompt.txt` instead, so
the same feedback can be produced by pasting it into Claude.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.session.feedback import FeedbackRefused, offline_prompt, request_feedback  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--offline", action="store_true", help="Only write the prompt; make no API call")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    have_credentials = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) or (
        Path.home() / ".config" / "anthropic"
    ).exists()
    if args.offline or not have_credentials:
        prompt_path = out.with_suffix(".prompt.txt")
        prompt_path.write_text(offline_prompt(report))
        reason = "--offline" if args.offline else "no Anthropic API credentials found"
        print(f"{reason}: wrote the prompt to {prompt_path}")
        print("Set ANTHROPIC_API_KEY (or run `ant auth login`) and re-run to call the API directly.")
        return

    import anthropic

    try:
        feedback = request_feedback(report, effort=args.effort)
    except FeedbackRefused as error:
        raise SystemExit(str(error))
    except anthropic.AuthenticationError:
        raise SystemExit("The API rejected the credentials - check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise SystemExit("Rate limited - wait a minute and re-run.")
    except anthropic.APIStatusError as error:
        raise SystemExit(f"API error {error.status_code}: {error.message}")
    except anthropic.APIConnectionError:
        raise SystemExit("Could not reach the API - check the network.")

    out.write_text(json.dumps(feedback, indent=1))
    print(f"Wrote {out}\n")
    print(feedback["headline"])


if __name__ == "__main__":
    main()
