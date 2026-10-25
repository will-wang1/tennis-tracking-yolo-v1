"""Write coaching feedback for a session report by asking a language model to
read it (src/session/feedback.py).

By default the model is Qwen3 running on this computer, so nothing leaves
the machine - through Apple's MLX on a Mac (pip install mlx-lm; the model,
~4.6GB, downloads from Hugging Face on first use), or through Ollama
anywhere else (the Acer server: FEEDBACK_MODEL=qwen3:4b, which fits its
4GB GPU). FEEDBACK_BACKEND / FEEDBACK_MODEL, or --backend / --model, choose:

    python scripts/session_feedback.py \\
        --report outputs/dingles_serve_volley/session_report.json \\
        --out outputs/dingles_serve_volley/session_feedback.json

If the model cannot run here, this writes the exact prompt to
`<out>.prompt.txt` instead and says what to do.

--provider claude uses the Anthropic API instead (needs ANTHROPIC_API_KEY).
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.session.feedback import (  # noqa: E402
    OLLAMA, FeedbackRefused, compact_report, default_backend, default_model, feedback_status, offline_prompt,
    request_feedback, request_feedback_local, request_feedback_mlx,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", default="qwen", choices=["qwen", "claude"])
    parser.add_argument("--backend", default=None, choices=["mlx", "ollama"],
                        help="How to run Qwen here (default: FEEDBACK_BACKEND, else mlx on a Mac, ollama elsewhere)")
    parser.add_argument("--model", default=None, help="Model to run (default: FEEDBACK_MODEL, else per backend)")
    parser.add_argument("--host", default=OLLAMA, help="Where Ollama is listening")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                        help="With --provider claude")
    parser.add_argument("--offline", action="store_true", help="Only write the prompt; call no model")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = out.with_suffix(".prompt.txt")

    if args.provider == "qwen":
        backend = args.backend or default_backend()
        model = args.model or default_model(backend)
        if args.offline:
            status = "offline"
        else:
            status = feedback_status(backend, model, args.host)
            if status == "not downloaded":
                print(f"Downloading {model} (~4.6GB, once)...", flush=True)
                status = "ready"  # MLX's load() fetches it
        if status != "ready":
            prompt_path.write_text(offline_prompt(compact_report(report)))
            print(f"Wrote the prompt to {prompt_path} instead of calling {model}:")
            hints = {"no mlx": "  MLX is not installed: pip install mlx-lm (needs an Apple silicon Mac)",
                     "no server": "  Ollama is not running: install it from https://ollama.com/download and open it",
                     "not pulled": f"  Download the model: ollama pull {model}"}
            if status in hints:
                print(hints[status])
            return
        print(f"Asking {model} on this computer - a minute or two...", flush=True)
        try:
            if backend == "mlx":
                feedback = request_feedback_mlx(report, model_id=model)
            else:
                feedback = request_feedback_local(report, model=model, host=args.host)
        except (RuntimeError, OSError) as error:
            raise SystemExit(f"{model} failed: {error}")
    else:
        have_credentials = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) or (
            Path.home() / ".config" / "anthropic"
        ).exists()
        if args.offline or not have_credentials:
            prompt_path.write_text(offline_prompt(report))
            print(f"No Anthropic API credentials: wrote the prompt to {prompt_path}")
            return
        import anthropic

        try:
            feedback = request_feedback(report, effort=args.effort)
        except FeedbackRefused as error:
            raise SystemExit(str(error))
        except anthropic.APIError as error:
            raise SystemExit(f"Anthropic API error: {error}")
        feedback["model"] = "Claude (Anthropic API)"

    out.write_text(json.dumps(feedback, indent=1))
    print(f"Wrote {out}\n")
    print(feedback["headline"])


if __name__ == "__main__":
    main()
