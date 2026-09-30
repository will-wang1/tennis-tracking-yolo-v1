"""Turn a session report (report.py) into written feedback for a coach, by
asking a language model to interpret it: Qwen running on this computer
through Ollama (the default), or Claude through the Anthropic API.

The report does the measuring; this does the reading. The model is given
the whole report and told the one rule the report is built around: a value
marked "estimated" or "lower_bound" must be described as such, never stated
as fact. The output is STRUCTURED (a fixed JSON shape, below) rather than
free prose, so a report page can lay out each part reliably and every
observation arrives tied to the metric it rests on.

Scope, stated in the prompt and repeated here because it is the easiest
thing for generated feedback to overstep: the data measures ACTIVITY -
balls, movement, time in play. It cannot see whether a drill was well
chosen, whether a correction was right, or whether anyone improved.
Feedback that implies otherwise is inventing, however fluent it reads.
"""

import json
from typing import Any

MODEL = "claude-opus-5"
# Opt into server-side refusal fallback: a declined request is re-run on a
# fallback model inside the same call instead of returning nothing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

FEEDBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One or two sentences a coach reads first.",
        },
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "evidence": {
                        "type": "string",
                        "description": "The metric names and values this rests on, with their basis.",
                    },
                    "kind": {"type": "string", "enum": ["strength", "worth_a_look", "neutral"]},
                },
                "required": ["point", "evidence", "kind"],
                "additionalProperties": False,
            },
        },
        "questions_for_the_coach": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Things only the coach can answer from knowing the session plan.",
        },
        "data_limits": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What this data could not show about this session, in plain words.",
        },
    },
    "required": ["headline", "observations", "questions_for_the_coach", "data_limits"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You help tennis coaches review their sessions. You receive a session \
report produced by a computer-vision system from a fixed camera behind the baseline, \
and you write feedback a coach can act on.

How to read the report:
- Every metric carries a `basis`. "measured" values you may state plainly. "estimated" \
values are real readings with a known error - present them as approximate. "lower_bound" \
values are minimums - say "at least", and use the caveat to explain the gap.
- Each metric's `caveats` list its known failure modes. If a caveat affects a point you \
make, say so in that point's evidence.
- "Player 3" is the same person all session (the system joins them up by their \
clothes), but the system does not know their name. Refer to players by number. \
The player marked as the coach is a guess unless the report says it was set.

What the report can and cannot tell you:
- It measures activity: time the ball is in play, how often it is hit or bounces, where \
it lands, how far people move.
- It cannot see drill choice, instruction quality, technique, or improvement. Do not \
infer them. Where an activity number raises a question only the coach can answer (for \
example, a low ball-in-play share could be deliberate teaching time), ask it in \
`questions_for_the_coach` instead of judging.

Write for a working coach: short, concrete, no jargon from the report's field names \
unless you quote them as evidence. Prefer three well-supported observations to eight \
thin ones."""


# ---- local Qwen through Ollama --------------------------------------------

LOCAL_MODEL = "qwen3:8b"  # ~5GB; runs on a 16GB M2 alongside the rest
OLLAMA = "http://localhost:11434"
LOCAL_CONTEXT = 16384  # tokens; the compact report is a few thousand


def compact_report(report: dict) -> dict:
    """The report without its raw arrays (ball tracks, events, heatmaps),
    which make the whole thing ~40k tokens - too much for a local model on
    16GB, and not needed to write feedback: the verdicts, metrics with their
    basis and caveats, players, themes and fed/live play all stay."""
    clip = report.get("clip", {})
    metrics = {
        k: {f: m.get(f) for f in ("value", "unit", "basis", "caveats") if f in m}
        for k, m in (report.get("metrics") or {}).items()
    }
    players = [
        {k: p.get(k) for k in ("player", "on_camera_s", "distance_m", "moving_share", "shots", "shots_per_minute",
                               "queue_share", "longest_queue_s", "idle_share", "zone_shares") if k in p}
        for p in report.get("players") or []
    ]
    feeding = {k: v for k, v in (report.get("feeding") or {}).items() if k != "rally_list"}
    summary = {k: v for k, v in (report.get("summary") or {}).items() if k not in ("targets",)}
    return {
        "clip": {k: clip.get(k) for k in ("name", "duration_s", "analysed_s", "cutaways_s") if k in clip},
        "summary": summary,
        "themes": report.get("themes") or [],
        "metrics": metrics,
        "players": players,
        "feeding": feeding or None,
        "segments": report.get("segments") or [],
    }


def request_feedback_local(report: dict, model: str = LOCAL_MODEL, host: str = OLLAMA, post=None,
                           timeout: float = 900) -> dict:
    """Ask a model served by Ollama on this computer, constrained to
    FEEDBACK_SCHEMA. `post(url, payload) -> dict` is injectable for tests."""
    if post is None:
        post = _post_json
    payload = {
        "model": model,
        "stream": False,
        "think": False,  # Qwen3's reasoning mode: slow on a laptop, and the schema does the structuring
        "format": FEEDBACK_SCHEMA,
        "options": {"num_ctx": LOCAL_CONTEXT, "temperature": 0.3},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(compact_report(report))},
        ],
    }
    reply = post(f"{host}/api/chat", payload, timeout)
    text = (reply.get("message") or {}).get("content") or ""
    try:
        feedback = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{model} did not return JSON: {text[:200]!r}") from error
    missing = [k for k in FEEDBACK_SCHEMA["required"] if k not in feedback]
    if missing:
        raise RuntimeError(f"{model} left out {missing}")
    feedback["model"] = f"{model} (on this computer, via Ollama)"
    return feedback


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    import urllib.request

    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def local_model_status(model: str = LOCAL_MODEL, host: str = OLLAMA) -> str:
    """'ready', 'no server' (Ollama not installed or not running) or
    'not pulled' (running, but the model is not downloaded)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as response:
            names = {m.get("name") for m in json.loads(response.read()).get("models", [])}
    except OSError:
        return "no server"
    return "ready" if model in names or f"{model}:latest" in names else "not pulled"


def build_user_message(report: dict) -> str:
    """The report is sent whole - including raw events and its own
    `how_to_read` - rather than a summary of it, so the model can check
    any number it is about to lean on."""
    return (
        "Here is the session report. Write feedback following your instructions.\n\n"
        "<session_report>\n"
        f"{json.dumps(report, indent=1)}\n"
        "</session_report>"
    )


class FeedbackRefused(RuntimeError):
    """The model (and its fallback) declined the request."""


def request_feedback(report: dict, client=None, effort: str = "high") -> dict:
    """Call Claude and return the parsed feedback dict (FEEDBACK_SCHEMA).

    `client` defaults to `anthropic.Anthropic()`, which picks up
    ANTHROPIC_API_KEY or an `ant auth login` profile. Passing one in is
    how tests run this without a network.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": FEEDBACK_SCHEMA},
        },
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(report)}],
    )

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise FeedbackRefused(f"Feedback request declined (category: {category})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Feedback was cut off at max_tokens; the JSON is incomplete")

    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise RuntimeError(f"No text in the response (stop_reason={response.stop_reason})")
    return json.loads(text)


def offline_prompt(report: dict) -> str:
    """The exact request as plain text, for when no API credentials are
    available: paste it into Claude and the reply has the same content the
    API call would return (without the schema guarantee)."""
    shape = json.dumps(FEEDBACK_SCHEMA, indent=1)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Reply with JSON matching this schema:\n{shape}\n\n"
        f"{build_user_message(report)}\n"
    )
