"""Turn a session report (report.py) into written feedback for a coach, by
asking Claude to interpret it.

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
- A tracked "identity" is one continuous stretch of one person on screen, not a named \
player. Never talk about "Player 3" as if the system knows who that is.

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
