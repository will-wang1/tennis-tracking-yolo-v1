import json
import unittest
from types import SimpleNamespace

from src.session.feedback import (
    FEEDBACK_SCHEMA,
    MODEL,
    FeedbackRefused,
    build_user_message,
    offline_prompt,
    request_feedback,
)

REPORT = {
    "schema_version": "0.1",
    "how_to_read": "basis rules",
    "metrics": {"ball_in_play": {"value": 0.79, "unit": "fraction", "basis": "estimated", "method": "m", "caveats": []}},
}

GOOD = {
    "headline": "Busy session.",
    "observations": [{"point": "Ball in play most of the time", "evidence": "ball_in_play 0.79 (estimated)", "kind": "strength"}],
    "questions_for_the_coach": ["Was the pause at 30s planned?"],
    "data_limits": ["Cannot see technique."],
}


class _FakeClient:
    """Stands in for anthropic.Anthropic(): records the request, returns a
    canned response. No network."""

    def __init__(self, response):
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return response

        self.beta = SimpleNamespace(messages=_Messages())


def _response(text=None, stop_reason="end_turn", category=None):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    details = SimpleNamespace(category=category) if stop_reason == "refusal" else None
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=details)


class RequestFeedbackTest(unittest.TestCase):
    def test_returns_the_parsed_structured_feedback(self):
        client = _FakeClient(_response(json.dumps(GOOD)))
        self.assertEqual(request_feedback(REPORT, client=client), GOOD)

    def test_request_shape(self):
        client = _FakeClient(_response(json.dumps(GOOD)))
        request_feedback(REPORT, client=client, effort="medium")
        call = client.calls[0]

        self.assertEqual(call["model"], MODEL)
        self.assertEqual(call["fallbacks"], "default")
        self.assertEqual(call["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(call["thinking"], {"type": "adaptive"})
        self.assertEqual(call["output_config"]["effort"], "medium")
        self.assertEqual(call["output_config"]["format"]["schema"], FEEDBACK_SCHEMA)
        self.assertIn('"ball_in_play"', call["messages"][0]["content"])

    def test_a_refusal_raises_instead_of_parsing_nothing(self):
        client = _FakeClient(_response(stop_reason="refusal", category="bio"))
        with self.assertRaises(FeedbackRefused):
            request_feedback(REPORT, client=client)

    def test_truncated_output_is_an_error_not_bad_json(self):
        client = _FakeClient(_response('{"headline": "cut', stop_reason="max_tokens"))
        with self.assertRaises(RuntimeError):
            request_feedback(REPORT, client=client)


class PromptTest(unittest.TestCase):
    def test_the_whole_report_is_sent_not_a_summary(self):
        self.assertEqual(json.loads(build_user_message(REPORT).split("<session_report>\n")[1].split("\n</session_report>")[0]), REPORT)

    def test_offline_prompt_carries_the_rules_and_the_schema(self):
        text = offline_prompt(REPORT)
        self.assertIn("lower_bound", text)
        self.assertIn('"questions_for_the_coach"', text)

    def test_schema_is_strict(self):
        self.assertFalse(FEEDBACK_SCHEMA["additionalProperties"])
        self.assertEqual(set(FEEDBACK_SCHEMA["required"]), set(FEEDBACK_SCHEMA["properties"]))


if __name__ == "__main__":
    unittest.main()
