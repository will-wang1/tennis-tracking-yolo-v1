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


class LocalQwenTest(unittest.TestCase):
    def test_compact_report_drops_raw_arrays_and_keeps_what_feedback_needs(self):
        from src.session.feedback import compact_report

        report = {"clip": {"name": "x", "analysed_s": 100}, "raw": {"ball_tracks": [1] * 1000},
                  "metrics": {"ball_in_play": {"value": 0.8, "unit": "share", "basis": "estimated", "method": "m",
                                               "caveats": ["c"]}},
                  "summary": {"headline": "h", "targets": {"a": 1}}, "feeding": {"rallies": 3, "rally_list": [1, 2]},
                  "players": [{"player": 2, "shots": 5, "zone_shares": {"net": 0.1}}]}
        c = compact_report(report)
        self.assertNotIn("raw", c)
        self.assertEqual(c["metrics"]["ball_in_play"], {"value": 0.8, "unit": "share", "basis": "estimated", "caveats": ["c"]})
        self.assertNotIn("rally_list", c["feeding"])
        self.assertNotIn("targets", c["summary"])
        self.assertEqual(c["players"][0]["shots"], 5)

    def test_asks_ollama_for_the_schema_and_labels_the_model(self):
        import json as _json

        from src.session.feedback import FEEDBACK_SCHEMA, request_feedback_local

        seen = {}

        def post(url, payload, timeout):
            seen.update(url=url, payload=payload)
            return {"message": {"content": _json.dumps(GOOD)}}

        out = request_feedback_local(REPORT, model="qwen3:8b", host="http://h:1", post=post)
        self.assertEqual(seen["url"], "http://h:1/api/chat")
        self.assertEqual(seen["payload"]["format"], FEEDBACK_SCHEMA)
        self.assertFalse(seen["payload"]["think"])
        self.assertEqual(out["headline"], GOOD["headline"])
        self.assertIn("qwen3:8b", out["model"])

    def test_bad_json_or_missing_fields_fail_loudly(self):
        from src.session.feedback import request_feedback_local

        with self.assertRaises(RuntimeError):
            request_feedback_local(REPORT, post=lambda u, p, t: {"message": {"content": "not json"}})
        with self.assertRaises(RuntimeError):
            request_feedback_local(REPORT, post=lambda u, p, t: {"message": {"content": '{"headline": "x"}'}})


class MlxQwenTest(unittest.TestCase):
    def test_fenced_json_is_accepted_and_labelled(self):
        import json as _json

        from src.session.feedback import request_feedback_mlx

        out = request_feedback_mlx(REPORT, generate_fn=lambda m: "```json\n" + _json.dumps(GOOD) + "\n```")
        self.assertEqual(out["headline"], GOOD["headline"])
        self.assertIn("MLX", out["model"])

    def test_one_retry_then_a_loud_failure(self):
        import json as _json

        from src.session.feedback import request_feedback_mlx

        replies = iter(["sorry, here you go: not json", _json.dumps(GOOD)])
        calls = []

        def gen(messages):
            calls.append(len(messages))
            return next(replies)

        self.assertEqual(request_feedback_mlx(REPORT, generate_fn=gen)["headline"], GOOD["headline"])
        self.assertEqual(calls, [2, 4])  # the retry sees its own bad reply and the complaint
        with self.assertRaises(RuntimeError):
            request_feedback_mlx(REPORT, generate_fn=lambda m: "never json")
