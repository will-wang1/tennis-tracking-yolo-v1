import importlib.util
import json
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "render_coach_report", Path(__file__).resolve().parent.parent / "scripts" / "render_coach_report.py"
)
render_coach_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(render_coach_report)


class RenderTest(unittest.TestCase):
    REPORT = {"clip": {"name": "x"}, "metrics": {}, "note": "a </script><b>bold</b> string"}

    def test_every_placeholder_is_filled(self):
        html = render_coach_report.render(self.REPORT, None, "Some Title")
        for placeholder in ("__REPORT__", "__FEEDBACK__", "__TITLE__"):
            self.assertNotIn(placeholder, html)
        self.assertIn("<title>Some Title</title>", html)

    def test_data_cannot_close_the_script_block(self):
        html = render_coach_report.render(self.REPORT, None, "T")
        script = html.split("const R = ", 1)[1]
        self.assertNotIn("</script><b>", script.split(";\nconst F")[0])

    def test_missing_feedback_embeds_null(self):
        html = render_coach_report.render(self.REPORT, None, "T")
        self.assertIn("const F = null;", html)

    def test_embedded_report_round_trips(self):
        html = render_coach_report.render(self.REPORT, None, "T")
        embedded = html.split("const R = ", 1)[1].split(";\nconst F", 1)[0]
        self.assertEqual(json.loads(embedded.replace("<\\/", "</")), self.REPORT)


if __name__ == "__main__":
    unittest.main()
