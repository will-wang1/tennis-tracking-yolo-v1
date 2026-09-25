import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_session", Path(__file__).resolve().parent.parent / "scripts" / "run_session.py"
)
run_session = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_session)


class StepTest(unittest.TestCase):
    def test_a_step_whose_output_exists_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "done.txt"
            out.write_text("already here")
            marker = Path(tmp) / "ran.txt"
            run_session.step("x", out, [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('1')"], force=False)
            self.assertFalse(marker.exists())

    def test_force_reruns_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "done.txt"
            out.write_text("already here")
            marker = Path(tmp) / "ran.txt"
            run_session.step("x", out, [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('1')"], force=True)
            self.assertTrue(marker.exists())

    def test_a_failing_step_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception):
                run_session.step("x", Path(tmp) / "never.txt", [sys.executable, "-c", "raise SystemExit(3)"], force=False)


class ShortPathTest(unittest.TestCase):
    def test_paths_are_printed_repo_relative(self):
        inside = str(run_session.REPO_ROOT / "outputs" / "x" / "report.json")
        self.assertEqual(run_session._short(inside), "outputs/x/report.json")
        self.assertEqual(run_session._short(run_session.PY), "python")
        self.assertEqual(run_session._short("--force"), "--force")


if __name__ == "__main__":
    unittest.main()
