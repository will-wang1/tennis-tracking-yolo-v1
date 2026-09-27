import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "make_swing_labeller", Path(__file__).resolve().parent.parent / "scripts" / "make_swing_labeller.py"
)
labeller = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(labeller)


class GuessTest(unittest.TestCase):
    def test_model_calls_become_prefilled_labels(self):
        self.assertEqual(labeller.guess({"stroke": "serve", "label": "SF"}), {"stroke": "serve"})
        self.assertEqual(labeller.guess({"stroke": "backhand", "label": "HNL"}), {"stroke": "backhand"})
        self.assertEqual(labeller.guess({"stroke": "net", "label": "HFL"}), {"stroke": "forehand", "volley": True})
        self.assertEqual(labeller.guess({"stroke": "net", "label": "HFR"}), {"stroke": "backhand", "volley": True})
        self.assertEqual(labeller.guess({"stroke": "unsure", "label": "HNR"}), {})
        self.assertEqual(labeller.guess(None), {})


if __name__ == "__main__":
    unittest.main()
