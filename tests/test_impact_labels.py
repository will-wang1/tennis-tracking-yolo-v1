import tempfile
import unittest
from pathlib import Path

from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.evaluation.impact_labels import Label, match_labels, read_labels, score_impacts

FPS = 100.0  # one frame per centisecond, so seconds read straight off `t`


def _marker(seconds: float, kind: str = "bounce") -> BounceCandidate:
    return BounceCandidate(
        frame_idx=int(round(seconds * FPS)),
        t=seconds * FPS,
        x=500.0,
        y=800.0,
        restitution=0.7,
        horizontal_ratio=0.9,
        speed_ratio=0.8,
        rmse=1.0,
        is_bounce=kind == "bounce",
        kind=kind,
    )


def _label(seconds: float, kind: str, tolerance_s: float = 0.25) -> Label:
    return Label(seconds=seconds, kind=kind, tolerance_s=tolerance_s)


class MatchLabelsTest(unittest.TestCase):
    def test_matches_a_marker_inside_the_tolerance(self):
        matched = match_labels([_label(1.0, "bounce")], [_marker(1.1)], FPS)

        self.assertIn(0, matched)

    def test_leaves_a_marker_outside_the_tolerance_unmatched(self):
        matched = match_labels([_label(1.0, "bounce")], [_marker(1.4)], FPS)

        self.assertEqual(matched, {})

    def test_honors_each_labels_own_tolerance(self):
        labels = [_label(1.0, "bounce", tolerance_s=0.5), _label(5.0, "bounce", tolerance_s=0.1)]
        matched = match_labels(labels, [_marker(1.4), _marker(5.4)], FPS)

        self.assertIn(0, matched)
        self.assertNotIn(1, matched)

    def test_takes_the_nearest_marker_not_the_first(self):
        matched = match_labels([_label(1.0, "bounce")], [_marker(1.2), _marker(1.01)], FPS)

        self.assertAlmostEqual(matched[0].t / FPS, 1.01)

    def test_one_marker_cannot_answer_two_labels(self):
        # video_input2's 0.34s non-event beside its 0.50s bounce: the single
        # correct marker used to be scored as a hit AND a false positive
        labels = [_label(0.34, "none"), _label(0.50, "bounce")]
        matched = match_labels(labels, [_marker(0.50)], FPS)

        self.assertNotIn(0, matched)
        self.assertAlmostEqual(matched[1].t / FPS, 0.50)

    def test_gives_each_label_the_marker_that_is_closest_to_it(self):
        labels = [_label(1.0, "bounce"), _label(1.2, "bounce")]
        matched = match_labels(labels, [_marker(1.02), _marker(1.18)], FPS)

        self.assertAlmostEqual(matched[0].t / FPS, 1.02)
        self.assertAlmostEqual(matched[1].t / FPS, 1.18)


class ScoreImpactsTest(unittest.TestCase):
    def test_counts_a_matching_kind_as_correct(self):
        score = score_impacts([_marker(1.0, "bounce")], [_label(1.0, "bounce")], FPS)

        self.assertEqual(score.correct, 1)
        self.assertEqual(score.scored[0].outcome, "ok")

    def test_counts_a_mismatched_kind_as_wrong(self):
        score = score_impacts([_marker(1.0, "contact")], [_label(1.0, "bounce")], FPS)

        self.assertEqual(score.wrong_kind, 1)

    def test_counts_a_label_with_no_marker_as_missed(self):
        score = score_impacts([], [_label(1.0, "bounce")], FPS)

        self.assertEqual(score.missed, 1)

    def test_counts_a_marker_on_a_confirmed_non_event_as_a_false_positive(self):
        score = score_impacts([_marker(1.0, "bounce")], [_label(1.0, "none")], FPS)

        self.assertEqual(score.false_positives, 1)

    def test_an_undrawn_non_event_is_correct(self):
        score = score_impacts([], [_label(1.0, "none")], FPS)

        self.assertEqual(score.correct, 1)

    def test_unknown_verdicts_are_never_scored(self):
        # "unknown" draws no marker, so it makes no claim to be right or
        # wrong about - it can neither answer a label nor violate a "none"
        labels = [_label(1.0, "bounce"), _label(5.0, "none")]
        score = score_impacts([_marker(1.0, "unknown"), _marker(5.0, "unknown")], labels, FPS)

        self.assertEqual(score.missed, 1)
        self.assertEqual(score.false_positives, 0)
        self.assertEqual(score.unclaimed, [])

    def test_reports_markers_that_no_label_covers(self):
        score = score_impacts([_marker(1.0), _marker(9.0)], [_label(1.0, "bounce")], FPS)

        self.assertEqual([m.t / FPS for m in score.unclaimed], [9.0])

    def test_every_label_gets_exactly_one_outcome(self):
        labels = [_label(1.0, "bounce"), _label(5.0, "contact"), _label(9.0, "none")]
        score = score_impacts([_marker(1.0), _marker(5.0, "bounce")], labels, FPS)

        self.assertEqual(len(score.scored), len(labels))
        self.assertEqual(score.correct + score.wrong_kind + score.missed + score.false_positives, 3)


class ReadLabelsTest(unittest.TestCase):
    def _write(self, text: str) -> Path:
        directory = tempfile.mkdtemp()
        path = Path(directory) / "labels.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_a_label_file(self):
        path = self._write(
            "seconds,kind,tolerance_s,note\n1.5,bounce,0.3,off a dropshot\n0.5,none,,\n"
        )
        labels = read_labels(path)

        self.assertEqual([label.seconds for label in labels], [0.5, 1.5])  # sorted by time
        self.assertEqual(labels[1].note, "off a dropshot")
        self.assertEqual(labels[1].tolerance_s, 0.3)

    def test_a_blank_tolerance_falls_back_to_the_default(self):
        labels = read_labels(self._write("seconds,kind,tolerance_s,note\n1.5,bounce,,\n"))

        self.assertEqual(labels[0].tolerance_s, 0.25)

    def test_refuses_a_kind_it_does_not_understand(self):
        path = self._write("seconds,kind,tolerance_s,note\n1.5,maybe,,\n")

        with self.assertRaises(ValueError):
            read_labels(path)


class CheckedInLabelFilesTest(unittest.TestCase):
    """The label files are data, not code, so nothing else would notice a
    typo in one until a score quietly moved."""

    def _files(self):
        return sorted((Path(__file__).resolve().parent.parent / "data" / "labels").glob("*.csv"))

    def test_there_are_label_files(self):
        self.assertTrue(self._files())

    def test_every_label_file_parses(self):
        for path in self._files():
            with self.subTest(path.name):
                labels = read_labels(path)
                self.assertTrue(labels)
                for label in labels:
                    self.assertGreaterEqual(label.seconds, 0.0)
                    self.assertGreater(label.tolerance_s, 0.0)

    def test_no_event_is_labelled_twice(self):
        # windows are allowed to overlap - 0.34s and 0.50s on video_input2
        # do, which is what match_labels exists for - but two rows at the
        # same instant would be two verdicts about one event
        for path in self._files():
            with self.subTest(path.name):
                times = [label.seconds for label in read_labels(path)]
                self.assertEqual(len(times), len(set(times)))


if __name__ == "__main__":
    unittest.main()
