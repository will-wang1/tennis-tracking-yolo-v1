import unittest

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.analysis.serve_sequences import classify_serve_sequences

FPS = 30.0


def _identity_calibration():
    # pixel_to_world becomes the identity - lets a test pick world_y
    # directly via an impact's own y (see test_match_stats.py's own copy
    # of this fixture).
    return CourtCalibration(homography=np.eye(3, dtype=np.float64))


def _impact(t, kind, y, x=5.0):
    return BounceCandidate(
        frame_idx=int(round(t * FPS)),
        t=t * FPS,  # BounceCandidate.t is in FRAMES elsewhere in this project - see below
        x=x,
        y=y,
        restitution=0.0,
        horizontal_ratio=0.0,
        speed_ratio=0.0,
        rmse=0.0,
        kind=kind,
    )


NEAR_Y = 22.0  # world_y on the near half (> net at ~11.885)
FAR_Y = 2.0    # world_y on the far half


class ClassifyServeSequencesTest(unittest.TestCase):
    def setUp(self):
        self.calibration = _identity_calibration()

    def _calibrations(self, impacts):
        return {impact.frame_idx: self.calibration for impact in impacts}

    def test_no_contacts_produces_no_serve_attempts(self):
        impacts = [_impact(0, "bounce", NEAR_Y)]
        result = classify_serve_sequences(impacts, {}, {}, fps=FPS)
        self.assertEqual(result, [])

    def test_the_last_serve_candidate_with_nothing_after_it_is_unknown(self):
        impacts = [_impact(0, "contact", NEAR_Y)]
        identities = {impacts[0].frame_idx: {"near": "player_a"}}
        result = classify_serve_sequences(impacts, self._calibrations(impacts), identities, fps=FPS)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].outcome, "unknown")
        self.assertEqual(result[0].identity, "player_a")
        self.assertEqual(result[0].side, "near")

    def test_a_rally_developing_before_the_next_serve_is_rally_developed(self):
        # serve, bounce, return contact, bounce, contact... then a long gap
        # and a second point's serve
        impacts = [
            _impact(0.0, "contact", NEAR_Y),
            _impact(0.5, "bounce", FAR_Y),
            _impact(1.0, "contact", FAR_Y),
            _impact(1.5, "bounce", NEAR_Y),
            _impact(2.0, "contact", NEAR_Y),
            _impact(20.0, "contact", NEAR_Y),  # next point's serve, well clear of any gap threshold
        ]
        identities = {
            impacts[0].frame_idx: {"near": "player_a", "far": "player_b"},
            impacts[5].frame_idx: {"near": "player_a", "far": "player_b"},
        }
        result = classify_serve_sequences(impacts, self._calibrations(impacts), identities, fps=FPS)

        serve_outcomes = [a.outcome for a in result]
        self.assertEqual(serve_outcomes[0], "rally_developed")
        self.assertGreaterEqual(result[0].shots_before_next_serve, 1)

    def test_same_server_serving_again_soon_with_nothing_between_is_a_fault(self):
        first_serve = _impact(0.0, "contact", NEAR_Y)
        second_serve = _impact(5.0, "contact", NEAR_Y)  # within the fault-gap window, same side
        impacts = [first_serve, second_serve]
        identities = {
            first_serve.frame_idx: {"near": "player_a", "far": "player_b"},
            second_serve.frame_idx: {"near": "player_a", "far": "player_b"},
        }
        result = classify_serve_sequences(impacts, self._calibrations(impacts), identities, fps=FPS)

        self.assertEqual(result[0].outcome, "fault")

    def test_a_different_server_serving_soon_after_is_not_a_fault(self):
        # near player serves, then (implausibly soon in real tennis, but
        # structurally what matters here) the FAR player's contact follows -
        # a different server, so this can't be read as the same player's
        # fault/retry
        first_serve = _impact(0.0, "contact", NEAR_Y)
        other_serve = _impact(5.0, "contact", FAR_Y)
        impacts = [first_serve, other_serve]
        identities = {
            first_serve.frame_idx: {"near": "player_a", "far": "player_b"},
            other_serve.frame_idx: {"near": "player_a", "far": "player_b"},
        }
        result = classify_serve_sequences(impacts, self._calibrations(impacts), identities, fps=FPS)

        self.assertEqual(result[0].outcome, "unreturned")

    def test_same_server_but_gap_too_long_is_not_a_fault(self):
        first_serve = _impact(0.0, "contact", NEAR_Y)
        later_point = _impact(40.0, "contact", NEAR_Y)  # well past the fault-gap window
        impacts = [first_serve, later_point]
        identities = {
            first_serve.frame_idx: {"near": "player_a", "far": "player_b"},
            later_point.frame_idx: {"near": "player_a", "far": "player_b"},
        }
        result = classify_serve_sequences(
            impacts, self._calibrations(impacts), identities, fps=FPS, fault_gap_seconds=8.0
        )

        self.assertEqual(result[0].outcome, "unreturned")

    def test_unknown_identity_never_claims_a_fault(self):
        # no identity information at all - structure alone (short gap, same
        # SIDE) must not be enough on its own to call a fault
        first_serve = _impact(0.0, "contact", NEAR_Y)
        second_serve = _impact(5.0, "contact", NEAR_Y)
        impacts = [first_serve, second_serve]

        result = classify_serve_sequences(impacts, self._calibrations(impacts), {}, fps=FPS)

        self.assertEqual(result[0].outcome, "unreturned")
        self.assertIsNone(result[0].identity)

    def test_bounces_and_unknowns_still_count_toward_the_serve_gap(self):
        # a bounce right before a "contact" means the rally was still
        # live - the contact after it should NOT be treated as a fresh
        # serve candidate just because it is itself a contact
        impacts = [
            _impact(0.0, "contact", NEAR_Y),
            _impact(2.0, "bounce", FAR_Y),   # keeps the gap under the serve threshold from here
            _impact(2.4, "contact", FAR_Y),  # a return, not a new serve
        ]
        identities = {impacts[0].frame_idx: {"near": "player_a", "far": "player_b"}}
        result = classify_serve_sequences(impacts, self._calibrations(impacts), identities, fps=FPS)

        # only the very first contact should be treated as a serve candidate
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].frame_idx, impacts[0].frame_idx)


if __name__ == "__main__":
    unittest.main()
