import unittest

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.analysis.rally_clusters import filter_excess_bounces_between_contacts, filter_non_rally_clusters


def _identity_calibration():
    # pixel_to_world becomes the identity - lets a test pick world_y/x
    # directly via an impact's own x/y (see test_match_stats.py's copy).
    return CourtCalibration(homography=np.eye(3, dtype=np.float64))


def _impact(t, kind, x, y, frame_idx=None, rmse=0.0):
    return BounceCandidate(
        frame_idx=frame_idx if frame_idx is not None else int(round(t * 30)),
        t=t * 30.0,
        x=x,
        y=y,
        restitution=0.0,
        horizontal_ratio=0.0,
        speed_ratio=0.0,
        rmse=rmse,
        kind=kind,
    )


class FilterNonRallyClustersTest(unittest.TestCase):
    def setUp(self):
        self.calibration = _identity_calibration()

    def _calibrations(self, impacts):
        return {impact.frame_idx: self.calibration for impact in impacts}

    def test_the_real_alcaraz_djokovic_cluster_is_reclassified(self):
        # Reproduces the actual world positions this module was built from
        # (see its docstring) - six same-spot "near" events (three real
        # ground bounces interleaved with three contacts the classifier
        # read as returns), then a genuine serve ~8m away that must
        # survive untouched.
        impacts = [
            _impact(29.35, "bounce", x=5.0, y=21.0),
            _impact(30.32, "bounce", x=4.98, y=21.5),
            _impact(31.96, "bounce", x=5.02, y=22.0),
            _impact(32.86, "contact", x=5.03, y=24.53),
            _impact(33.27, "contact", x=4.96, y=24.38),
            _impact(33.74, "contact", x=4.94, y=24.50),
            _impact(36.38, "contact", x=4.94, y=23.35),
            _impact(36.93, "contact", x=4.76, y=20.48),
            _impact(37.43, "contact", x=4.90, y=23.98),
            _impact(38.82, "contact", x=4.75, y=16.01),  # the real serve
        ]
        calibrations = self._calibrations(impacts)

        result = filter_non_rally_clusters(impacts, calibrations)

        # everything before the real serve was part of the cluster
        for impact in result[:-1]:
            self.assertEqual(impact.kind, "unknown")
            self.assertFalse(impact.is_bounce)
        # the real serve, ~8m from the cluster, must survive as a contact
        self.assertEqual(result[-1].kind, "contact")

    def test_fewer_than_the_minimum_cluster_size_is_left_alone(self):
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.5, "contact", x=5.0, y=20.1),
        ]
        result = filter_non_rally_clusters(impacts, self._calibrations(impacts), min_cluster_size=3)

        self.assertEqual([i.kind for i in result], ["contact", "contact"])

    def test_a_real_rally_with_shots_far_apart_is_untouched(self):
        # alternating near/far positions, each pair many metres apart -
        # nothing here should ever cluster
        impacts = [
            _impact(0.0, "contact", x=5.0, y=22.0),
            _impact(1.0, "bounce", x=5.0, y=15.0),
            _impact(2.0, "contact", x=5.0, y=2.0),
            _impact(3.0, "bounce", x=5.0, y=15.0),
            _impact(4.0, "contact", x=5.0, y=22.0),
        ]
        result = filter_non_rally_clusters(impacts, self._calibrations(impacts))

        self.assertEqual([i.kind for i in result], ["contact", "bounce", "contact", "bounce", "contact"])

    def test_unknown_impacts_are_never_touched(self):
        impacts = [
            _impact(0.0, "unknown", x=5.0, y=20.0),
            _impact(0.5, "unknown", x=5.0, y=20.1),
            _impact(1.0, "unknown", x=5.0, y=20.2),
        ]
        result = filter_non_rally_clusters(impacts, self._calibrations(impacts))

        self.assertEqual([i.kind for i in result], ["unknown", "unknown", "unknown"])

    def test_impacts_with_no_calibration_are_left_exactly_as_is(self):
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.5, "contact", x=5.0, y=20.1),
            _impact(1.0, "contact", x=5.0, y=20.2),
        ]
        result = filter_non_rally_clusters(impacts, calibrations_by_frame={})

        self.assertEqual([i.kind for i in result], ["contact", "contact", "contact"])

    def test_a_gap_impact_with_no_position_does_not_break_a_real_run(self):
        # an "unknown" impact with no calibration sits in the middle of an
        # otherwise-clustered run - it shouldn't reset the cluster, since
        # it carries no positional evidence the pattern actually broke
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.3, "unknown", x=999.0, y=999.0, frame_idx=9),  # no calibration below
            _impact(0.6, "contact", x=5.0, y=20.1),
            _impact(0.9, "contact", x=5.0, y=20.2),
        ]
        calibrations = {i.frame_idx: self.calibration for i in impacts if i.frame_idx != 9}

        result = filter_non_rally_clusters(impacts, calibrations)

        contact_kinds = [i.kind for i in result if i.t != impacts[1].t]
        self.assertEqual(contact_kinds, ["unknown", "unknown", "unknown"])

    def test_a_real_shot_that_travels_far_from_the_cluster_survives(self):
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.3, "contact", x=5.0, y=20.1),
            _impact(0.6, "contact", x=5.0, y=20.2),
            _impact(1.0, "contact", x=5.0, y=2.0),  # far away - the real serve
        ]
        result = filter_non_rally_clusters(impacts, self._calibrations(impacts))

        self.assertEqual(result[-1].kind, "contact")
        self.assertEqual(result[-1].y, 2.0)

    def test_custom_thresholds_are_respected(self):
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.3, "contact", x=5.0, y=20.1),
        ]
        result = filter_non_rally_clusters(impacts, self._calibrations(impacts), min_cluster_size=2)

        self.assertEqual([i.kind for i in result], ["unknown", "unknown"])

    def test_no_impacts_at_all_returns_empty(self):
        self.assertEqual(filter_non_rally_clusters([], {}), [])


class FilterExcessBouncesBetweenContactsTest(unittest.TestCase):
    def test_reproduces_the_real_4_57s_case(self):
        # the real Alcaraz-Djokovic opening rally: two contacts with THREE
        # bounces between them - the serve's real landing (best fit) plus
        # two spurious ones, one of them the 4.57s "random blob over the
        # net" the user flagged by eye
        impacts = [
            _impact(2.03, "contact", x=5.0, y=20.0),
            _impact(2.44, "bounce", x=5.0, y=15.0, rmse=3.1),
            _impact(4.57, "bounce", x=6.0, y=12.0, rmse=2.4),  # the spurious blob
            _impact(5.03, "bounce", x=5.0, y=10.0, rmse=0.8),  # the real landing - best fit
            _impact(5.68, "contact", x=5.0, y=2.0),
        ]

        result = filter_excess_bounces_between_contacts(impacts)

        kinds_by_t = {round(i.t / 30.0, 2): i.kind for i in result}
        self.assertEqual(kinds_by_t[2.44], "unknown")
        self.assertEqual(kinds_by_t[4.57], "unknown")
        self.assertEqual(kinds_by_t[5.03], "bounce")  # the best-fitting one survives

    def test_a_single_bounce_between_contacts_is_untouched(self):
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.5, "bounce", x=5.0, y=10.0, rmse=1.0),
            _impact(1.0, "contact", x=5.0, y=2.0),
        ]
        result = filter_excess_bounces_between_contacts(impacts)
        self.assertEqual([i.kind for i in result], ["contact", "bounce", "contact"])

    def test_ties_keep_the_first_by_stable_min(self):
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.5, "bounce", x=5.0, y=10.0, rmse=1.0),
            _impact(0.7, "bounce", x=5.0, y=10.0, rmse=1.0),
            _impact(1.0, "contact", x=5.0, y=2.0),
        ]
        result = filter_excess_bounces_between_contacts(impacts)
        kinds = [i.kind for i in result]
        self.assertEqual(kinds.count("bounce"), 1)

    def test_the_gap_before_the_first_contact_is_also_checked(self):
        impacts = [
            _impact(0.0, "bounce", x=5.0, y=20.0, rmse=2.0),
            _impact(0.3, "bounce", x=5.0, y=20.0, rmse=0.5),
            _impact(1.0, "contact", x=5.0, y=2.0),
        ]
        result = filter_excess_bounces_between_contacts(impacts)
        bounce_rmses = [i.rmse for i in result if i.kind == "bounce"]
        self.assertEqual(bounce_rmses, [0.5])

    def test_the_gap_after_the_last_contact_is_also_checked(self):
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.5, "bounce", x=5.0, y=10.0, rmse=2.0),
            _impact(0.8, "bounce", x=5.0, y=10.0, rmse=0.3),
        ]
        result = filter_excess_bounces_between_contacts(impacts)
        bounce_rmses = [i.rmse for i in result if i.kind == "bounce"]
        self.assertEqual(bounce_rmses, [0.3])

    def test_unknown_impacts_are_never_reclassified_or_counted(self):
        impacts = [
            _impact(0.0, "contact", x=5.0, y=20.0),
            _impact(0.3, "unknown", x=5.0, y=15.0),
            _impact(0.5, "bounce", x=5.0, y=10.0, rmse=1.0),
            _impact(1.0, "contact", x=5.0, y=2.0),
        ]
        result = filter_excess_bounces_between_contacts(impacts)
        self.assertEqual([i.kind for i in result], ["contact", "unknown", "bounce", "contact"])

    def test_no_impacts_returns_empty(self):
        self.assertEqual(filter_excess_bounces_between_contacts([]), [])

    def test_no_contacts_at_all_still_checks_the_one_gap(self):
        impacts = [
            _impact(0.0, "bounce", x=5.0, y=20.0, rmse=2.0),
            _impact(0.5, "bounce", x=5.0, y=10.0, rmse=0.4),
        ]
        result = filter_excess_bounces_between_contacts(impacts)
        bounce_rmses = [i.rmse for i in result if i.kind == "bounce"]
        self.assertEqual(bounce_rmses, [0.4])


if __name__ == "__main__":
    unittest.main()
