import unittest

from src.session.outcomes import per_player_errors, rally_outcomes

FPS = 25.0


def swing(frame, player, end, contact=None):
    s = {"frame": frame, "player": player, "end": end, "confirmed_by": []}
    if contact is not None:
        s.update(contact_frame=contact, confirmed_by=["ball"])
    return s


def bounce(frame, half, bounds="singles", depth="deep", pos=(5.0, 5.0)):
    return {"frame": frame, "half": half, "bounds": bounds, "depth": depth, "position_m": list(pos)}


class OutcomesTest(unittest.TestCase):
    def test_out_net_and_not_returned(self):
        swings = [swing(100, 5, "near", 100), swing(140, 6, "far", 140),  # rally 1: 6 hits it out
                  swing(300, 5, "near", 300),                              # rally 2: 5 hits into the net
                  swing(500, 7, "near", 500)]                              # rally 3: lands in, no reply
        rallies = [{"start_s": 4.0, "swing_frames": [100, 140]}, {"start_s": 12.0, "swing_frames": [300]},
                   {"start_s": 20.0, "swing_frames": [500]}]
        contacts = [{"frame": 300, "stood_m": [5.0, 22.0]}]
        bounces = [bounce(120, "far"), bounce(160, "near", bounds="out", pos=(12.5, 20.0)),
                   bounce(315, "near", pos=(5.0, 17.0)), bounce(520, "far")]
        out = rally_outcomes(rallies, swings, contacts, bounces, FPS)
        self.assertEqual([(o["outcome"], o["last_hitter"], o["ball_number"]) for o in out],
                         [("out", 6, 2), ("net", 5, 1), ("not returned", 7, 1)])
        errs = per_player_errors(out, swings, coach=None)
        self.assertEqual((errs["6"]["errors"], errs["6"]["errors_out"], errs["6"]["errors_first_two_balls"]), (1, 1, 1))
        self.assertEqual((errs["5"]["errors_net"], errs["7"]["not_returned"]), (1, 1))

    def test_nothing_seen_is_unclear_and_the_coach_is_left_out(self):
        swings = [swing(100, 9, "far"), swing(200, 5, "near")]
        out = rally_outcomes([{"start_s": 4.0, "swing_frames": [100]}], swings, [], [], FPS)
        self.assertEqual(out[0]["outcome"], "unclear")
        self.assertNotIn("9", per_player_errors(out, swings, coach=9))


if __name__ == "__main__":
    unittest.main()
