"""Regression tests for `resolve_device`.

A written-but-empty env var (`PIPELINE_DEVICE=` in a .env) reads as "", not
as unset. That empty string used to pass straight through to
`torch.load(map_location="")`, which fails with the very unhelpful "don't
know how to restore data location of torch.storage.UntypedStorage (tagged
with )" from deep inside torch, nowhere near the actual cause.
"""

import unittest
from unittest.mock import patch

from src.detection._tracknet_arch import resolve_device


class ResolveDeviceTest(unittest.TestCase):
    def test_blank_is_treated_as_unset(self):
        with patch("torch.cuda.is_available", return_value=False):
            for blank in ("", "   ", "\t"):
                with self.subTest(blank=blank):
                    self.assertEqual(resolve_device(blank), "cpu")

    def test_none_is_unset(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device(None), "cpu")

    def test_unset_prefers_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device(None), "cuda")
            self.assertEqual(resolve_device(""), "cuda")

    def test_explicit_device_is_passed_through(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(resolve_device("cuda:1"), "cuda:1")

    def test_bare_digit_is_a_cuda_index(self):
        self.assertEqual(resolve_device("0"), "cuda:0")
        self.assertEqual(resolve_device(" 1 "), "cuda:1")


if __name__ == "__main__":
    unittest.main()
