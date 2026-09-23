"""CPU-only checks that the numerical gate rejects invalid/inequivalent output."""
import unittest

import numpy as np

from numerical import compare_logits


class NumericalGateTest(unittest.TestCase):
    def test_absolute_and_relative_tolerances_use_reference(self):
        result = compare_logits([[0., 10., -10.]], [[.01, 10.1, -10.1]], .02, .01)[0]
        self.assertTrue(result["numerical_pass"])
        result = compare_logits([[0., 10., -10.]], [[.03, 10.2, -10.2]], .02, .01)[0]
        self.assertEqual(result["out_of_tolerance"], 3)

    def test_large_candidate_does_not_expand_relative_tolerance(self):
        self.assertFalse(compare_logits([[0., 1.]], [[0., 2.]], 0, .75)[0]["numerical_pass"])

    def test_numerical_pass_does_not_hide_changed_argmax(self):
        result = compare_logits([[1., 1.01, 0.]], [[1.01, 1., 0.]], .02, 0)[0]
        self.assertTrue(result["numerical_pass"])
        self.assertFalse(result["exact_argmax"])
        self.assertAlmostEqual(result["reference_margin"], .01)
        self.assertAlmostEqual(result["reference_gap_at_candidate"], .01)

    def test_ties_choose_lowest_token_id(self):
        result = compare_logits([[1., 1., 0.]], [[1., 1., 0.]], 0, 0)[0]
        self.assertEqual(result["candidate_argmax"], 0)
        self.assertEqual(result["reference_margin"], 0)
        self.assertTrue(result["exact_argmax"])

    def test_rejects_nonfinite_values_even_if_equal(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_logits([[0., value]], [[0., value]], 0, 0)

    def test_rejects_empty_truncated_or_broadcastable_shapes(self):
        for reference, candidate in (([], []), ([[1., 2.]], [[1.]]),
                                     ([[1., 2.], [3., 4.]], [[1., 2.]])):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare_logits(reference, candidate, 0, 0)

    def test_rejects_invalid_tolerances(self):
        for value in (-1., np.nan, np.inf):
            for atol, rtol in ((value, 0), (0, value)):
                with self.subTest(atol=atol, rtol=rtol), self.assertRaises(ValueError):
                    compare_logits([[0., 1.]], [[0., 1.]], atol, rtol)


if __name__ == "__main__":
    unittest.main()
