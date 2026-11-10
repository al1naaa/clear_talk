import unittest

import numpy as np

from backend.metrics import matched_si_sdr_improvement, si_sdr


class MetricTests(unittest.TestCase):
    def test_scale_invariance(self):
        rng = np.random.default_rng(22)
        reference = rng.normal(size=4000)
        estimate = reference + 0.2 * rng.normal(size=4000)
        self.assertAlmostEqual(si_sdr(estimate, reference), si_sdr(3 * estimate, reference), places=6)

    def test_permutation_and_improvement(self):
        rng = np.random.default_rng(12)
        first, second, noise = rng.normal(size=(3, 4000))
        mixture = first + second + 0.2 * noise
        output = matched_si_sdr_improvement([second + 0.01 * noise, first + 0.01 * noise], [first, second], mixture)
        self.assertEqual(output["reference_order"], [1, 0])
        self.assertGreater(output["mean_improvement_db"], 30)
        baseline = matched_si_sdr_improvement([mixture, mixture], [first, second], mixture)
        self.assertAlmostEqual(baseline["mean_improvement_db"], 0, places=8)

    def test_silent_reference_is_not_a_score(self):
        with self.assertRaises(ValueError):
            si_sdr(np.ones(100), np.zeros(100))


if __name__ == "__main__":
    unittest.main()
