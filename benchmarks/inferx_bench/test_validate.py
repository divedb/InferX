"""Regression tests for benchmark integrity; run without a GPU."""
import copy
import hashlib
import unittest
from validate import validate_metadata, validate_trials


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(id='test', concurrency=1, output_len=2)]
        self.rows = [dict(case='test', repeat=0, e2e_ms=3.0,
                         outputs=[[10, 20]], arrivals_ms=[[1.0, 2.0]],
                         peak_device_mib=100.0, peak_device_delta_mib=50.0)]
        self.meta = dict(workload_sha256=hashlib.sha256(b'workload').hexdigest(),
                         weights={'model': 'hash'}, config_sha256='config',
                         gpu='gpu', driver='driver', packages={'vllm': 'pinned'},
                         platform='platform', repeats=1)

    def test_valid_trial(self):
        validate_trials(self.rows, self.cases, 1)
        validate_metadata(self.meta, self.meta, b'workload')

    def test_missing_or_duplicate_repeat(self):
        for rows in ([], self.rows * 2):
            with self.assertRaises(ValueError):
                validate_trials(rows, self.cases, 1)

    def test_truncated_outputs_fail(self):
        self.rows[0]['outputs'][0].pop()
        with self.assertRaises(ValueError):
            validate_trials(self.rows, self.cases, 1)

    def test_invalid_times_fail(self):
        for times in ([2.0, 1.0], [1.0, 4.0], [1.0, float('nan')], [0.0, 1.0]):
            self.rows[0]['arrivals_ms'] = [times]
            with self.assertRaises(ValueError):
                validate_trials(self.rows, self.cases, 1)

    def test_different_workload_fails(self):
        with self.assertRaises(ValueError):
            validate_metadata(self.meta, self.meta, b'different workload')

    def test_different_weights_or_environment_fail(self):
        for field in ('weights', 'config_sha256', 'gpu', 'driver', 'packages', 'platform', 'repeats'):
            changed = copy.deepcopy(self.meta)
            changed[field] = 'different'
            with self.assertRaises(ValueError):
                validate_metadata(changed, self.meta, b'workload')


if __name__ == '__main__':
    unittest.main()
