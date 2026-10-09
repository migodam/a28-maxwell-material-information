import json
from pathlib import Path
import tempfile
import unittest
import torch
from a28.physics import CountedMaxwell
from a28.pipeline import _frozen


class ProtocolGuardTests(unittest.TestCase):
    def test_multi_state_batch_rejected_before_any_reservation(self):
        # Deliberately uninitialized object: a valid rejection cannot access
        # a budget or Maxwell buffer before rejecting the batched input.
        adapter = object.__new__(CountedMaxwell)
        with self.assertRaisesRegex(ValueError, 'SINGLE_MATERIAL_STATE'):
            adapter(torch.zeros((2, 12), dtype=torch.float64))

    def test_frozen_json_values_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results/a28/FROZEN_PROTOCOL.json'
            path.parent.mkdir(parents=True)
            original = {'frozen': True, 'rank': 16, 'cohort': [1, 2]}
            path.write_text(json.dumps(original))
            self.assertEqual(_frozen(directory, dict(original)), original)
            with self.assertRaisesRegex(ValueError, 'CONFIG_MISMATCH'):
                _frozen(directory, {**original, 'rank': 24})
