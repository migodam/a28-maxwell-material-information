"""Synthetic unit tests only; these are not Maxwell experimental evidence."""
from copy import deepcopy
import inspect
import json
import unittest

import numpy as np

from a28.edits import (EDIT_ORDER, RULE_VERSION, calibrate_train_rules,
                       make_structural_edits)


class A28EditTests(unittest.TestCase):
    def setUp(self):
        self.grid = (12, 12)
        self.phase = (1. + .2j) / abs(1. + .2j)
        self.bg = np.full(self.grid, .1 + .04j, dtype=np.complex128)
        base = np.zeros(self.grid)
        base[3:9, 3:9] = 1.2
        self.center = self.bg + self.phase * base
        bank = np.stack([self.center, self.center, self.bg + self.phase * base * .25])
        self.rules, self.audit = calibrate_train_rules(
            bank, self.bg, split="train", parent_ids=np.array([250001, 250002, 250001]),
            amplitudes=np.array([1.2, 1.2, .3]), grid=self.grid)

    def edits(self, center=None, rules=None, **kwargs):
        return make_structural_edits(self.center if center is None else center,
                                     self.bg, self.rules if rules is None else rules,
                                     center_provenance="SYNTHETIC_UNIT_TEST", **kwargs)

    def test_train_freeze_parent_balanced_and_json_serializable(self):
        self.assertAlmostEqual(self.rules["material_scale"], 1.2)
        self.assertAlmostEqual(self.rules["support_threshold"], .3)
        self.assertEqual(self.audit["row_count_used"], 2)
        self.assertEqual(self.audit["independent_parent_count"], 2)
        self.assertEqual(self.audit["physics_calls"], 0)
        self.assertLess(self.audit["fixed_phase_relative_error"], 1e-14)
        json.dumps(self.rules, allow_nan=False)
        json.dumps(self.audit, allow_nan=False)

    def test_calibration_denies_development_locked_and_wrong_parent(self):
        for role in ("validation", "locked_id", "locked_ood"):
            with self.assertRaises(ValueError):
                calibrate_train_rules(self.center[None], self.bg, split=role,
                                      parent_ids=np.array([250001]), grid=self.grid)
        with self.assertRaises(ValueError):
            calibrate_train_rules(self.center[None], self.bg, split="train",
                                  parent_ids=np.array([250065]), grid=self.grid)
        with self.assertRaises(ValueError):
            calibrate_train_rules(self.center[None], self.bg, split="train",
                                  parent_ids=np.array([250001]), grid=self.grid,
                                  amplitudes=np.array([.3]))

    def test_exact_same_geometry_rules_for_online_and_offline_centers(self):
        online = make_structural_edits(self.center, self.bg, self.rules,
                                      center_provenance="ONLINE_CNN_PREDICTION")
        offline = make_structural_edits(self.center, self.bg, self.rules,
                                       center_provenance="OFFLINE_TRUTH_DIAGNOSTIC")
        self.assertEqual(tuple(e.method_id for e in online), EDIT_ORDER)
        for a, b in zip(online, offline):
            np.testing.assert_array_equal(a.chi, b.chi)
            np.testing.assert_array_equal(a.change, b.change)
            self.assertTrue(b.metadata["diagnostic_only"])
            self.assertFalse(b.metadata["online_candidate_eligible"])
            json.dumps(a.summary(), allow_nan=False)
        self.assertTrue(online[0].metadata["online_candidate_eligible"])
        self.assertFalse(online[-1].metadata["online_candidate_eligible"])

    def test_deterministic_outputs_flat_native_order_and_no_mutation(self):
        before = self.center.copy()
        a, b = self.edits(), self.edits(self.center.ravel())
        for aa, bb in zip(a, b):
            self.assertEqual(aa.chi.shape, (144,))
            self.assertEqual(aa.chi.dtype, np.dtype("complex128"))
            np.testing.assert_array_equal(aa.chi, bb.chi)
            self.assertFalse(aa.metadata["result_driven_resampling"])
            self.assertEqual(aa.metadata["physics_calls"], 0)
        np.testing.assert_array_equal(self.center, before)

    def test_exact_delta_and_separate_compensation_do_not_erase_edit(self):
        for e in self.edits():
            np.testing.assert_allclose(e.change, e.chi - self.center.ravel(), rtol=0, atol=0)
            np.testing.assert_allclose(e.change, e.primary_change + e.compensation_change,
                                       rtol=0, atol=1e-15)
            self.assertFalse(np.any(e.primary_mask & e.compensation_mask))
            np.testing.assert_allclose(e.compensation_change[e.primary_mask], 0, rtol=0, atol=0)
            self.assertLessEqual(e.metadata["box_feasibility_defect"], 1e-10)
            self.assertTrue(e.valid)

    def test_structural_edits_preserve_declared_phase(self):
        for e in self.edits():
            delta = e.chi - self.bg.ravel()
            scalar = np.real(delta * self.phase.conjugate())
            np.testing.assert_allclose(delta, self.phase * scalar, atol=1e-14, rtol=1e-14)
        self.assertAlmostEqual(self.edits()[-1].metadata["contrast_multiplier"], .85)

    def test_cavity_open_and_existing_hole_fill_are_frozen_branches(self):
        open_edit = self.edits()[0]
        self.assertEqual(open_edit.metadata["operation"], "OPEN_INTERIOR_CAVITY")
        self.assertEqual(open_edit.metadata["primary_pixels"], 5)
        ring = self.center.copy()
        ring[5:7, 5:7] = self.bg[5:7, 5:7]
        fill_edit = self.edits(ring)[0]
        self.assertEqual(fill_edit.metadata["operation"], "FILL_LARGEST_EXISTING_HOLE")
        self.assertEqual(fill_edit.metadata["primary_pixels"], 4)
        self.assertGreater(np.real((fill_edit.chi - ring.ravel()) * self.phase.conjugate()).max(), 0)

    def test_fixed_boundary_dilate_and_optional_preregistered_erode(self):
        dilation = self.edits()[1]
        self.assertEqual(dilation.metadata["operation"], "DILATE_BOUNDARY")
        self.assertEqual(dilation.metadata["primary_pixels"], 24)
        rules = {**self.rules, "boundary_mode": "erode"}
        erosion = self.edits(rules=rules)[1]
        self.assertEqual(erosion.metadata["operation"], "ERODE_BOUNDARY")
        self.assertEqual(erosion.metadata["primary_pixels"], 20)

    def test_small_component_remove_or_add_without_family_metadata(self):
        extra = self.center.copy()
        extra[1, 1] = self.bg[1, 1] + self.phase
        removed = self.edits(extra)[2]
        self.assertEqual(removed.metadata["operation"], "REMOVE_SMALLEST_ELIGIBLE_COMPONENT")
        self.assertEqual(removed.metadata["removed_component_cells"], 1)
        self.assertEqual(removed.chi.reshape(self.grid)[1, 1], self.bg[1, 1])
        added = self.edits()[2]
        self.assertEqual(added.metadata["operation"], "ADD_SEPARATE_SMALL_COMPONENT")
        self.assertEqual(added.metadata["primary_pixels"], 5)
        names = inspect.signature(make_structural_edits).parameters
        self.assertNotIn("truth", names)
        self.assertNotIn("family", names)
        self.assertNotIn("error_direction", names)

    def test_empty_center_invalid_in_place_never_resampled(self):
        e = self.edits(self.bg)
        self.assertEqual(len(e), 4)
        for candidate in e:
            self.assertEqual(candidate.status, "INVALID_ZERO_EDIT")
            np.testing.assert_array_equal(candidate.chi, self.bg.ravel())
            np.testing.assert_array_equal(candidate.change, np.zeros(144, dtype=complex))
            self.assertIsNone(candidate.metadata["relative_center_delta_change"])

    def test_unavailable_interior_is_invalid_not_randomly_relocated(self):
        thin = self.bg.copy()
        thin[6, 2:10] += self.phase * 1.2
        cavity = self.edits(thin)[0]
        self.assertEqual(cavity.status, "INVALID_ZERO_EDIT")
        self.assertEqual(cavity.metadata["invalid_reason"], "NO_FULL_INTERIOR_CAVITY_SITE")

    def test_compensation_cap_and_box_limit_explicit(self):
        rules = {**self.rules, "compensation_cap_fraction": 1e-4}
        e = self.edits(rules=rules)[0]
        self.assertTrue(e.metadata["compensation_box_or_cap_limited"])
        self.assertLessEqual(abs(e.metadata["compensation_scalar_per_cell"]), 1.2e-4)
        # Legal arbitrary-complex CNN center can touch a box wall.  Uniform
        # compensation then respects that wall rather than clipping secretly.
        wall = self.center.copy()
        wall[3, 3] = 4. + .5j
        e = self.edits(wall)[0]
        self.assertLessEqual(e.chi.real.max(), 4. + 1e-10)
        self.assertLessEqual(e.chi.imag.max(), .5 + 1e-10)
        self.assertTrue(e.metadata["compensation_box_or_cap_limited"])

    def test_illegal_center_and_unfrozen_rules_rejected(self):
        illegal = self.center.copy()
        illegal[0, 0] = .1 - .1j
        with self.assertRaises(ValueError):
            self.edits(illegal)
        rules = deepcopy(self.rules)
        rules["calibration_split"] = "validation"
        with self.assertRaises(ValueError):
            self.edits(rules=rules)
        with self.assertRaises(ValueError):
            make_structural_edits(self.center, self.bg, self.rules, center_provenance="unknown")

    def test_rule_version_and_secondary_control_count(self):
        self.assertEqual(self.rules["version"], RULE_VERSION)
        self.assertEqual(len(self.edits(include_contrast=False)), 3)


if __name__ == "__main__":
    unittest.main()
