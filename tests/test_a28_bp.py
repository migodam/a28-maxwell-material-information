"""Tiny cached-array BP checks: no Maxwell material states or labels.

The parent runs these under the A28 CPU ledger.  Random finite arrays test
packing/information algebra only; they are not Maxwell experimental evidence.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from a25.torch_physics import mass_pack, mass_unpack, pack_data
from a28.bp import (audit_bp, build_bp_map, candidate_information,
                    covariance_information, factor_bp_matrix,
                    pack_measurement, unboxed_bp, unpack_measurement)


def _reference(seed=2810):
    rng = np.random.default_rng(seed)
    q, n, s = 3, 5, 2
    db = (rng.normal(size=(q, n)) + 1j*rng.normal(size=(q, n))).astype(np.complex128)
    eb = (rng.normal(size=(n, s)) + 1j*rng.normal(size=(n, s))).astype(np.complex128)
    yb = (rng.normal(size=(q, s)) + 1j*rng.normal(size=(q, s))).astype(np.complex128)
    return SimpleNamespace(Db=db, eb=eb, yb=yb, volume=.04, step=.007,
                           yb_norm2=float(np.sum(abs(yb)**2)), background_id="TINY_ARRAY_FIXTURE")


class BPPackingAndMapTests(unittest.TestCase):
    def test_source_major_real_packing_matches_frozen_torch(self):
        data = np.asarray([[1+2j, 3+4j], [5+6j, 7+8j]], np.complex128)
        np.testing.assert_array_equal(pack_measurement(data), [1.,5.,2.,6.,3.,7.,4.,8.])
        np.testing.assert_array_equal(pack_measurement(data), pack_data(torch.as_tensor(data)).numpy())
        np.testing.assert_array_equal(unpack_measurement(pack_measurement(data), 2, 2), data)
        batch = np.stack((data, 2*data))
        np.testing.assert_array_equal(unpack_measurement(pack_measurement(batch), 2, 2), batch)

    def test_map_is_exact_unboxed_a26_n3_formula_and_mass_scale(self):
        ref = _reference()
        delta = np.asarray([[.13-.17j, .05+.2j], [-.01+.11j, .03+.04j], [.1-.02j, -.2+.02j]], np.complex128)
        model = build_bp_map(ref)
        load = (torch.as_tensor(ref.yb)-torch.as_tensor(ref.yb+delta))/ref.yb_norm2
        adjoint = torch.as_tensor(ref.Db).mH @ load
        grad = mass_pack((torch.as_tensor(ref.eb).conj()*adjoint).sum(-1), 1.)/np.sqrt(ref.volume)
        actual_image = mass_unpack(-ref.step*grad, ref.volume).numpy()
        real_image = np.r_[actual_image.real, actual_image.imag]
        np.testing.assert_allclose(model["matrix"] @ pack_measurement(delta), real_image, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(unboxed_bp(ref, ref.yb+delta), actual_image, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(model["mass_matrix"], np.sqrt(ref.volume)*model["matrix"], rtol=1e-14, atol=0)
        self.assertEqual(model["matrix"].shape, (10, 12))
        self.assertEqual(model["summary"]["new_full_material_states"], 0)

    def test_complex_jvp_real_adjoint_and_conjugation(self):
        ref = _reference()
        rng = np.random.default_rng(2811)
        material = rng.normal(size=10)
        complex_delta = (material[:5]+1j*material[5:])/np.sqrt(ref.volume)
        tangent = ref.Db @ (complex_delta[:, None]*ref.eb)
        model = build_bp_map(ref)
        np.testing.assert_allclose(model["background_jacobian"] @ material, pack_measurement(tangent), rtol=1e-13, atol=1e-14)
        load = rng.normal(size=12)
        lhs = float(load @ pack_measurement(tangent))
        rhs = float(material @ (model["background_jacobian"].T @ load))
        self.assertAlmostEqual(lhs, rhs, places=12)

    def test_synchronous_source_permutation_does_not_change_bp_image(self):
        ref = _reference()
        rng = np.random.default_rng(2812)
        data = (rng.normal(size=ref.yb.shape)+1j*rng.normal(size=ref.yb.shape)).astype(np.complex128)
        perm = [1,0]
        changed = deepcopy(ref)
        changed.eb, changed.yb = ref.eb[:, perm], ref.yb[:, perm]
        np.testing.assert_allclose(unboxed_bp(ref, data), unboxed_bp(changed, data[:,perm]), rtol=1e-13, atol=1e-14)

    def test_cache_only_and_normalization_conflict_rejected(self):
        class CacheOnly(SimpleNamespace):
            @property
            def truth(self):
                raise AssertionError("truth read forbidden")
            @property
            def J(self):
                raise AssertionError("full J field must not be used")
        ref = CacheOnly(**vars(_reference()))
        build_bp_map(ref)
        ref.yb_norm2 *= 2
        with self.assertRaisesRegex(ValueError, "NORMALIZATION_MISMATCH"):
            build_bp_map(ref)

    def test_actual_channel_fp32_trace_and_pair_metadata(self):
        ref = _reference()
        a, b = ref.yb+.03j, ref.yb-.07
        ia, ib = unboxed_bp(ref, a), unboxed_bp(ref, b)
        actual_a = np.stack((ia.real, ia.imag)).astype(np.float32)
        actual_b = np.stack((ib.real, ib.imag)).astype(np.float32)
        result = audit_bp(ref, [{"data_a":a, "data_b":b, "sigma":.02,
                                 "pair_id":"tiny_pair", "family":"tiny_fixture",
                                 "actual_bp_a":actual_a, "actual_bp_b":actual_b}])
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual(result["summary"]["candidate_pair_count"], 1)
        for row in result["rows"]:
            self.assertEqual(row["pair_id"], "tiny_pair")
            self.assertLess(row["matrix_direct_relative_error"], 1e-13)
            self.assertLess(row["actual_feature_vs_expected_FP32_absolute_error"], 1e-12)
        self.assertIn("NOT_ESTABLISHED", result["summary"]["formal_injectivity"])

    def test_actual_full_twelve_channel_trace_extracts_unboxed_bp(self):
        ref = _reference()
        a, b = ref.yb+.03j, ref.yb-.07
        ia, ib = unboxed_bp(ref, a), unboxed_bp(ref, b)
        fa, fb = np.zeros((12, 1, 5), np.float32), np.zeros((12, 1, 5), np.float32)
        fa[2,0], fa[3,0], fb[2,0], fb[3,0] = ia.real, ia.imag, ib.real, ib.imag
        # A non-BP channel difference must be retained without pretending its
        # norm is a linear-map or Gaussian likelihood preservation certificate.
        fa[0] = 1.
        result = audit_bp(ref, [{"data_a":a, "data_b":b, "sigma":.02,
                                 "actual_features_a":fa, "actual_features_b":fb}])
        for row in result["rows"]:
            self.assertEqual(row["actual_feature_status"], "ACTUAL_12_CHANNEL_TRACE")
            self.assertEqual(row["whole_12_channel_noise_covariance_status"], "NOT_ESTIMATED_NONLINEAR_CHANNELS")
            self.assertGreater(row["actual_12_channel_difference_norm"], row["actual_BP_channel_difference_norm"])
            self.assertLess(row["actual_feature_vs_expected_FP32_absolute_error"], 1e-12)
            self.assertIn("actual_BP_feature_measurement_reconstruction_relative_error", row)
        with self.assertRaisesRegex(ValueError, "CHOOSE_EXTRACTED_OR_FULL"):
            audit_bp(ref, [{"data_a":a, "data_b":b, "sigma":.02,
                           "actual_features_a":fa, "actual_features_b":fb,
                           "actual_bp_a":ia, "actual_bp_b":ib}])


class BPInformationTests(unittest.TestCase):
    def test_fixed_rank_cutoffs_keep_tiny_singular_values_and_no_injectivity_claim(self):
        matrix = np.zeros((6,4))
        matrix[:4] = np.diag([1.,.1,1e-10,1e-14])
        factor = factor_bp_matrix(matrix)
        reports = {row["cutoff_id"]:row for row in factor["summary"]["rank_reports"]}
        self.assertEqual([reports[key]["numerical_rank"] for key in ("machine","1e-12","1e-9","1e-6")], [4,3,2,2])
        self.assertEqual(factor["summary"]["arithmetic_relative_min"], 1e-14)
        self.assertEqual(factor["summary"]["formal_injectivity"], "NOT_ESTABLISHED_BY_FINITE_PRECISION_SVD")

    def test_pair_covariance_likelihood_not_naive_bp_euclidean_norm(self):
        factor = factor_bp_matrix(np.diag([10.,.01,0.]))
        delta = np.asarray([1.,2.,3.])
        with patch("numpy.linalg.pinv", side_effect=AssertionError("no matrix pinv allowed")):
            rows = candidate_information(factor, delta, .5)
        for row in rows:
            self.assertAlmostEqual(row["data_whitened_distance"]**2, 2*14/.5**2)
            self.assertAlmostEqual(row["BP_induced_covariance_whitened_distance"]**2, 2*5/.5**2)
            self.assertAlmostEqual(row["information_lost_fraction"], 9/14)
            self.assertAlmostEqual(row["lost_pair_KL"], 9/.5**2)
            self.assertNotAlmostEqual(row["naive_BP_difference_norm"], row["BP_induced_covariance_whitened_distance"])
            self.assertLess(row["orthogonal_energy_identity_relative_error"], 1e-14)

    def test_rescaling_bp_map_does_not_change_likelihood_information(self):
        matrix = np.asarray([[1.,.5,0.],[.2,2.,0.]])
        delta = np.asarray([.3,.6,.8])
        left = candidate_information(factor_bp_matrix(matrix), delta, .1)
        right = candidate_information(factor_bp_matrix(1e-8*matrix), delta, .1)
        for a,b in zip(left,right):
            self.assertAlmostEqual(a["information_preserved_fraction"], b["information_preserved_fraction"], places=13)
            self.assertAlmostEqual(a["BP_induced_covariance_whitened_distance"], b["BP_induced_covariance_whitened_distance"], places=12)
            self.assertAlmostEqual(a["naive_BP_difference_norm"]*1e-8, b["naive_BP_difference_norm"], places=20)

    def test_correlated_noise_and_measurement_coordinate_transformation(self):
        matrix = np.asarray([[1.,2.,0.],[0.,.3,1.],[1.,0.,1.],[.2,.4,.6]])
        covariance = np.asarray([[2.,.2,.1],[.2,1.,.3],[.1,.3,3.]])
        delta = np.asarray([.7,-.4,.9])
        base = covariance_information(matrix, delta, covariance)
        transform = np.asarray([[2.,.2,0.],[0.,.7,.1],[0.,0.,3.]])
        transformed = covariance_information(matrix @ np.linalg.inv(transform),
            transform @ delta, transform @ covariance @ transform.T)
        for key in ("raw_mahalanobis_squared", "BP_mahalanobis_squared", "information_preserved_fraction"):
            self.assertAlmostEqual(base[key], transformed[key], places=12)
        self.assertAlmostEqual(base["information_preserved_fraction"], 1., places=12)

    def test_fp32_absolute_rounding_can_destroy_pair_despite_fp64_full_rank(self):
        factor = factor_bp_matrix(np.ones((1,1)))
        absolute = candidate_information(factor, [.25], 1., bp_a=np.asarray([1e8+.25]), bp_b=np.asarray([1e8]))
        delta_only = candidate_information(factor, [.25], 1.)
        self.assertAlmostEqual(absolute[0]["FP32_measurement_reconstruction_relative_error"], 1.)
        self.assertAlmostEqual(delta_only[0]["FP32_measurement_reconstruction_relative_error"], 0.)
        self.assertEqual(absolute[0]["information_preserved_fraction"], 1.)

    def test_zero_and_noise_free_cases_are_undefined_without_floor(self):
        factor = factor_bp_matrix(np.zeros((3,2)))
        rows = candidate_information(factor, [0.,0.], 0.)
        for row in rows:
            self.assertEqual(row["retained_rank"], 0)
            self.assertIsNone(row["information_preserved_fraction"])
            self.assertIsNone(row["data_whitened_distance"])
            self.assertIsNone(row["FP32_measurement_reconstruction_relative_error"])
            self.assertTrue(row["zero_pair"])
        nonzero = candidate_information(factor, [1.,2.], 0.)
        self.assertEqual(nonzero[0]["information_lost_fraction"], 1.)
        self.assertIsNone(nonzero[0]["BP_induced_covariance_whitened_distance"])

    def test_invalid_layout_noise_and_spd_refused(self):
        with self.assertRaises(ValueError):
            pack_measurement(np.zeros((2,3)))
        with self.assertRaises(ValueError):
            unpack_measurement(np.zeros(9), 2, 2)
        with self.assertRaises(ValueError):
            candidate_information(factor_bp_matrix(np.eye(2)), [1.,2.], -.1)
        with self.assertRaises(np.linalg.LinAlgError):
            covariance_information(np.eye(2), np.ones(2), np.diag([1.,-1.]))


if __name__ == "__main__":
    unittest.main()
