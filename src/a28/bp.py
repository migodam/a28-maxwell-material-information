"""OFFLINE A28-E1 information audit of the *actual* A26 N3 BP channel.

This module does not construct a material state, apply a Green operator, run
an encoder, or solve Maxwell.  It only reads the already certified known-
background Db, eb, yb, cell volume and frozen spectral step.  Its statistical
inverse is an explicitly truncated, OFFLINE covariance pseudoinverse written
in SVD coordinates.  It is never a physical solver or an online proposal.

A26 packs data source by source as [Re q, Im q].  Its unboxed N3 feature is
the physical material increment (not mass coordinates):

    BP(y) = step / (sqrt(volume) * ||yb||^2) * Jb.T * pack(y - yb),

where Jb is the real MASS-coordinate background material Jacobian.  The
initial box, later physical update and other feature channels are separate.
Finite-precision numerical ranks do not establish mathematical injectivity.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


RELATIVE_CUTOFFS = (("machine", None), ("1e-12", 1e-12),
                    ("1e-9", 1e-9), ("1e-6", 1e-6))


def _array(value):
    # Existing tensors only: no data loader, physics builder or GPU dispatch.
    if value is None:
        raise ValueError("A28_BP_ARRAY_REQUIRED")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("A28_BP_FINITE_ARRAY_REQUIRED")
    return value


def _get(reference, key, default=None):
    if isinstance(reference, Mapping):
        return reference.get(key, default)
    return getattr(reference, key, default)


def _scalar(value, name, *, positive=True):
    raw = _array(value)
    if raw.size != 1 or np.iscomplexobj(raw):
        raise ValueError("A28_BP_REAL_SCALAR_REQUIRED:" + name)
    scalar = float(raw.reshape(()))
    if not math.isfinite(scalar) or (positive and scalar <= 0):
        raise ValueError("A28_BP_POSITIVE_SCALAR_REQUIRED:" + name)
    return scalar


def pack_measurement(data):
    """(..., q, s) complex -> (..., s*[Re q, Im q]) real FP64."""
    data = _array(data)
    if not np.iscomplexobj(data) or data.ndim < 2:
        raise ValueError("A28_BP_COMPLEX_RECEIVER_SOURCE_DATA_REQUIRED")
    transposed = np.swapaxes(data, -1, -2)
    joined = np.concatenate((transposed.real, transposed.imag), axis=-1)
    return joined.reshape(*data.shape[:-2], 2 * data.shape[-2] * data.shape[-1]).astype(np.float64)


def unpack_measurement(packed, q, s):
    """Exact inverse of the declared source-major real packing."""
    packed = _array(packed)
    if packed.ndim < 1 or np.iscomplexobj(packed) or packed.shape[-1] != 2 * int(q) * int(s):
        raise ValueError("A28_BP_REAL_PACKED_DATA_SHAPE_MISMATCH")
    shaped = packed.reshape(*packed.shape[:-1], int(s), 2 * int(q))
    return np.swapaxes(shaped[..., :q] + 1j * shaped[..., q:], -1, -2)


def build_bp_map(reference):
    """Derive the linear map using an existing TorchReference or cache mapping.

    A FrozenEncoder is also accepted via its existing ``reference`` attribute.
    Returns raw matrices for a caller-owned NPZ; neither this function nor
    ``audit_bp`` writes files.  No full-J/oracle/truth field is accessed.
    """
    reference = _get(reference, "reference", reference)
    db, eb, yb = (_array(_get(reference, key)) for key in ("Db", "eb", "yb"))
    volume = _scalar(_get(reference, "volume"), "volume")
    step_value = _get(reference, "step", _get(reference, "fixed_step"))
    step = _scalar(step_value, "step")
    if (not all(np.iscomplexobj(value) for value in (db, eb, yb))
            or db.ndim != 2 or eb.ndim != 2 or yb.ndim != 2
            or db.shape[1] != eb.shape[0]
            or yb.shape != (db.shape[0], eb.shape[1])):
        raise ValueError("A28_BP_BACKGROUND_LAYOUT_MISMATCH")
    q, n, s = db.shape[0], db.shape[1], eb.shape[1]
    norm2 = float(np.sum(np.abs(yb) ** 2))
    if not math.isfinite(norm2) or norm2 <= 0:
        raise ValueError("A28_BP_POSITIVE_BACKGROUND_DATA_SCALE_REQUIRED")
    supplied_norm = _get(reference, "yb_norm2", _get(reference, "data_norm2"))
    if supplied_norm is not None and not math.isclose(
            norm2, _scalar(supplied_norm, "yb_norm2"), rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("A28_BP_BACKGROUND_DATA_NORMALIZATION_MISMATCH")
    response = np.einsum("qn,ns->sqn", db, eb) / math.sqrt(volume)
    real_columns = np.concatenate((response.real, response.imag), axis=1).reshape(2*q*s, n)
    imag_columns = np.concatenate((-response.imag, response.real), axis=1).reshape(2*q*s, n)
    jacobian = np.concatenate((real_columns, imag_columns), axis=1)
    mass_coefficient = step / norm2
    physical_coefficient = mass_coefficient / math.sqrt(volume)
    return {
        "matrix": physical_coefficient * jacobian.T,
        "mass_matrix": mass_coefficient * jacobian.T,
        "background_jacobian": jacobian,
        "summary": {
            "method": "A26_N3_UNBOXED_BP_PHYSICAL_IMAGE",
            "role": "OFFLINE_BACKGROUND_ENCODER_AUDIT",
            "data_packing": "source_major:[Re_q,Im_q]",
            "material_packing": "[Re_n,Im_n]",
            "matrix_shape": [2*n, 2*q*s], "n": n, "q": q, "sources": s,
            "volume": volume, "step": step, "background_data_norm2": norm2,
            "mass_coefficient": mass_coefficient,
            "physical_coefficient": physical_coefficient,
            "background_id": str(_get(reference, "background_id", "not_provided")),
            "new_full_material_states": 0,
            "affine_offset": "B*pack(y-yb); candidate differences cancel yb",
            "scope": "unboxed BP feature only; other N3 features may add information",
        },
        "background_packed": pack_measurement(yb),
        "Db": db, "eb": eb,
    }


def unboxed_bp(reference, observation):
    """Actual pre-box, pre-float32 N3 BP image from cached Db/eb only."""
    reference = _get(reference, "reference", reference)
    db, eb, yb = (_array(_get(reference, key)) for key in ("Db", "eb", "yb"))
    observation = _array(observation)
    if not np.iscomplexobj(observation) or observation.shape[-2:] != yb.shape:
        raise ValueError("A28_BP_OBSERVATION_LAYOUT_MISMATCH")
    volume = _scalar(_get(reference, "volume"), "volume")
    step = _scalar(_get(reference, "step", _get(reference, "fixed_step")), "step")
    norm2 = float(np.sum(np.abs(yb)**2))
    if not math.isfinite(norm2) or norm2 <= 0:
        raise ValueError("A28_BP_POSITIVE_BACKGROUND_DATA_SCALE_REQUIRED")
    adjoint = np.einsum("nq,...qs->...ns", db.conj().T, observation-yb)
    return (step / (volume * norm2)) * np.sum(eb.conj() * adjoint, axis=-1)


def factor_bp_matrix(matrix):
    """FP64 SVD and fixed relative rank reports, with raw tiny singular values."""
    matrix = _array(matrix)
    if np.iscomplexobj(matrix) or matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("A28_BP_REAL_MATRIX_REQUIRED")
    matrix = matrix.astype(np.float64)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    top = float(singular[0])
    relative = singular/top if top > 0 else np.zeros_like(singular)
    cutoffs, ranks = {}, []
    for name, value in RELATIVE_CUTOFFS:
        rtol = float(max(matrix.shape) * np.finfo(np.float64).eps if value is None else value)
        keep = relative > rtol if top > 0 else np.zeros(len(singular), dtype=bool)
        cutoffs[name] = keep
        retained = singular[keep]
        ranks.append({"cutoff_id": name, "relative_cutoff": rtol,
                      "absolute_cutoff": rtol*top, "numerical_rank": int(keep.sum()),
                      "data_dimension": int(matrix.shape[1]),
                      "full_column_numerical_rank": bool(keep.sum() == matrix.shape[1]),
                      "condition_retained": float(top/retained[-1]) if len(retained) else None,
                      "minimum_retained_singular": float(retained[-1]) if len(retained) else None,
                      "tiny_or_dropped_singular_count": int((~keep).sum())})
    return {"matrix": matrix, "left": u, "singular": singular, "right": vh,
            "cutoffs": cutoffs, "summary": {
                "shape": list(matrix.shape), "arithmetic_singular_max": top,
                "arithmetic_singular_min": float(singular[-1]),
                "arithmetic_relative_min": float(relative[-1]),
                "arithmetic_condition": float(top/singular[-1]) if singular[-1] > 0 else None,
                "tiny_last_16_singular_values": singular[-16:].tolist(),
                "tiny_last_16_relative_singular_values": relative[-16:].tolist(),
                "rank_reports": ranks,
                "precision": "numpy float64 SVD; numerical evidence only",
                "formal_injectivity": "NOT_ESTABLISHED_BY_FINITE_PRECISION_SVD",
                "inverse_role": "OFFLINE induced-noise covariance pseudoinverse; never Maxwell solver",
            }}


def _physical_packed(value, n):
    value = _array(value)
    if np.iscomplexobj(value):
        if value.size != n:
            raise ValueError("A28_BP_ACTUAL_IMAGE_SHAPE_MISMATCH")
        value = value.reshape(n)
        return np.concatenate((value.real, value.imag))
    if value.size != 2*n:
        raise ValueError("A28_BP_ACTUAL_IMAGE_SHAPE_MISMATCH")
    return value.reshape(2*n).astype(np.float64)


def candidate_information(factor, delta, sigma, *, bp_a=None, bp_b=None,
                          actual_bp_difference=None):
    """Pairwise likelihood-distance preservation, fixed cutoffs and FP32 audit.

    Noise is proper complex iid CN(0,sigma² I): each real datum has variance
    sigma²/2.  Sigma==0 leaves geometric capture defined, but likelihood/KL
    statistics undefined.  Zero pair differences similarly have no fraction.
    BP Euclidean norm is not a likelihood distance.
    """
    matrix, u, singular, vh = (factor[key] for key in ("matrix", "left", "singular", "right"))
    delta = _array(delta)
    if np.iscomplexobj(delta) or delta.shape != (matrix.shape[1],):
        raise ValueError("A28_BP_CANDIDATE_PACKING_MISMATCH")
    delta = delta.astype(np.float64)
    sigma = _scalar(sigma, "sigma", positive=False)
    if sigma < 0:
        raise ValueError("A28_BP_NONNEGATIVE_SIGMA_REQUIRED")
    norm2 = float(delta @ delta)
    delta_bp = matrix @ delta
    if (bp_a is None) != (bp_b is None):
        raise ValueError("A28_BP_BOTH_ABSOLUTE_IMAGES_REQUIRED_FOR_FP32")
    if bp_a is None:
        # This probes rounding of the difference only; absolute feature pairs
        # are preferred because cancellation after two rounded maps differs.
        delta_32 = delta_bp.astype(np.float32).astype(np.float64)
        rounding_mode = "DIFFERENCE_ONLY_NO_ABSOLUTE_PAIR_PROVIDED"
    else:
        bp_a, bp_b = _array(bp_a), _array(bp_b)
        if bp_a.shape != delta_bp.shape or bp_b.shape != delta_bp.shape:
            raise ValueError("A28_BP_ABSOLUTE_IMAGE_PACKING_MISMATCH")
        delta_32 = bp_a.astype(np.float32).astype(np.float64) - bp_b.astype(np.float32).astype(np.float64)
        rounding_mode = "DIFFERENCE_OF_TWO_ROUNDED_ABSOLUTE_BP_FEATURES"
    spectral_bp64, spectral_bp32 = u.T @ delta_bp, u.T @ delta_32
    spectral_actual = None
    if actual_bp_difference is not None:
        actual_delta = _array(actual_bp_difference)
        if np.iscomplexobj(actual_delta) or actual_delta.shape != delta_bp.shape:
            raise ValueError("A28_BP_ACTUAL_FEATURE_PACKING_MISMATCH")
        spectral_actual = u.T @ actual_delta
    coordinates = vh @ delta
    rows = []
    for cutoff_id, keep in factor["cutoffs"].items():
        retained = vh[keep].T @ coordinates[keep]
        lost = delta-retained
        retained2, lost2 = float(retained @ retained), float(lost @ lost)
        recovered64 = vh[keep].T @ (spectral_bp64[keep] / singular[keep])
        recovered32 = vh[keep].T @ (spectral_bp32[keep] / singular[keep])
        total_d2 = 2*norm2/sigma**2 if sigma > 0 else None
        retained_d2 = 2*retained2/sigma**2 if sigma > 0 else None
        lost_d2 = 2*lost2/sigma**2 if sigma > 0 else None
        row = {
            "cutoff_id": cutoff_id, "retained_rank": int(keep.sum()),
            "raw_data_difference_norm": math.sqrt(norm2),
            "naive_BP_difference_norm": float(np.linalg.norm(delta_bp)),
            "naive_BP_difference_is_likelihood": False,
            "data_whitened_distance": math.sqrt(total_d2) if total_d2 is not None else None,
            "BP_induced_covariance_whitened_distance": math.sqrt(retained_d2) if retained_d2 is not None else None,
            "lost_whitened_distance": math.sqrt(lost_d2) if lost_d2 is not None else None,
            "lost_pair_KL": lost_d2/2 if lost_d2 is not None else None,
            "information_preserved_fraction": retained2/norm2 if norm2 > 0 else None,
            "information_lost_fraction": lost2/norm2 if norm2 > 0 else None,
            "orthogonal_energy_identity_relative_error": abs(norm2-retained2-lost2)/norm2 if norm2 > 0 else None,
            "FP64_measurement_reconstruction_relative_error": float(np.linalg.norm(recovered64-delta))/math.sqrt(norm2) if norm2 > 0 else None,
            "FP32_measurement_reconstruction_relative_error": float(np.linalg.norm(recovered32-delta))/math.sqrt(norm2) if norm2 > 0 else None,
            "FP64_retained_coordinates_relative_error": float(np.linalg.norm(recovered64-retained))/math.sqrt(retained2) if retained2 > 0 else None,
            "FP32_retained_coordinates_relative_error": float(np.linalg.norm(recovered32-retained))/math.sqrt(retained2) if retained2 > 0 else None,
            "FP32_rounding_mode": rounding_mode,
            "noise_status": "POSITIVE_PROPER_COMPLEX_GAUSSIAN" if sigma > 0 else "NOISE_FREE_GEOMETRIC_ONLY",
            "zero_pair": norm2 == 0,
            "covariance_inverse_role": "OFFLINE_ONLY:(sigma²/2 * B Bᵀ)^+ at declared B-SVD cutoff",
        }
        if spectral_actual is not None:
            recovered_actual = vh[keep].T @ (spectral_actual[keep]/singular[keep])
            row.update(actual_BP_feature_measurement_reconstruction_relative_error=
                       float(np.linalg.norm(recovered_actual-delta))/math.sqrt(norm2) if norm2 > 0 else None,
                       actual_BP_feature_retained_coordinates_relative_error=
                       float(np.linalg.norm(recovered_actual-retained))/math.sqrt(retained2) if retained2 > 0 else None)
        rows.append(row)
    return rows


def audit_bp(reference, data_pairs=None):
    """Audit cached background BP and optional frozen candidate responses.

    Each pair requires ``data_a`` and ``data_b`` complex (q,s), plus ``sigma``.
    Metadata ``pair_id``, ``parent_id``, ``family``, ``background_id`` and
    ``candidate_kind`` is copied for OFFLINE grouping.  Optional ``actual_bp_a``
    / ``actual_bp_b`` are extracted A26 channels 2/3 (complex n or real [2,n]).
    Alternatively ``actual_features_a``/``actual_features_b`` accept actual
    pre-normalizer 12-channel samples and extract channels 2/3.  Complete
    feature differences are recorded but are not treated as a linear Gaussian
    sufficient statistic: other channels include boxes and nonlinearities.
    These comparisons do not call the encoder again.
    Return arrays separately from JSON-compatible summary/rows for caller-owned
    accounting and storage.  No cohort decisions or scientific verdict here.
    """
    model = build_bp_map(reference)
    factor = factor_bp_matrix(model["matrix"])
    n, q, s = (model["summary"][key] for key in ("n", "q", "sources"))
    rows = []
    for index, pair in enumerate(data_pairs or ()):
        if not isinstance(pair, Mapping) or not {"data_a", "data_b", "sigma"} <= set(pair):
            raise ValueError("A28_BP_FROZEN_RESPONSE_PAIR_REQUIRED")
        a, b = _array(pair["data_a"]), _array(pair["data_b"])
        if (not np.iscomplexobj(a) or not np.iscomplexobj(b)
                or a.shape != (q, s) or b.shape != (q, s)):
            raise ValueError("A28_BP_CANDIDATE_RESPONSE_LAYOUT_MISMATCH")
        pa, pb = pack_measurement(a), pack_measurement(b)
        bp_a = model["matrix"] @ (pa-model["background_packed"])
        bp_b = model["matrix"] @ (pb-model["background_packed"])
        direct_a = _physical_packed(unboxed_bp(reference, a), n)
        direct_b = _physical_packed(unboxed_bp(reference, b), n)
        matrix_direct_absolute_error = max(float(np.linalg.norm(bp_a-direct_a)), float(np.linalg.norm(bp_b-direct_b)))
        direct_scale = max(float(np.linalg.norm(direct_a)), float(np.linalg.norm(direct_b)))
        pair_metadata = {key: pair[key] for key in ("pair_id", "parent_id", "family", "background_id", "candidate_kind") if key in pair}
        pair_metadata.setdefault("pair_id", str(index))
        pair_metadata.update(matrix_direct_absolute_error=matrix_direct_absolute_error,
                             matrix_direct_relative_error=matrix_direct_absolute_error/direct_scale if direct_scale > 0 else None)
        actual_delta = None
        if ("actual_bp_a" in pair) != ("actual_bp_b" in pair):
            raise ValueError("A28_BP_BOTH_ACTUAL_FEATURE_PAIRS_REQUIRED")
        if ("actual_features_a" in pair) != ("actual_features_b" in pair):
            raise ValueError("A28_BP_BOTH_ACTUAL_12_CHANNEL_PAIRS_REQUIRED")
        if "actual_bp_a" in pair and "actual_features_a" in pair:
            raise ValueError("A28_BP_CHOOSE_EXTRACTED_OR_FULL_FEATURE_TRACE")
        if "actual_features_a" in pair:
            fa, fb = (_array(pair[key]) for key in ("actual_features_a", "actual_features_b"))
            if (np.iscomplexobj(fa) or np.iscomplexobj(fb) or fa.shape != fb.shape
                    or fa.ndim < 2 or fa.shape[0] != 12 or fa[0].size != n):
                raise ValueError("A28_BP_ACTUAL_12_CHANNEL_SAMPLE_SHAPE_MISMATCH")
            actual_a, actual_b = fa[2:4].reshape(2*n).astype(np.float64), fb[2:4].reshape(2*n).astype(np.float64)
            actual_delta = actual_a-actual_b
            pair_metadata.update(actual_12_channel_difference_norm=float(np.linalg.norm(fa.astype(np.float64)-fb.astype(np.float64))),
                                 actual_BP_channel_difference_norm=float(np.linalg.norm(actual_delta)),
                                 actual_features_input_semantics="pre-normalizer A26 FEATURE_NAMES; BP channels 2/3",
                                 whole_12_channel_noise_covariance_status="NOT_ESTIMATED_NONLINEAR_CHANNELS",
                                 actual_feature_status="ACTUAL_12_CHANNEL_TRACE")
        if "actual_bp_a" in pair:
            actual_a, actual_b = (_physical_packed(pair[key], n) for key in ("actual_bp_a", "actual_bp_b"))
            actual_delta = actual_a-actual_b
            pair_metadata["actual_feature_status"] = "ACTUAL_EXTRACTED_BP_CHANNEL_TRACE"
        if actual_delta is not None:
            expected_delta32 = bp_a.astype(np.float32).astype(np.float64)-bp_b.astype(np.float32).astype(np.float64)
            pair_metadata.update(actual_feature_absolute_difference_norm=float(np.linalg.norm(actual_delta)),
                                 actual_feature_vs_expected_FP32_absolute_error=float(np.linalg.norm(actual_delta-expected_delta32)))
        else:
            pair_metadata["actual_feature_status"] = "NOT_PROVIDED; FP32 conversion simulated from exact BP map"
        for row in candidate_information(factor, pa-pb, pair["sigma"], bp_a=bp_a, bp_b=bp_b,
                                         actual_bp_difference=actual_delta):
            rows.append({**pair_metadata, **row})
    summary = {**model["summary"], **factor["summary"],
               "candidate_pair_count": len(rows)//len(RELATIVE_CUTOFFS),
               "noise_covariance": "real data=(sigma²/2)I; BP=(sigma²/2) B Bᵀ",
               "statistical_scope": "finite declared response pairs; not global material identifiability",
               "float32_scope": "BP feature cast is not treated as exact Gaussian linear map; decoder finite capacity not analyzed",
               "other_channel_scope": "full 12-channel features include nonlinear maps; their covariance/information loss is not established by the BP-only SVD"}
    return {"summary": summary, "rows": rows,
            "arrays": {"BP_matrix": model["matrix"], "BP_mass_matrix": model["mass_matrix"],
                       "J_background": model["background_jacobian"],
                       "BP_singular": factor["singular"], "BP_left": factor["left"],
                       "BP_right": factor["right"]}}


def covariance_information(matrix, difference, noise_covariance, relative_cutoff=1e-12):
    """General SPD-noise version for offline unit/coordinate checks.

    A Cholesky whitening defines C=B*chol(Sigma).  With C=U S Vᵀ,
    (B delta)ᵀ (B Sigma Bᵀ)^+ (B delta)=||V_ret delta_white||².
    The covariance pseudoinverse is intentionally never formed.  Cutoff is
    relative to C's singular values, not covariance eigenvalues (squared).
    """
    matrix, difference, covariance = map(_array, (matrix, difference, noise_covariance))
    if (np.iscomplexobj(matrix) or np.iscomplexobj(difference) or np.iscomplexobj(covariance)
            or matrix.ndim != 2 or difference.shape != (matrix.shape[1],)
            or covariance.shape != (matrix.shape[1], matrix.shape[1])
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0)):
        raise ValueError("A28_BP_REAL_MATCHED_SPD_NOISE_COVARIANCE_REQUIRED")
    if not math.isfinite(float(relative_cutoff)) or not 0 < relative_cutoff < 1:
        raise ValueError("A28_BP_RELATIVE_CUTOFF_REQUIRED")
    root = np.linalg.cholesky(covariance)
    white = np.linalg.solve(root, difference)
    _, singular, vh = np.linalg.svd(matrix @ root, full_matrices=False)
    keep = (singular > relative_cutoff * singular[0]) if singular[0] > 0 else np.zeros(len(singular), bool)
    retained = vh[keep].T @ (vh[keep] @ white)
    total2, retained2 = float(white @ white), float(retained @ retained)
    return {"raw_mahalanobis_squared": total2, "BP_mahalanobis_squared": retained2,
            "lost_mahalanobis_squared": float(np.sum((white-retained)**2)),
            "information_preserved_fraction": retained2/total2 if total2 > 0 else None,
            "numerical_rank": int(keep.sum()), "relative_cutoff": float(relative_cutoff),
            "role": "OFFLINE_STATISTICAL_COVARIANCE_PSEUDOINVERSE_ONLY"}
