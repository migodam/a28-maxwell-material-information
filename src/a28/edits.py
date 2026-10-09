"""TRAIN-frozen, full-grid material edits for A28-E1.

This module never loads data, solves physics, scores candidates, or accepts a
family, truth boundary, or material-error direction.  The exact same rules
operate on an OFFLINE truth center and an ONLINE CNN center.  Inputs and
outputs are absolute physical ``chi`` (complex128), not mass coordinates.
Native C-order raster axes follow A2's ``meshgrid(indexing='ij')``.

The support rule is a scalar projection along the fixed TRAIN phase; it is a
candidate-generation convention, not a recoverability or topology certificate.
Invalid/zero edits are returned in place and are never resampled.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import json
import math

import numpy as np
from scipy import ndimage as ndi


RULE_VERSION = "A28-E1-TRAIN-MORPHOLOGY-v1"
TRAIN_PARENT_IDS = frozenset(range(250001, 250065))
CENTER_ROLES = frozenset({"ONLINE_CNN_PREDICTION", "OFFLINE_TRUTH_DIAGNOSTIC",
                          "SYNTHETIC_UNIT_TEST"})
EDIT_ORDER = ("CAVITY", "BOUNDARY", "COMPONENT", "CONTRAST_ONLY")
_CONNECTIVITY = ndi.generate_binary_structure(2, 1)
_DEFAULTS = {
    "support_threshold_fraction": .25,
    "fill_support_quantile": .75,
    "cavity_radius_cells": 1,
    "boundary_radius_cells": 1,
    "boundary_mode": "dilate",
    "component_radius_cells": 1,
    "component_max_cells": 5,
    "component_gap_cells": 1,
    "component_preferred_offset_cells": 4,
    "contrast_multiplier": .85,
    "compensation_strength": 1.,
    "compensation_cap_fraction": .25,
    "real_lower": -.5,
    "real_upper": 4.,
    "imag_lower": 0.,
    "imag_upper": .5,
    "feasibility_tolerance": 1e-10,
    "zero_edit_relative_tolerance": 1e-12,
}


@dataclass(frozen=True)
class MaterialEdit:
    method_id: str
    chi: np.ndarray
    change: np.ndarray
    primary_change: np.ndarray
    compensation_change: np.ndarray
    primary_mask: np.ndarray
    compensation_mask: np.ndarray
    status: str
    metadata: dict

    @property
    def valid(self):
        return self.status == "OK"

    @property
    def delta(self):
        """Exact candidate-minus-center change (not candidate-minus-background)."""
        return self.change

    def summary(self):
        return {"method_id": self.method_id, "status": self.status,
                "valid": self.valid, **self.metadata}


def _json_safe_rules(rules):
    if not isinstance(rules, Mapping):
        raise ValueError("A28_FROZEN_RULES_OBJECT_REQUIRED")
    # JSON round trip provides detached primitives and rejects nonfinite values.
    return json.loads(json.dumps(dict(rules), allow_nan=False))


def _validate_rules(rules):
    rules = _json_safe_rules(rules)
    if rules.get("version") != RULE_VERSION or rules.get("calibration_split") != "train":
        raise ValueError("A28_TRAIN_FROZEN_RULES_REQUIRED")
    for key in _DEFAULTS:
        if key not in rules:
            raise ValueError("A28_INCOMPLETE_FROZEN_RULES:" + key)
    grid = rules.get("grid")
    if (not isinstance(grid, list) or len(grid) != 2 or
            any(not isinstance(x, int) or x < 3 for x in grid)):
        raise ValueError("A28_TWO_DIMENSIONAL_GRID_REQUIRED")
    if rules.get("raster_order") != "C_NATIVE_A2_IJ":
        raise ValueError("A28_NATIVE_C_ORDER_REQUIRED")
    phase = complex(*rules.get("phase", []))
    if (not np.isfinite(phase) or abs(abs(phase) - 1.) > 1e-12 or
            phase.real <= 0 or phase.imag < 0):
        raise ValueError("A28_POSITIVE_NORMALIZED_TRAIN_PHASE_REQUIRED")
    for key in ("material_scale", "support_threshold", "support_threshold_fraction",
                "compensation_cap_fraction", "feasibility_tolerance",
                "zero_edit_relative_tolerance"):
        if not math.isfinite(float(rules[key])) or float(rules[key]) <= 0:
            raise ValueError("A28_POSITIVE_RULE_REQUIRED:" + key)
    if not math.isclose(float(rules["support_threshold"]),
                        float(rules["material_scale"]) *
                        float(rules["support_threshold_fraction"]),
                        rel_tol=1e-12, abs_tol=0.):
        raise ValueError("A28_THRESHOLD_SCALE_MISMATCH")
    if rules["boundary_mode"] not in {"dilate", "erode"}:
        raise ValueError("A28_FIXED_BOUNDARY_MODE_REQUIRED")
    for key in ("cavity_radius_cells", "boundary_radius_cells",
                "component_radius_cells", "component_max_cells",
                "component_gap_cells", "component_preferred_offset_cells"):
        if not isinstance(rules[key], int) or rules[key] < 1:
            raise ValueError("A28_POSITIVE_INTEGER_RULE_REQUIRED:" + key)
    if (not 0 < float(rules["fill_support_quantile"]) <= 1 or
            not 0 < float(rules["contrast_multiplier"]) < 1 or
            not 0 <= float(rules["compensation_strength"]) <= 1):
        raise ValueError("A28_INVALID_CONTRAST_OR_COMPENSATION_RULE")
    if not (rules["real_lower"] < rules["real_upper"] and
            0 <= rules["imag_lower"] < rules["imag_upper"]):
        raise ValueError("A28_ABSOLUTE_MATERIAL_BOX_REQUIRED")
    return rules, phase, tuple(grid)


def _grid(value, grid, name):
    value = np.asarray(value, dtype=np.complex128)
    if value.shape not in {grid, (math.prod(grid),)}:
        raise ValueError("A28_FULL_GRID_COMPLEX_MATERIAL_REQUIRED:" + name)
    if not np.isfinite(value).all():
        raise ValueError("A28_NONFINITE_MATERIAL:" + name)
    return value.reshape(grid, order="C").copy()


def _box_defect(chi, rules):
    return float(max(0., np.max(rules["real_lower"] - chi.real),
                     np.max(chi.real - rules["real_upper"]),
                     np.max(rules["imag_lower"] - chi.imag),
                     np.max(chi.imag - rules["imag_upper"])))


def _feasible(chi, rules, name):
    defect = _box_defect(chi, rules)
    if defect > float(rules["feasibility_tolerance"]):
        raise ValueError("A28_ILLEGAL_MATERIAL_CENTER:" + name)


def calibrate_train_rules(chi, background, *, split, parent_ids,
                          amplitudes=None, grid=(24, 24), phase=(1., .2),
                          reference_amplitude=1.2, options=None):
    """Freeze scalar/morphology rules using TRAIN labels only, without physics.

    When amplitudes are provided, the prespecified high-amplitude TRAIN rows
    are used.  Peak contrast is first averaged by median within each parent,
    then median across parents.  No outcome/physics labels are accepted.
    Callers must write the returned rules/audit before opening development.
    """
    grid = tuple(map(int, grid))
    if len(grid) != 2 or min(grid) < 3:
        raise ValueError("A28_TWO_DIMENSIONAL_GRID_REQUIRED")
    chi = np.asarray(chi, dtype=np.complex128)
    if chi.ndim == 2 and chi.shape[-1] == math.prod(grid):
        chi = chi.reshape((-1,) + grid)
    if chi.ndim != 3 or chi.shape[1:] != grid or not np.isfinite(chi).all():
        raise ValueError("A28_TRAIN_FULL_GRID_BANK_REQUIRED")
    m = chi.shape[0]
    split_values = np.asarray(split)
    if split_values.shape == ():
        split_values = np.repeat(split_values, m)
    if split_values.shape != (m,) or not np.all(split_values == "train"):
        raise ValueError("A28_CALIBRATION_TRAIN_ONLY")
    parents = np.asarray(parent_ids)
    if (parents.shape != (m,) or parents.dtype.kind not in "iu" or not m or
            not set(map(int, parents)).issubset(TRAIN_PARENT_IDS)):
        raise ValueError("A28_CALIBRATION_TRAIN_PARENT_WHITELIST")
    background = np.asarray(background, dtype=np.complex128)
    if background.shape == (math.prod(grid),):
        background = background.reshape(grid)
    if background.shape == grid:
        background = np.broadcast_to(background, chi.shape)
    elif background.shape == (m, math.prod(grid)):
        background = background.reshape(chi.shape)
    if background.shape != chi.shape or not np.isfinite(background).all():
        raise ValueError("A28_KNOWN_BACKGROUND_BANK_REQUIRED")
    custom = dict(options or {})
    if set(custom) - set(_DEFAULTS):
        raise ValueError("A28_UNKNOWN_RULE_OPTION")
    rules = {**_DEFAULTS, **custom}
    _feasible(chi, rules, "train")
    _feasible(background, rules, "background")
    p = complex(*phase)
    if not np.isfinite(p) or abs(p) == 0 or p.real <= 0 or p.imag < 0:
        raise ValueError("A28_PASSIVE_TRAIN_PHASE_REQUIRED")
    p /= abs(p)
    selection = np.ones(m, dtype=bool)
    if amplitudes is not None:
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        if amplitudes.shape != (m,) or not np.isfinite(amplitudes).all():
            raise ValueError("A28_ALIGNED_TRAIN_AMPLITUDES_REQUIRED")
        selection = np.isclose(amplitudes, float(reference_amplitude), rtol=0, atol=1e-12)
        if not selection.any():
            raise ValueError("A28_PRESPECIFIED_TRAIN_AMPLITUDE_MISSING")
    delta = chi - background
    scalar = np.real(delta * p.conjugate())
    peaks = np.max(scalar, axis=(1, 2))
    used_parents = sorted(set(map(int, parents[selection])))
    parent_peak = {str(pid): float(np.median(peaks[selection & (parents == pid)]))
                   for pid in used_parents}
    scale = float(np.median(list(parent_peak.values())))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("A28_NONZERO_TRAIN_CONTRAST_REQUIRED")
    threshold = scale * float(rules["support_threshold_fraction"])
    residual = delta[selection] - p * scalar[selection]
    denominator = float(np.linalg.norm(delta[selection].ravel()))
    phase_error = float(np.linalg.norm(residual.ravel()) / denominator) if denominator else None
    rules.update(version=RULE_VERSION, calibration_split="train", grid=list(grid),
                 raster_order="C_NATIVE_A2_IJ", phase=[float(p.real), float(p.imag)],
                 material_scale=scale, support_threshold=threshold,
                 calibration_parent_ids=used_parents,
                 calibration_reference_amplitude=(float(reference_amplitude)
                     if amplitudes is not None else None))
    rules, _, _ = _validate_rules(rules)
    audit = {"version": RULE_VERSION, "data_role": "TRAIN_ONLY_CALIBRATION",
             "row_count_available": m, "row_count_used": int(selection.sum()),
             "independent_parent_count": len(used_parents),
             "parent_peak_contrasts": parent_peak, "parent_balanced_peak_median": scale,
             "support_threshold": threshold, "fixed_phase_relative_error": phase_error,
             "physics_calls": 0, "development_reads": 0, "locked_reads": 0,
             "choice_rule": "prespecified high amplitude; median within then across TRAIN parents",
             "scope": "structural candidate convention, not inferred recoverability"}
    return rules, audit


def _disk(radius):
    axis = np.arange(-radius, radius + 1)
    ii, jj = np.meshgrid(axis, axis, indexing="ij")
    return ii * ii + jj * jj <= radius * radius


def _largest_component(mask):
    labels, count = ndi.label(mask, _CONNECTIVITY)
    if not count:
        return np.zeros_like(mask), 0
    sizes = np.bincount(labels.ravel())[1:]
    index = int(np.argmax(sizes)) + 1  # C-order first label breaks equal-size ties.
    return labels == index, int(sizes[index - 1])


def _phase_interval(base, phase, rules):
    """Per-cell scalar intervals for base + t*phase in the absolute box."""
    lower = np.full(base.shape, -np.inf)
    upper = np.full(base.shape, np.inf)
    for values, component, lo, hi in ((base.real, phase.real, rules["real_lower"], rules["real_upper"]),
                                      (base.imag, phase.imag, rules["imag_lower"], rules["imag_upper"])):
        if component != 0:
            a, b = (lo - values) / component, (hi - values) / component
            lower = np.maximum(lower, np.minimum(a, b))
            upper = np.minimum(upper, np.maximum(a, b))
    return lower, upper


def _replace_with_amplitude(raw, background, mask, amplitude, phase, rules):
    lower, upper = _phase_interval(background[mask], phase, rules)
    clipped = np.clip(float(amplitude), lower, upper)
    raw[mask] = background[mask] + phase * clipped
    return int(np.count_nonzero(np.abs(clipped - float(amplitude)) > 1e-14))


def _cavity(center, background, support, fill, phase, rules):
    raw = center.copy()
    holes = ndi.binary_fill_holes(support, structure=_CONNECTIVITY) & ~support
    target, count = _largest_component(holes)
    if count:
        clipped = _replace_with_amplitude(raw, background, target, fill, phase, rules)
        return raw, target, {"operation": "FILL_LARGEST_EXISTING_HOLE",
                             "primary_box_limited_cells": clipped}
    radius = rules["cavity_radius_cells"]
    distance = ndi.distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
    eligible = support & (distance >= radius + 1.)
    if not eligible.any():
        return raw, np.zeros_like(support), {"operation": "OPEN_INTERIOR_CAVITY",
                                            "invalid_reason": "NO_FULL_INTERIOR_CAVITY_SITE"}
    score = np.where(eligible, distance, -np.inf)
    site = np.unravel_index(int(np.argmax(score)), support.shape)
    seed = np.zeros_like(support)
    seed[site] = True
    target = ndi.binary_dilation(seed, structure=_disk(radius)) & support
    if np.count_nonzero(support & ~target) < 1:
        return raw, np.zeros_like(support), {"operation": "OPEN_INTERIOR_CAVITY",
                                            "invalid_reason": "WOULD_REMOVE_ENTIRE_SUPPORT"}
    raw[target] = background[target]
    return raw, target, {"operation": "OPEN_INTERIOR_CAVITY", "site_ij": list(map(int, site)),
                         "primary_box_limited_cells": 0}


def _boundary(center, background, support, fill, phase, rules):
    raw = center.copy()
    footprint = _disk(rules["boundary_radius_cells"])
    if rules["boundary_mode"] == "dilate":
        target = ndi.binary_dilation(support, structure=footprint) & ~support
        clipped = _replace_with_amplitude(raw, background, target, fill, phase, rules)
        return raw, target, {"operation": "DILATE_BOUNDARY",
                             "primary_box_limited_cells": clipped}
    retained = ndi.binary_erosion(support, structure=footprint, border_value=0)
    if not retained.any():
        return raw, np.zeros_like(support), {"operation": "ERODE_BOUNDARY",
                                            "invalid_reason": "WOULD_REMOVE_ENTIRE_SUPPORT"}
    target = support & ~retained
    raw[target] = background[target]
    return raw, target, {"operation": "ERODE_BOUNDARY", "primary_box_limited_cells": 0}


def _component(center, background, support, fill, phase, rules):
    raw = center.copy()
    labels, count = ndi.label(support, _CONNECTIVITY)
    sizes = np.bincount(labels.ravel())[1:]
    eligible = [i + 1 for i in range(count)
                if sizes[i] <= rules["component_max_cells"] and sizes[i] < support.sum()]
    if eligible:
        index = min(eligible, key=lambda i: (int(sizes[i - 1]), i))
        target = labels == index
        raw[target] = background[target]
        return raw, target, {"operation": "REMOVE_SMALLEST_ELIGIBLE_COMPONENT",
                             "removed_component_cells": int(sizes[index - 1]),
                             "primary_box_limited_cells": 0}
    radius = rules["component_radius_cells"]
    distance = ndi.distance_transform_edt(~support)
    ii, jj = np.indices(support.shape)
    full_disk = ((ii >= radius) & (ii < support.shape[0] - radius) &
                 (jj >= radius) & (jj < support.shape[1] - radius))
    eligible = (~support & full_disk &
                (distance > radius + rules["component_gap_cells"]))
    if not eligible.any():
        return raw, np.zeros_like(support), {"operation": "ADD_SEPARATE_SMALL_COMPONENT",
                                            "invalid_reason": "NO_SEPARATED_FULL_DISK_SITE"}
    centroid = np.mean(np.argwhere(support), axis=0)
    preferred = centroid + np.array([rules["component_preferred_offset_cells"], 0.])
    score = (ii - preferred[0])**2 + (jj - preferred[1])**2
    score = np.where(eligible, score, np.inf)
    site = np.unravel_index(int(np.argmin(score)), support.shape)
    seed = np.zeros_like(support)
    seed[site] = True
    target = ndi.binary_dilation(seed, structure=_disk(radius))
    clipped = _replace_with_amplitude(raw, background, target, fill, phase, rules)
    return raw, target, {"operation": "ADD_SEPARATE_SMALL_COMPONENT",
                         "site_ij": list(map(int, site)), "primary_box_limited_cells": clipped}


def _compensate(center, raw, primary_mask, support, phase, rules):
    # Never modify the primary edit cells, so compensation cannot erase it.
    target = support & ~primary_mask
    requested = float(rules["compensation_strength"]) * float(np.sum(
        np.real((center - raw) * phase.conjugate())))
    if not target.any() or requested == 0:
        return raw.copy(), np.zeros_like(support), {
            "compensation_requested_integrated_scalar": requested,
            "compensation_applied_integrated_scalar": 0., "compensation_scalar_per_cell": 0.,
            "compensation_box_or_cap_limited": bool(requested != 0)}
    count = int(target.sum())
    scalar = requested / count
    cap = float(rules["compensation_cap_fraction"]) * float(rules["material_scale"])
    lower, upper = _phase_interval(raw[target], phase, rules)
    scalar = float(np.clip(scalar, max(-cap, float(np.max(lower))),
                           min(cap, float(np.min(upper)))))
    final = raw.copy()
    final[target] += phase * scalar
    applied = float(np.sum(np.real((final - raw) * phase.conjugate())))
    return final, target & (scalar != 0), {
        "compensation_requested_integrated_scalar": requested,
        "compensation_applied_integrated_scalar": applied,
        "compensation_scalar_per_cell": scalar,
        "compensation_box_or_cap_limited": bool(abs(applied - requested) >
                                                 1e-12 * max(1., abs(requested)))}


def make_structural_edits(center, background, rules, *,
                          center_provenance="ONLINE_CNN_PREDICTION",
                          include_contrast=True):
    """Return the frozen three compensated structures plus a contrast control.

    ``center_provenance`` is copied to the audit and never changes candidate
    generation.  OFFLINE truth-centered edits are diagnostic-only.  Empty
    support and unavailable morphology produce invalid rows, not fresh draws.
    All output arrays are flat native complex/material or bool vectors.
    """
    if center_provenance not in CENTER_ROLES:
        raise ValueError("A28_EXPLICIT_CENTER_INFORMATION_ROLE_REQUIRED")
    rules, phase, grid = _validate_rules(rules)
    center = _grid(center, grid, "center")
    background = _grid(background, grid, "known_background")
    _feasible(center, rules, "center")
    _feasible(background, rules, "known_background")
    amplitude = np.real((center - background) * phase.conjugate())
    support = amplitude >= float(rules["support_threshold"])
    fill = (float(np.quantile(amplitude[support], rules["fill_support_quantile"]))
            if support.any() else 0.)
    outputs = []
    builders = (("CAVITY", _cavity), ("BOUNDARY", _boundary), ("COMPONENT", _component))
    for method_id, builder in builders:
        if support.any():
            raw, mask, meta = builder(center, background, support, fill, phase, rules)
        else:
            raw, mask, meta = center.copy(), np.zeros(grid, dtype=bool), {
                "operation": method_id, "invalid_reason": "EMPTY_FIXED_THRESHOLD_SUPPORT"}
        final, comp_mask, comp_meta = _compensate(center, raw, mask, support, phase, rules)
        # Unavailable primary edit stays invalid even if numerical compensation
        # were to introduce an unrelated change.
        primary_norm = float(np.linalg.norm((raw - center).ravel()))
        minimum = float(rules["zero_edit_relative_tolerance"]) * max(
            float(np.linalg.norm((center - background).ravel())), float(rules["material_scale"]))
        if primary_norm <= minimum:
            final, comp_mask = center.copy(), np.zeros_like(support)
            status = "INVALID_ZERO_EDIT"
            meta.setdefault("invalid_reason", "ZERO_OR_BOX_BLOCKED_PRIMARY_EDIT")
        elif _box_defect(final, rules) > float(rules["feasibility_tolerance"]):
            # Explicit failure rather than projection/jitter hiding a bug.
            status = "INVALID_FEASIBILITY"
        else:
            status = "OK"
        common = _audit(center, background, raw, final, mask, comp_mask, rules,
                        center_provenance, method_id)
        outputs.append(MaterialEdit(method_id, final.ravel(), (final-center).ravel(),
                                    (raw-center).ravel(), (final-raw).ravel(), mask.ravel(),
                                    comp_mask.ravel(), status, {**common, **meta, **comp_meta}))
    if include_contrast:
        raw = background + float(rules["contrast_multiplier"]) * (center - background)
        mask = np.abs(raw - center) > 0
        norm = float(np.linalg.norm((raw - center).ravel()))
        minimum = float(rules["zero_edit_relative_tolerance"]) * max(
            float(np.linalg.norm((center - background).ravel())), float(rules["material_scale"]))
        status = "OK" if norm > minimum else "INVALID_ZERO_EDIT"
        outputs.append(MaterialEdit("CONTRAST_ONLY", raw.ravel(), (raw-center).ravel(),
                                    (raw-center).ravel(), np.zeros(raw.size, np.complex128),
                                    mask.ravel(), np.zeros(raw.size, bool), status,
                                    {**_audit(center, background, raw, raw, mask,
                                              np.zeros_like(mask), rules, center_provenance,
                                              "CONTRAST_ONLY"),
                                     "operation": "FIXED_RELATIVE_CONTRAST_SHRINK",
                                     "contrast_multiplier": rules["contrast_multiplier"],
                                     "usage": "IDENTIFIABILITY_ONLY_CONTRAST_CONTROL"}))
    return outputs


def _audit(center, background, raw, final, mask, comp_mask, rules, role, method_id):
    phase = complex(*rules["phase"])
    reference_norm = float(np.linalg.norm((center - background).ravel()))
    change_norm = float(np.linalg.norm((final - center).ravel()))
    return {"rule_version": RULE_VERSION, "center_provenance": role,
            "online_candidate_eligible": role == "ONLINE_CNN_PREDICTION" and method_id != "CONTRAST_ONLY",
            "diagnostic_only": role != "ONLINE_CNN_PREDICTION" or method_id == "CONTRAST_ONLY",
            "calibration_split": "train", "grid": rules["grid"],
            "coordinates": "physical_absolute_complex_chi_native_C_order",
            "sum_units": "unweighted cell sum; multiply by cell area for physical integral",
            "support_threshold": rules["support_threshold"],
            "material_scale": rules["material_scale"],
            "primary_pixels": int(mask.sum()), "compensation_pixels": int(comp_mask.sum()),
            "primary_change_l2": float(np.linalg.norm((raw - center).ravel())),
            "compensation_change_l2": float(np.linalg.norm((final - raw).ravel())),
            "total_change_l2": change_norm, "center_delta_l2": reference_norm,
            "relative_center_delta_change": change_norm / reference_norm if reference_norm else None,
            "integrated_scalar_before": float(np.sum(np.real((center-background)*phase.conjugate()))),
            "integrated_scalar_after": float(np.sum(np.real((final-background)*phase.conjugate()))),
            "box_feasibility_defect": _box_defect(final, rules),
            "physics_calls": 0, "new_labels": 0, "result_driven_resampling": False}
