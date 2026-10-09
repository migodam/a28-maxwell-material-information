"""A28 E1 saved-evidence tables/figures: no physics, model or gate decision.

Only final files in results/a28 are read.  The frozen cohort is authoritative.
Two known-background conditions are averaged within each parent, then parents
are equally weighted.  Failed/missing/duplicated conditions make the primary
estimate incomplete; no successful-image subset substitutes for it.  Truth is
used only to display the metadata-fixed first parent of each family.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import csv
import json
import math
from pathlib import Path

import numpy as np

from .budget import write_json


FAMILIES = ("shell", "asymmetric_piecewise", "smooth", "mixed_scale")
COMPLEX_FAMILIES = ("shell", "asymmetric_piecewise", "mixed_scale")
ROLES = ("TRUTH_CENTERED_ORACLE", "CNN_CENTERED_DATA_ONLY")
EDIT_IDS = {ROLES[0]: ("CAVITY", "BOUNDARY", "COMPONENT", "CONTRAST_ONLY"),
            ROLES[1]: ("CAVITY", "BOUNDARY", "COMPONENT")}
METHODS = ("L_TV", "F_TV")
DISTRIBUTIONS = ("UNIFORM", "F_LIKELIHOOD", "L_LIKELIHOOD")
HEADROOM_METRICS = ("delta_nrmse", "support_iou", "edge_f1",
                    "full_data_residual_absolute", "cumulative_wall_seconds")
PAIR_METRICS = ("F_white_distance", "L_white_distance", "material_difference_delta_nrmse",
               "support_iou_pair", "edge_f1_pair", "binary_F_validation_error",
               "binary_L_on_F_validation_error", "feedback_secant_fraction")
POSTERIOR_CORE = ("ESS", "max_weight", "risk_mean", "risk_ratio", "risk_excess",
                  "ratio_bias_mass", "kappa_norm_mass")
POSTERIOR_OPTIONAL = ("risk_excess_relative", "forward_commutator_identity",
                       "decision_material_nrmse", "ratio_material_nrmse",
                       "F_vs_L_material_gain", "constitutive_mean_identity",
                       "field_mean_identity", "closed_mean_identity", "data_mean_identity",
                       "ratio_identity_absolute", "risk_excess_identity_absolute")
COMPARISONS = (("F6_vs_L6", ("F_TV", 6), ("L_TV", 6)),
               ("F6_vs_N3", ("F_TV", 6), ("F_TV", 0)),
               ("F6_vs_F1", ("F_TV", 6), ("F_TV", 1)),
               ("F1_vs_N3", ("F_TV", 1), ("F_TV", 0)),
               ("L6_vs_N3", ("L_TV", 6), ("L_TV", 0)))


def _num(value):
    if value is None or value == "":
        return None
    if value in (True, "True", "true"):
        return 1.
    if value in (False, "False", "false"):
        return 0.
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _flag(value):
    if value in (True, 1, "True", "true", "1"):
        return True
    if value in (False, 0, "False", "false", "0"):
        return False
    return None


def _ok(row):
    valid, complete = row.get("valid", True), row.get("complete", True)
    # Union-column CSVs leave optional k0 flags blank; absent flags are not a
    # failure. Explicit false flags still invalidate the record.
    valid = True if valid is None or valid == "" else valid
    complete = True if complete is None or complete == "" else complete
    return (_flag(valid) is True
            and _flag(complete) is True
            and str(row.get("status", "OK")).upper() not in
            {"FAILED", "INVALID", "INCOMPLETE", "MISSING", "NOT_RUN", "BUDGET_STOP"})


def _check(book):
    if book is not None:
        book.check()


def _span(book, name):
    return book.span(name) if book is not None else nullcontext()


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, book=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            if index % 4096 == 0:
                _check(book)
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _normal(row):
    row = dict(row)
    aliases = {"background_id": ("bg", "background"), "outer_attempt": ("k", "iteration"),
               "role": ("bank",),
               "max_weight": ("maximum_weight",),
               "full_data_residual_absolute": ("full_data_absolute_residual",)}
    for key, alternatives in aliases.items():
        if key not in row:
            for alt in alternatives:
                if alt in row:
                    row[key] = row[alt]
                    break
    if "outer_attempt" in row and _num(row["outer_attempt"]) is not None:
        row["outer_attempt"] = int(float(row["outer_attempt"]))
    return row


def frozen_cohort(config):
    """Expected metadata comes only from the pre-run frozen cohort, never results."""
    cohort = [_normal(row) for row in config.get("cohort", ())]
    if not cohort:
        raise ValueError("A28_ARTIFACTS_REQUIRE_FROZEN_COHORT")
    result, seen = [], set()
    for row in cohort:
        if not {"parent_id", "family", "background_id"} <= set(row):
            raise ValueError("A28_FROZEN_COHORT_METADATA_REQUIRED")
        parent = _num(row["parent_id"])
        if parent is None or int(parent) != parent:
            raise ValueError("A28_FROZEN_PARENT_INTEGER_REQUIRED")
        row["parent_id"] = int(parent)
        if "case_id" in row and (not str(row["case_id"]) or Path(str(row["case_id"])).name != str(row["case_id"])
                                  or any(c in str(row["case_id"]) for c in "/\\\0")):
            raise ValueError("A28_FROZEN_CASE_ID_MUST_BE_SIMPLE_NAME")
        key = (row["parent_id"], str(row["background_id"]))
        if key in seen:
            raise ValueError("A28_FROZEN_COHORT_DUPLICATED_CONDITION")
        seen.add(key)
        result.append(row)
    family_by_parent = {}
    backgrounds = tuple(config.get("backgrounds", ("weak", "ring")))
    for row in result:
        old = family_by_parent.setdefault(row["parent_id"], str(row["family"]))
        if old != row["family"] or row["family"] not in FAMILIES or row["background_id"] not in backgrounds:
            raise ValueError("A28_FROZEN_COHORT_FAMILY_OR_BACKGROUND_CONFLICT")
    for parent in family_by_parent:
        if {r["background_id"] for r in result if r["parent_id"] == parent} != set(backgrounds):
            raise ValueError("A28_FROZEN_COHORT_MISSING_BACKGROUND")
    return sorted(result, key=lambda r: (r["parent_id"], backgrounds.index(r["background_id"])))


def _metadata_matches(row, expected):
    if (_num(row.get("parent_id")) != expected["parent_id"]
            or str(row.get("family")) != expected["family"]
            or str(row.get("background_id")) != expected["background_id"]):
        return False
    for key in ("case_id",):
        if key in row and key in expected and str(row[key]) != str(expected[key]):
            return False
    for key in ("amplitude", "sigma"):
        if key in row and key in expected:
            value, fixed = _num(row[key]), _num(expected[key])
            if value is None or fixed is None or not math.isclose(value, fixed, rel_tol=1e-12, abs_tol=0):
                return False
    return True


def parent_table(raw_rows, cohort, axes, metric_names, *, required_metrics=None):
    """Strict two-background parent means for every preregistered arm/iteration.

    ``axes`` is a list of expected grouping dictionaries. Empty/invalid cells
    are generated explicitly. Rows outside the expected cohort/axes are not
    silently selected away: they mark a protocol conflict and null the table.
    """
    axes = list(axes)
    required_metrics = tuple(metric_names if required_metrics is None else required_metrics)
    if not axes:
        return [], []
    axis_names = tuple(axes[0])
    if any(tuple(axis) != axis_names for axis in axes):
        raise ValueError("A28_IDENTICAL_GROUPING_AXES_REQUIRED")
    axis_keys = {tuple(axis[name] for name in axis_names) for axis in axes}
    expected = {(row["parent_id"], row["background_id"]): row for row in cohort}
    indexed, unexpected = defaultdict(list), []
    for original in raw_rows:
        row = _normal(original)
        parent = _num(row.get("parent_id"))
        parent = int(parent) if parent is not None and int(parent) == parent else None
        key = (parent, str(row.get("background_id", "")))
        group = tuple(row.get(name) for name in axis_names)
        if key not in expected or group not in axis_keys:
            unexpected.append(row)
        else:
            indexed[group + key].append(row)
    parents = sorted({row["parent_id"] for row in cohort})
    out = []
    for axis in axes:
        group = tuple(axis[name] for name in axis_names)
        for parent in parents:
            conditions = [row for row in cohort if row["parent_id"] == parent]
            failures, local = [], []
            for expected_row in conditions:
                found = indexed.get(group + (parent, expected_row["background_id"]), [])
                reason = ("MISSING" if not found else "DUPLICATED" if len(found) > 1 else
                          "METADATA_MISMATCH" if not _metadata_matches(found[0], expected_row) else
                          "FAILED_OR_INVALID" if not _ok(found[0]) else None)
                if reason:
                    failures.append({"background_id": expected_row["background_id"], "reason": reason,
                                     "errors": [r.get("error", r.get("edit_status")) for r in found]})
                else:
                    local.append(found[0])
            numeric = {metric: [_num(row.get(metric)) for row in local] for metric in metric_names}
            undefined = [metric for metric, values in numeric.items() if len(values) != len(conditions) or any(v is None for v in values)]
            complete = not failures and not any(key in undefined for key in required_metrics) and not unexpected
            row = {**axis, "parent_id": parent, "family": conditions[0]["family"],
                   "aggregation": "equal_conditions_within_parent", "expected_conditions": len(conditions),
                   "valid_conditions": len(local), "complete": complete,
                   "status": "COMPLETE" if complete else "INCOMPLETE",
                   "failures": failures, "undefined_metrics": undefined,
                   "unexpected_input_rows": len(unexpected)}
            row.update({metric: float(np.mean(values)) if complete and metric not in undefined else None for metric, values in numeric.items()})
            if "outer_attempt" in axis:
                row.update(accepted_conditions=sum(_flag(r.get("accepted")) is True for r in local),
                           unknown_acceptance_conditions=sum(_flag(r.get("accepted")) is None for r in local),
                           rejected_conditions=sum(_flag(r.get("accepted")) is False for r in local))
            out.append(row)
    return out, unexpected


def parent_groups(cohort):
    family = {row["parent_id"]: row["family"] for row in cohort}
    groups = [("ALL_FAMILIES", sorted(family)),
              ("COMPLEX_FAMILIES", sorted(p for p in family if family[p] in COMPLEX_FAMILIES))]
    groups.extend((name, sorted(p for p in family if family[p] == name)) for name in FAMILIES)
    return [(name, parents) for name, parents in groups if parents]


def group_table(parent_rows, cohort, axis_names, metrics):
    indexed = defaultdict(dict)
    for row in parent_rows:
        indexed[tuple(row[name] for name in axis_names)][int(row["parent_id"])] = row
    output = []
    for group, by_parent in indexed.items():
        for name, parents in parent_groups(cohort):
            local = [by_parent.get(parent) for parent in parents]
            complete = all(row is not None and row["complete"] for row in local)
            result = {**dict(zip(axis_names, group)), "group": name, "aggregation": "equal_parent",
                      "expected_parents": len(parents),
                      "complete_parents": sum(row is not None and row["complete"] for row in local),
                      "complete": complete, "status": "COMPLETE" if complete else "INCOMPLETE"}
            undefined = [metric for metric in metrics if not complete or any(_num(row.get(metric)) is None for row in local)]
            result.update({metric: float(np.mean([row[metric] for row in local])) if complete and metric not in undefined else None for metric in metrics})
            result["undefined_metrics"] = undefined
            output.append(result)
    return output


def paired_headroom(parent_rows, cohort, config, book=None):
    """F6 vs L6/N3/F1; fixed 2000 paired bootstrap over independent parents."""
    samples = int(config.get("bootstrap_samples", 2000))
    if samples != 2000:
        raise ValueError("A28_FROZEN_2000_PARENT_BOOTSTRAP_REQUIRED")
    seed = int(config.get("bootstrap_seed", config.get("primary_seed", 281010)))
    indexed = {(row["method"], int(row["outer_attempt"]), int(row["parent_id"])): row for row in parent_rows}
    scalar, per_parent = [], []
    for comparison_index, (name, candidate, control) in enumerate(COMPARISONS):
        for group_index, (group, parents) in enumerate(parent_groups(cohort)):
            for metric_index, metric in enumerate(HEADROOM_METRICS[:-1]):
                higher = metric in {"support_iou", "edge_f1"}
                values, failures = [], []
                for parent in parents:
                    ca, co = indexed.get(candidate+(parent,)), indexed.get(control+(parent,))
                    av, bv = (_num(ca.get(metric)) if ca else None), (_num(co.get(metric)) if co else None)
                    reason = ("MISSING_OR_FAILED_PARENT" if ca is None or co is None or not ca["complete"] or not co["complete"] else
                              "UNDEFINED_METRIC" if av is None or bv is None else
                              "ZERO_OR_NEGATIVE_CONTROL_DENOMINATOR" if not higher and bv <= 0 else None)
                    gain = (av-bv if higher else (bv-av)/bv) if reason is None else None
                    if reason:
                        failures.append({"parent_id": parent, "reason": reason})
                    else:
                        values.append(gain)
                    per_parent.append({"comparison": name, "group": group, "metric": metric,
                                       "parent_id": parent, "family": next(r["family"] for r in cohort if r["parent_id"] == parent),
                                       "candidate_value": av, "control_value": bv, "gain": gain,
                                       "complete": reason is None, "reason": reason,
                                       "gain_definition": "candidate-control absolute" if higher else "(control-candidate)/control"})
                complete = bool(parents) and not failures
                point = low = high = None
                if complete:
                    _check(book)
                    values = np.asarray(values, dtype=float)
                    rng = np.random.default_rng(seed+10000*comparison_index+100*group_index+metric_index)
                    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
                    point = float(values.mean())
                    low, high = map(float, np.quantile(draws, [.025, .975]))
                scalar.append({"comparison": name, "group": group, "metric": metric,
                               "expected_parents": len(parents), "paired_parents": len(values),
                               "complete": complete, "status": "COMPLETE" if complete else "INCOMPLETE",
                               "mean_gain": point, "ci95_low": low, "ci95_high": high,
                               "bootstrap_samples": samples, "seed": seed+10000*comparison_index+100*group_index+metric_index,
                               "unit": "independent_exposed_development_parent", "failed_parents": failures,
                               "gain_definition": "candidate-control absolute" if higher else "(control-candidate)/control",
                               "small_cluster_note": "one-parent empirical interval is degenerate" if len(parents) == 1 else None})
    return {"summary_rows": scalar, "per_parent_rows": per_parent,
            "unit": "independent_parent; average two backgrounds before relative gain",
            "scientific_verdict_performed": False, "holdout": False,
            "exposure": "EXPOSED_DEVELOPMENT", "samples": samples}


def paired_bank_decisions(parent_rows,cohort,config,book=None):
    """Finite-bank material decisions, separated by oracle/data-only bank.

    These are exploratory finite-candidate diagnostics. The truth-centered
    bank cannot establish deployable algorithm headroom; no gate is assigned.
    """
    if int(config.get("bootstrap_samples",2000))!=2000:
        raise ValueError("A28_FROZEN_2000_PARENT_BOOTSTRAP_REQUIRED")
    seed=int(config.get("bootstrap_seed",config.get("primary_seed",281010)))+900000
    indexed={(row["role"],int(row["parent_id"])):row for row in parent_rows}
    contrasts=(("F_mean_vs_L_mean","F_mean_error","L_mean_error"),
               ("F_mean_vs_N3","F_mean_error","N3_error"))
    out=[]
    for role_index,role in enumerate(ROLES):
        for comparison_index,(comparison,candidate,control) in enumerate(contrasts):
            for group_index,(group,parents) in enumerate(parent_groups(cohort)):
                gains,failures=[],[]
                for parent in parents:
                    row=indexed.get((role,parent))
                    av,bv=(_num(row.get(candidate)) if row else None),(_num(row.get(control)) if row else None)
                    reason=("MISSING_OR_FAILED_PARENT" if row is None or not row["complete"] else
                            "UNDEFINED_OR_ZERO_CONTROL" if av is None or bv is None or bv<=0 else None)
                    if reason:failures.append({"parent_id":parent,"reason":reason})
                    else:gains.append((bv-av)/bv)
                complete=bool(parents) and not failures;point=low=high=None
                local_seed=seed+10000*role_index+100*comparison_index+group_index
                if complete:
                    _check(book);gains=np.asarray(gains,float)
                    rng=np.random.default_rng(local_seed)
                    sampled=gains[rng.integers(0,len(gains),size=(2000,len(gains)))].mean(axis=1)
                    point=float(gains.mean());low,high=map(float,np.quantile(sampled,[.025,.975]))
                out.append({"role":role,"comparison":comparison,"group":group,"mean_gain":point,
                            "ci95_low":low,"ci95_high":high,"complete":complete,
                            "expected_parents":len(parents),"paired_parents":len(gains),"failed_parents":failures,
                            "bootstrap_samples":2000,"seed":local_seed,"unit":"independent_exposed_development_parent",
                            "gain_definition":"average two background errors, then (control-candidate)/control",
                            "scope":"OFFLINE_ORACLE_ONLY" if role==ROLES[0] else "FINITE_DATA_DEPENDENT_CANDIDATE_BANK",
                            "scientific_verdict_performed":False})
    return out


def _figure(fig, name, raw_rows, out, manifest, book, *, note, source_status="AVAILABLE"):
    _check(book)
    png, svg, raw = out/"figures"/(name+".png"), out/"figures"/(name+".svg"), out/"figures"/(name+".csv")
    png.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(raw, raw_rows, book)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    manifest.append({"name": name, "status": source_status, "png": str(png.relative_to(out)),
                     "svg": str(svg.relative_to(out)), "raw_csv": str(raw.relative_to(out)),
                     "raw_rows": len(raw_rows), "note": note, "synthetic_science": False})


def _empty_figure(name, manifest, reason):
    manifest.append({"name": name, "status": "NOT_RUN", "reason": reason,
                     "png": None, "svg": None, "raw_csv": None, "synthetic_science": False})


def _draw_distances(rows, config, out, manifest, book):
    if not rows:
        _empty_figure("counterfactual_distances", manifest, "FINAL_PAIR_METRICS_MISSING")
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    structural = float(config.get("gate_structural_material_delta_min", .1))
    for ri, role in enumerate(ROLES):
        selected = [row for row in rows if row.get("role") == role]
        valid = [row for row in selected if _ok(row)]
        for method, color, marker in (("F", "#326c96", "o"), ("L", "#cc7c33", "x")):
            points = [(r, _num(r.get(method+"_white_distance"))) for r in valid]
            for ci, key in enumerate(("material_difference_delta_nrmse", "support_iou_pair")):
                pairs = [(_num(r.get(key)), y) for r,y in points if _num(r.get(key)) is not None and y is not None]
                axes[ri,ci].scatter([x if ci == 0 else 1-x for x,y in pairs], [y for x,y in pairs],
                                     alpha=.6, s=18, label=method, color=color, marker=marker)
            error_key = "binary_F_validation_error" if method == "F" else "binary_L_on_F_validation_error"
            pairs = [(_num(r.get("F_white_distance")), _num(r.get(error_key))) for r in valid]
            pairs = [(x,y) for x,y in pairs if x is not None and y is not None]
            axes[ri,2].scatter([x for x,y in pairs], [y for x,y in pairs], alpha=.6,s=18,
                              label="F decision" if method == "F" else "L decision on F data", color=color,marker=marker)
        for ci in (0,1):
            axes[ri,ci].axhline(1.,color="gray",ls=":",lw=1)
            axes[ri,ci].axhline(3.,color="gray",ls="--",lw=1)
            axes[ri,ci].set_ylabel("Candidate whitened distance")
            axes[ri,ci].legend(fontsize=8)
        axes[ri,0].axvline(structural,color="gray",ls=":",lw=1)
        axes[ri,0].set_xlabel("Material difference / center delta norm")
        axes[ri,1].set_xlabel("1 - support IoU of candidate pair")
        axes[ri,2].set_xlabel("F whitened candidate distance")
        axes[ri,2].set_ylabel("Independent-noise binary error")
        axes[ri,2].legend(fontsize=8)
        axes[ri,0].set_title(f"{role}\nvalid {len(valid)}/{len(selected)}; invalid retained in raw CSV", fontsize=9)
    fig.suptitle("Fixed structural edits; exposed development; truth-centered bank is OFFLINE ONLY", fontsize=11)
    _figure(fig,"counterfactual_distances",rows,out,manifest,book,
            note="All rows preserved. Scatter shows defined valid pairs with denominator/count annotations; not global identifiability.")


def _draw_headroom(groups, paired, out, manifest, book):
    if not groups:
        _empty_figure("headroom_quality_cost",manifest,"FINAL_PER_ITERATION_MISSING")
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,4.4),constrained_layout=True)
    subset = [r for r in paired["summary_rows"] if r["group"] == "COMPLEX_FAMILIES" and r["metric"] == "delta_nrmse"
              and r["comparison"] in {"F6_vs_L6","F6_vs_N3","F6_vs_F1"}]
    for index,row in enumerate(subset):
        if row["complete"]:
            mean,lo,hi = [100*row[key] for key in ("mean_gain","ci95_low","ci95_high")]
            axes[0].vlines(index,lo,hi,color="#326c96")
            axes[0].plot(index,mean,"o",color="#326c96")
            axes[0].hlines([lo,hi],index-.06,index+.06,color="#326c96")
        else:
            axes[0].text(index,0,f"INCOMPLETE\n{row['paired_parents']}/{row['expected_parents']}",ha="center",fontsize=8,color="firebrick")
    axes[0].axhline(0,color="gray",lw=1)
    axes[0].set_xticks(range(len(subset)),[r["comparison"].replace("_vs_"," / ") for r in subset],fontsize=9)
    axes[0].set_ylabel("Paired material gain (%)")
    axes[0].set_title("Complex families; 2000 parent resamples")
    curve_rows = []
    for method,color in (("L_TV","#cc7c33"),("F_TV","#326c96")):
        local = sorted([r for r in groups if r["group"] == "COMPLEX_FAMILIES" and r["method"] == method],key=lambda r:r["outer_attempt"])
        curve_rows.extend(local)
        x = [r["outer_attempt"] for r in local]
        y = [r["delta_nrmse"] if r["complete"] else np.nan for r in local]
        t = [r["cumulative_wall_seconds"] if r["complete"] else np.nan for r in local]
        axes[1].plot(x,y,"o-",color=color,label=method)
        axes[2].plot(t,y,"o-",color=color,label=method)
        missing = [r["outer_attempt"] for r in local if not r["complete"]]
        if missing:
            axes[1].text(.02,.98 if method == "L_TV" else .90,f"{method} missing k={missing}",transform=axes[1].transAxes,va="top",fontsize=8)
    axes[1].set_xlabel("Outer attempts (rejected steps retained)")
    axes[1].set_ylabel("Equal-parent delta-chi NRMSE")
    axes[2].set_xlabel("Cumulative update wall seconds\nsetup/shared initial cache excluded")
    axes[2].set_ylabel("Equal-parent delta-chi NRMSE")
    axes[1].legend();axes[2].legend()
    fig.suptitle("N3-start, matched prior/query budget; update-only costs are not deployment time",fontsize=11)
    _figure(fig,"headroom_quality_cost",[{"record_type":"paired_gain",**r} for r in subset]
            +[{"record_type":"trajectory_mean",**r} for r in curve_rows],out,manifest,book,
            note="Means only with complete frozen cohort; all missing/failing parents null primary statistics. Two backgrounds averaged before parent gain.")


def _draw_posterior(rows,out,manifest,book):
    if not rows:
        _empty_figure("kappa_conditional_risk",manifest,"FINAL_PER_DISTRIBUTION_MISSING")
        return
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.6),constrained_layout=True)
    colors={"UNIFORM":"#7a7a7a","F_LIKELIHOOD":"#326c96","L_LIKELIHOOD":"#cc7c33"}
    for ri,role in enumerate(ROLES):
        selected=[r for r in rows if r.get("role")==role]
        valid=[r for r in selected if _ok(r)]
        for dist,color in colors.items():
            local=[r for r in valid if r.get("distribution")==dist]
            pairs=[(_num(r.get("ratio_bias_mass")),_num(r.get("risk_excess"))) for r in local]
            pairs=[(x*x,y) for x,y in pairs if x is not None and y is not None]
            axes[ri,0].scatter([x for x,y in pairs],[y for x,y in pairs],s=20,alpha=.65,label=dist,color=color)
            pairs=[(_num(r.get("distinct_material_ESS",r.get("ESS"))),_num(r.get("risk_excess_relative"))) for r in local]
            pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
            axes[ri,1].scatter([x for x,y in pairs],[y for x,y in pairs],s=20,alpha=.65,label=dist,color=color)
        vals=[_num(r.get("risk_excess")) for r in valid]+[_num(r.get("ratio_bias_mass"))**2 for r in valid if _num(r.get("ratio_bias_mass")) is not None]
        vals=[v for v in vals if v is not None]
        limit=max(vals) if vals and max(vals)>0 else 1.
        axes[ri,0].plot([0,limit],[0,limit],":",color="black",lw=1,label="identity y=x")
        axes[ri,0].set_xlabel("Material mass ratio-bias norm squared")
        axes[ri,0].set_ylabel("Conditional risk excess: ratio minus mean")
        axes[ri,1].set_xlabel("Distinct-material effective sample size")
        axes[ri,1].set_ylabel("Risk excess / conditional mean risk\nundefined tiny-risk ratios remain absent")
        axes[ri,0].set_title(f"{role}\nvalid {len(valid)}/{len(selected)}",fontsize=9)
        for axis in axes[ri]:axis.legend(fontsize=7)
    fig.suptitle("Finite declared diagnostic distributions; covariance identity is not calibrated posterior evidence",fontsize=11)
    _figure(fig,"kappa_conditional_risk",rows,out,manifest,book,
            note="Zero/collapsed distributions retained in raw rows; undefined risk ratios remain undefined, no plotting floor.")


def _bp_spectra(folder,book):
    rows=[]
    for path in sorted(folder.glob("*.npz")):
        _check(book)
        if path.is_symlink():raise ValueError("A28_BP_SPECTRAL_SYMLINK_FORBIDDEN")
        with np.load(path,allow_pickle=False) as archive:
            if "BP_singular" not in archive.files:continue
            values=np.asarray(archive["BP_singular"],float)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():raise ValueError("A28_INVALID_BP_SPECTRUM")
            if "background_id" in archive.files:
                background=str(archive["background_id"])
            else:
                background=next((bg for bg in ("weak","ring") if bg in path.stem.lower()),path.stem)
            shape=np.asarray(archive["BP_matrix"]).shape if "BP_matrix" in archive.files else None
            relative=values/values[0] if values[0]>0 else np.zeros_like(values)
            rows.extend({"background_id":background,"source_file":path.name,"index":i+1,
                         "singular_value":float(v),"relative_singular_value":float(relative[i]),
                         "matrix_rows":shape[0] if shape else None,"matrix_columns":shape[1] if shape else None}
                        for i,v in enumerate(values))
    return rows


def _draw_bp(rows,spectra,out,manifest,book):
    if not rows and not spectra:
        _empty_figure("bp_information_preservation",manifest,"FINAL_BP_INFORMATION_AND_SPECTRUM_MISSING")
        return
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for bg in dict.fromkeys(r["background_id"] for r in spectra):
        local=[r for r in spectra if r["background_id"]==bg and r["relative_singular_value"]>0]
        axes[0].semilogy([r["index"] for r in local],[r["relative_singular_value"] for r in local],label=bg)
    if spectra:
        for cutoff in (1e-12,1e-9,1e-6):axes[0].axhline(cutoff,color="gray",ls=":",lw=.8)
        axes[0].legend(fontsize=8)
    else:axes[0].text(.5,.5,"Full saved BP spectrum NOT_RUN",ha="center",transform=axes[0].transAxes)
    axes[0].set_xlabel("Singular value index")
    axes[0].set_ylabel("BP singular value / maximum")
    axes[0].set_title("Numerical rank is not formal injectivity")
    for cutoff in ("machine","1e-12","1e-9","1e-6"):
        selected=[r for r in rows if str(r.get("cutoff_id"))==cutoff and _ok(r)]
        points=[(_num(r.get("data_whitened_distance")),_num(r.get("BP_induced_covariance_whitened_distance"))) for r in selected]
        points=[(x,y) for x,y in points if x is not None and y is not None]
        axes[1].scatter([x for x,y in points],[y for x,y in points],alpha=.45,s=15,label=cutoff)
    values=[_num(r.get("data_whitened_distance")) for r in rows]
    values=[v for v in values if v is not None]
    limit=max(values) if values and max(values)>0 else 1.
    axes[1].plot([0,limit],[0,limit],":",color="black",lw=1)
    axes[1].set_xlabel("Raw-data whitened pair distance")
    axes[1].set_ylabel("BP induced-covariance pair distance")
    axes[1].set_title("OFFLINE statistical covariance inverse only")
    axes[1].legend(fontsize=8)
    fig.suptitle("Actual unboxed N3 BP channel; FP32 reconstruction and all undefined rows saved separately",fontsize=11)
    _figure(fig,"bp_information_preservation",[{"record_type":"spectrum",**r} for r in spectra]
            +[{"record_type":"pair",**r} for r in rows],out,manifest,book,
            note="Neither Euclidean BP norm nor finite SVD establishes whole-encoder information sufficiency; other 12 channels are nonlinear.")


def _fixed_images(base,cohort,config,out,manifest,book):
    trajectory_path=base/"headroom/TRAJECTORIES.npz"
    if trajectory_path.is_symlink() or not trajectory_path.resolve().is_relative_to(base.resolve()):
        raise ValueError("A28_FIXED_TRAJECTORY_SOURCE_FORBIDDEN")
    if not trajectory_path.is_file():
        _empty_figure("fixed_material_real",manifest,"FINAL_TRAJECTORIES_MISSING")
        _empty_figure("fixed_material_imag",manifest,"FINAL_TRAJECTORIES_MISSING")
        return []
    selected=[]
    for family in FAMILIES:
        parents=sorted({r["parent_id"] for r in cohort if r["family"]==family})
        if not parents:continue
        for bg in config.get("backgrounds",("weak","ring")):
            selected.extend(r for r in cohort if r["parent_id"]==parents[0] and r["background_id"]==bg)
    grid=int(config.get("grid",24));records=[];panels=[]
    with np.load(trajectory_path,allow_pickle=False) as trajectories:
        for meta in selected:
            _check(book)
            case=str(meta.get("case_id",""));path=base/"anatomy"/(case+".npz")
            if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
                raise ValueError("A28_FIXED_IMAGE_SOURCE_FORBIDDEN")
            row={"meta":meta,"images":{},"status":{}}
            if path.is_file():
                with np.load(path,allow_pickle=False) as archive:
                    for key in ("truth","background",ROLES[1]+"_chi"):
                        if key not in archive.files:raise ValueError("A28_FIXED_IMAGE_REQUIRED_ARRAY_MISSING:"+key)
                    truth=np.asarray(archive["truth"],np.complex128).reshape(-1)
                    background=np.asarray(archive["background"],np.complex128).reshape(-1)
                    center=np.asarray(archive[ROLES[1]+"_chi"][0],np.complex128).reshape(-1)
                if any(v.shape!=(grid*grid,) or not np.isfinite(v).all() for v in (truth,background,center)):
                    raise ValueError("A28_FIXED_IMAGE_LAYOUT_MISMATCH")
                row["images"].update(Truth=truth-background,N3=center-background)
                row["status"].update(Truth="AVAILABLE",N3="AVAILABLE")
                for name,method in (("L6","L_TV"),("F6","F_TV")):
                    key=case+"_"+method
                    if key not in trajectories.files or len(trajectories[key])<7:
                        row["status"][name]="MISSING_K6; not substituted with last available step"
                    else:
                        value=np.asarray(trajectories[key][6],np.complex128).reshape(-1)
                        if value.shape!=(grid*grid,) or not np.isfinite(value).all():raise ValueError("A28_FIXED_TRAJECTORY_LAYOUT_MISMATCH")
                        row["images"][name]=value-background;row["status"][name]="AVAILABLE"
            else:
                row["status"]={name:"ANATOMY_FILE_MISSING" for name in ("Truth","N3","L6","F6")}
            for name in ("Truth","N3","L6","F6"):
                value=row["images"].get(name)
                if value is None:
                    records.append({**meta,"image":name,"status":row["status"][name],"pixel_row":None,"pixel_col":None,"delta_real":None,"delta_imag":None})
                else:
                    records.extend({**meta,"image":name,"status":"AVAILABLE","pixel_row":pixel//grid,"pixel_col":pixel%grid,
                                    "delta_real":float(v.real),"delta_imag":float(v.imag)} for pixel,v in enumerate(value))
            panels.append(row)
    if not panels:
        _empty_figure("fixed_material_real",manifest,"FROZEN_METADATA_HAS_NO_FAMILIES")
        _empty_figure("fixed_material_imag",manifest,"FROZEN_METADATA_HAS_NO_FAMILIES")
        return records
    import matplotlib.pyplot as plt
    for component in ("real","imag"):
        fig,axes=plt.subplots(len(panels),4,figsize=(11.2,2.45*len(panels)),squeeze=False,constrained_layout=True)
        for ri,panel in enumerate(panels):
            meta=panel["meta"]
            values=[getattr(v,component) for v in panel["images"].values()]
            vmin=min(float(v.min()) for v in values) if values else 0.
            vmax=max(float(v.max()) for v in values) if values else 1.
            if vmin==vmax:vmax=vmin+1.
            image=None
            for ci,name in enumerate(("Truth","N3","L6","F6")):
                axis=axes[ri,ci];axis.set_xticks([]);axis.set_yticks([])
                if ri==0:axis.set_title(name)
                if ci==0:axis.set_ylabel(f"{meta['family']}\np{meta['parent_id']} / {meta['background_id']}",fontsize=9)
                if name in panel["images"]:
                    image=axis.imshow(getattr(panel["images"][name],component).reshape(grid,grid),origin="lower",cmap="viridis",vmin=vmin,vmax=vmax,interpolation="nearest")
                else:
                    axis.text(.5,.5,"NOT_RUN / FAILED\n"+panel["status"][name],transform=axis.transAxes,ha="center",va="center",fontsize=7,color="firebrick")
            if image is not None:fig.colorbar(image,ax=axes[ri,:],shrink=.82,pad=.01)
        fig.suptitle(f"Actual delta material ({component}); first frozen parent per family, both backgrounds\nShared color scale within row; rejected updates remain; OFFLINE truth display",fontsize=11)
        _figure(fig,"fixed_material_"+component,records,out,manifest,book,
                note="Metadata-fixed objects, not best-case selection; delta material with known background subtracted. Missing k6 is never replaced.")
    return records


def report_artifacts(root,config,book=None):
    """Build artifacts exclusively from final saved A28 evidence, without verdict."""
    root=Path(root).resolve();base=root/"results/a28";out=base/"artifacts";out.mkdir(parents=True,exist_ok=True)
    cohort=frozen_cohort(config);manifest=[]
    sources={"anatomy":base/"anatomy/PAIR_METRICS.csv","baseline":base/"anatomy/BASELINE_DIAGNOSTICS.csv",
             "headroom":base/"headroom/PER_ITERATION.csv","posterior":base/"posterior/PER_DISTRIBUTION.csv",
             "decisions":base/"posterior/DECISIONS.csv","bp":base/"bp/PAIR_INFORMATION.csv",
             "cnn_counterfactual":base/"bp/CNN_COUNTERFACTUAL_RESPONSE.csv"}
    for key,path in sources.items():
        if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError("A28_ARTIFACT_SOURCE_OUTSIDE_ROOT")
    data={key:[_normal(row) for row in _read_csv(path)] for key,path in sources.items()}
    for key,path in sources.items():
        _write_csv(out/"raw"/path.name,data[key],book)
    with _span(book,"a28_saved_evidence_parent_aggregation"):
        head_axes=[{"method":method,"outer_attempt":k} for method in METHODS for k in range(7)]
        head_optional=tuple(key for key in ("full_nrmse","real_nrmse","imag_nrmse","ssim_delta",
                                            "passive","feasible","full_data_relative_residual",
                                            "full_data_normalization_denominator","full_backward_residual")
                            if any(key in row for row in data["headroom"]))
        head_metrics=HEADROOM_METRICS+head_optional
        head_parents,head_unexpected=parent_table(data["headroom"],cohort,head_axes,head_metrics,required_metrics=HEADROOM_METRICS)
        head_groups=group_table(head_parents,cohort,("method","outer_attempt"),head_metrics)
        paired=paired_headroom(head_parents,cohort,config,book)
        _write_csv(out/"HEADROOM_PER_PARENT.csv",head_parents,book)
        _write_csv(out/"HEADROOM_GROUPS.csv",head_groups,book)
        _write_csv(out/"HEADROOM_PAIRED_STATISTICS.csv",paired["summary_rows"],book)
        _write_csv(out/"HEADROOM_PAIRED_PER_PARENT.csv",paired["per_parent_rows"],book)
        write_json(out/"HEADROOM_PAIRED_STATISTICS.json",paired)
        pair_axes=[{"role":role,"edit":edit} for role in ROLES for edit in EDIT_IDS[role]]
        pair_parents,pair_unexpected=parent_table(data["anatomy"],cohort,pair_axes,PAIR_METRICS,
                                                  required_metrics=PAIR_METRICS[:-1])
        pair_groups=group_table(pair_parents,cohort,("role","edit"),PAIR_METRICS)
        _write_csv(out/"COUNTERFACTUAL_PER_PARENT.csv",pair_parents,book)
        _write_csv(out/"COUNTERFACTUAL_GROUPS.csv",pair_groups,book)
        post_extra=tuple(dict.fromkeys(key for row in data["posterior"] for key in row
                                      if (key.startswith("mean_feedback_") or key in
                                          {"mean_full_data_residual","ratio_full_data_residual","forward_commutator_absolute","forward_commutator_white",
                                           "ratio_bias_delta_scale","N3_material_nrmse","ratio_feasible",
                                           "distinct_material_count","duplicate_label_count","distinct_material_ESS",
                                           "distinct_material_entropy","distinct_material_maximum_weight"})
                                      and any(_num(item.get(key)) is not None for item in data["posterior"])))
        post_metrics=POSTERIOR_CORE+tuple(key for key in POSTERIOR_OPTIONAL if any(key in row for row in data["posterior"]))+post_extra
        post_axes=[{"role":role,"distribution":dist} for role in ROLES for dist in DISTRIBUTIONS]
        post_parents,post_unexpected=parent_table(data["posterior"],cohort,post_axes,post_metrics,required_metrics=POSTERIOR_CORE)
        post_groups=group_table(post_parents,cohort,("role","distribution"),post_metrics)
        _write_csv(out/"POSTERIOR_PER_PARENT.csv",post_parents,book)
        _write_csv(out/"POSTERIOR_GROUPS.csv",post_groups,book)
        decision_metrics=("F_mean_error","L_mean_error","N3_error")
        decision_extra=tuple(key for key in ("F_mean_risk_under_F","L_mean_risk_under_F","F_MAP_material_error","L_MAP_material_error","F_ESS","L_ESS")
                             if any(key in row for row in data["decisions"]))
        decision_parents,decision_unexpected=parent_table(data["decisions"],cohort,[{"role":role} for role in ROLES],
                                                         decision_metrics+decision_extra,required_metrics=decision_metrics)
        _write_csv(out/"POSTERIOR_DECISIONS_PER_PARENT.csv",decision_parents,book)
        _write_csv(out/"POSTERIOR_DECISIONS_GROUPS.csv",group_table(decision_parents,cohort,("role",),decision_metrics+decision_extra),book)
        bank_paired=paired_bank_decisions(decision_parents,cohort,config,book)
        _write_csv(out/"POSTERIOR_DECISION_PAIRED.csv",bank_paired,book)
        write_json(out/"POSTERIOR_DECISION_PAIRED.json",{"rows":bank_paired,"scientific_verdict_performed":False})
        cnn_metrics=("candidate_a_CNN_material_nrmse","candidate_b_CNN_material_nrmse",
                     "CNN_material_change_over_true_change","CNN_nearest_pair_correct_a","CNN_nearest_pair_correct_b")
        cnn_optional=("CNN_change_material_alignment",) if any("CNN_change_material_alignment" in r for r in data["cnn_counterfactual"]) else ()
        cnn_parents,cnn_unexpected=parent_table(data["cnn_counterfactual"],cohort,pair_axes,cnn_metrics+cnn_optional,required_metrics=cnn_metrics)
        _write_csv(out/"CNN_COUNTERFACTUAL_PER_PARENT.csv",cnn_parents,book)
        _write_csv(out/"CNN_COUNTERFACTUAL_GROUPS.csv",group_table(cnn_parents,cohort,("role","edit"),cnn_metrics+cnn_optional),book)
        _write_csv(out/"UNEXPECTED_INPUT_ROWS.csv",[{"source":"headroom",**r} for r in head_unexpected]
                   +[{"source":"anatomy",**r} for r in pair_unexpected]+[{"source":"posterior",**r} for r in post_unexpected]
                   +[{"source":"decisions",**r} for r in decision_unexpected]+[{"source":"cnn_counterfactual",**r} for r in cnn_unexpected],book)
        baseline_metrics=tuple(key for key in data["baseline"][0] if key not in
            {"case_id","parent_id","family","background_id","amplitude","sigma","valid","status","error"}
            and any(_num(row.get(key)) is not None for row in data["baseline"])) if data["baseline"] else ()
        if baseline_metrics:
            baseline_parents,baseline_unexpected=parent_table(data["baseline"],cohort,[{}],baseline_metrics)
            _write_csv(out/"BASELINE_PER_PARENT.csv",baseline_parents,book)
            _write_csv(out/"BASELINE_GROUPS.csv",group_table(baseline_parents,cohort,(),baseline_metrics),book)
        else:baseline_unexpected=[]
    with _span(book,"a28_saved_evidence_figure_render"):
        import matplotlib
        matplotlib.use("Agg",force=True)
        _draw_distances(data["anatomy"],config,out,manifest,book)
        if data["headroom"]:
            _draw_headroom(head_groups,paired,out,manifest,book)
        else:
            _empty_figure("headroom_quality_cost",manifest,"FINAL_PER_ITERATION_MISSING")
        _draw_posterior(data["posterior"],out,manifest,book)
        spectra=_bp_spectra(base/"bp",book)
        _draw_bp(data["bp"],spectra,out,manifest,book)
        _write_csv(out/"BP_SPECTRUM.csv",spectra,book)
        _fixed_images(base,cohort,config,out,manifest,book)
    summary={"status":"ARTIFACTS_WRITTEN","scientific_verdict_performed":False,
             "scope":"EXPLORATORY_EXPOSED_DEVELOPMENT; finite candidate diagnosis",
             "cohort_conditions":len(cohort),"cohort_parents":len({r["parent_id"] for r in cohort}),
             "sources":{key:{"path":str(path.relative_to(root)),"available":path.is_file(),"rows":len(data[key])} for key,path in sources.items()},
             "unexpected_rows":{"headroom":len(head_unexpected),"anatomy":len(pair_unexpected),"posterior":len(post_unexpected),
                                "decisions":len(decision_unexpected),"baseline":len(baseline_unexpected),"cnn_counterfactual":len(cnn_unexpected)},
             "incomplete_parent_rows":{"headroom":sum(not r["complete"] for r in head_parents),
                                       "anatomy":sum(not r["complete"] for r in pair_parents),
                                       "posterior":sum(not r["complete"] for r in post_parents),
                                       "decisions":sum(not r["complete"] for r in decision_parents),
                                       "cnn_counterfactual":sum(not r["complete"] for r in cnn_parents)},
             "bootstrap_samples":2000,"primary_group":"COMPLEX_FAMILIES","figures":manifest,
             "cost_scope":"headroom curves show paid cumulative updates; setup/load/shared initial full state remain separate in cost ledger",
             "no_physics_run":True,"new_labels":0,"training":False,"locked_opened":False,
             "failure_policy":"Missing/invalid/duplicates retained. Primary estimates/CI null with incomplete pairing; no success-only means."}
    write_json(out/"FIGURE_MANIFEST.json",{"figures":manifest,"synthetic_science":False})
    write_json(out/"SUMMARY.json",summary)
    return summary
