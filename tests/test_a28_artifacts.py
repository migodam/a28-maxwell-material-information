"""Tiny aggregation fixtures only; no generated scientific plots or physics.

Executed by the parent inside its paid unit-test driver.  Temporary records
are fabricated arithmetic fixtures, never actual A28 result evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from a28.artifacts import (HEADROOM_METRICS, frozen_cohort, group_table,
                           paired_headroom, parent_table, report_artifacts)


def _config():
    rows=[{"parent_id":parent,"family":family,"background_id":bg,
           "case_id":f"fixture_{parent}_{bg}","amplitude":1.2,"sigma":.01}
          for parent,family in ((10,"shell"),(20,"asymmetric_piecewise"))
          for bg in ("weak","ring")]
    return {"cohort":rows,"backgrounds":["weak","ring"],"grid":24,
            "primary_seed":281010,"bootstrap_samples":2000}


def _raw(config, *, method="F_TV", k=6):
    rows=[]
    for meta in config["cohort"]:
        value=1. if meta["background_id"]=="weak" else 3.
        rows.append({**meta,"method":method,"outer_attempt":k,"valid":True,"accepted":False,
                     "delta_nrmse":value,"support_iou":.5,"edge_f1":.6,
                     "full_data_residual_absolute":.2,"cumulative_wall_seconds":.3})
    return rows


def _paired_parents(config):
    rows=[]
    for parent,family,control,candidate in ((10,"shell",1.,.5),(20,"asymmetric_piecewise",10.,9.)):
        for method,k,error in (("F_TV",0,control),("F_TV",1,control),("F_TV",6,candidate),("L_TV",6,control)):
            rows.append({"method":method,"outer_attempt":k,"parent_id":parent,"family":family,
                         "complete":True,"delta_nrmse":error,"support_iou":.5,"edge_f1":.6,
                         "full_data_residual_absolute":.2,"cumulative_wall_seconds":.3})
    return rows


class FrozenAggregationTests(unittest.TestCase):
    def test_frozen_cohort_rejects_duplicates_missing_background_and_path_id(self):
        config=_config();self.assertEqual(len(frozen_cohort(config)),4)
        duplicated=deepcopy(config);duplicated["cohort"].append(duplicated["cohort"][0])
        with self.assertRaisesRegex(ValueError,"DUPLICATED"):
            frozen_cohort(duplicated)
        incomplete=deepcopy(config);incomplete["cohort"].pop()
        with self.assertRaisesRegex(ValueError,"MISSING_BACKGROUND"):
            frozen_cohort(incomplete)
        forbidden=deepcopy(config);forbidden["cohort"][0]["case_id"]="../locked/input"
        with self.assertRaisesRegex(ValueError,"SIMPLE_NAME"):
            frozen_cohort(forbidden)

    def test_parent_means_before_equal_parent_grouping(self):
        config=_config();cohort=frozen_cohort(config)
        rows=_raw(config)
        for row in rows:
            if row["parent_id"]==20:row["delta_nrmse"]*=2
        parents,unexpected=parent_table(rows,cohort,[{"method":"F_TV","outer_attempt":6}],HEADROOM_METRICS)
        self.assertFalse(unexpected)
        self.assertEqual([row["delta_nrmse"] for row in parents],[2.,4.])
        groups=group_table(parents,cohort,("method","outer_attempt"),HEADROOM_METRICS)
        all_group=next(row for row in groups if row["group"]=="ALL_FAMILIES")
        self.assertEqual(all_group["delta_nrmse"],3.)
        self.assertTrue(all_group["complete"])

    def test_one_failed_condition_nulls_parent_and_group_no_success_only_mean(self):
        config=_config();cohort=frozen_cohort(config);rows=_raw(config)
        rows[0].update(valid=False,error="fixture_failure")
        parents,_=parent_table(rows,cohort,[{"method":"F_TV","outer_attempt":6}],HEADROOM_METRICS)
        self.assertIsNone(parents[0]["delta_nrmse"])
        self.assertFalse(parents[0]["complete"])
        groups=group_table(parents,cohort,("method","outer_attempt"),HEADROOM_METRICS)
        self.assertIsNone(next(r for r in groups if r["group"]=="ALL_FAMILIES")["delta_nrmse"])

    def test_rejected_valid_update_is_retained_and_blank_union_csv_flag_is_absent(self):
        config=_config();rows=_raw(config)
        rows[0]["valid"]=""
        rows[1]["accepted"]="False"
        parents,_=parent_table(rows,frozen_cohort(config),[{"method":"F_TV","outer_attempt":6}],HEADROOM_METRICS)
        self.assertTrue(parents[0]["complete"])
        self.assertEqual(parents[0]["rejected_conditions"],2)
        self.assertEqual(parents[0]["delta_nrmse"],2.)

    def test_duplicate_or_unexpected_rows_cannot_be_silently_selected(self):
        config=_config();rows=_raw(config);rows.append(deepcopy(rows[0]))
        parents,_=parent_table(rows,frozen_cohort(config),[{"method":"F_TV","outer_attempt":6}],HEADROOM_METRICS)
        self.assertFalse(parents[0]["complete"])
        rows=_raw(config);extra=deepcopy(rows[0]);extra["parent_id"]=999;rows.append(extra)
        parents,unexpected=parent_table(rows,frozen_cohort(config),[{"method":"F_TV","outer_attempt":6}],HEADROOM_METRICS)
        self.assertEqual(len(unexpected),1)
        self.assertTrue(all(r["delta_nrmse"] is None for r in parents))

    def test_optional_undefined_risk_ratio_does_not_hide_valid_core_moments(self):
        config=_config();rows=[{**meta,"ESS":1.,"risk_mean":0.,"risk_excess_relative":None}
                              for meta in config["cohort"]]
        parents,_=parent_table(rows,frozen_cohort(config),[{}],("ESS","risk_mean","risk_excess_relative"),
                               required_metrics=("ESS","risk_mean"))
        self.assertTrue(all(r["complete"] for r in parents))
        self.assertEqual(parents[0]["ESS"],1.)
        self.assertIsNone(parents[0]["risk_excess_relative"])
        groups=group_table(parents,frozen_cohort(config),(),("ESS","risk_mean","risk_excess_relative"))
        self.assertEqual(groups[0]["ESS"],1.)
        self.assertIsNone(groups[0]["risk_excess_relative"])

    def test_paired_gain_is_mean_of_parent_gains_not_ratio_of_means(self):
        config=_config()
        result=paired_headroom(_paired_parents(config),frozen_cohort(config),config)
        row=next(r for r in result["summary_rows"] if r["comparison"]=="F6_vs_L6"
                 and r["group"]=="COMPLEX_FAMILIES" and r["metric"]=="delta_nrmse")
        self.assertAlmostEqual(row["mean_gain"],.3)
        self.assertNotAlmostEqual(row["mean_gain"],(11.-9.5)/11.)
        self.assertEqual(row["bootstrap_samples"],2000)
        second=paired_headroom(_paired_parents(config),frozen_cohort(config),config)
        self.assertEqual(result,second)

    def test_missing_pair_nulls_primary_point_and_interval(self):
        config=_config();parents=_paired_parents(config)
        next(r for r in parents if r["method"]=="F_TV" and r["outer_attempt"]==6 and r["parent_id"]==10)["complete"]=False
        result=paired_headroom(parents,frozen_cohort(config),config)
        row=next(r for r in result["summary_rows"] if r["comparison"]=="F6_vs_L6"
                 and r["group"]=="COMPLEX_FAMILIES" and r["metric"]=="delta_nrmse")
        self.assertFalse(row["complete"])
        self.assertIsNone(row["mean_gain"])
        self.assertIsNone(row["ci95_low"])
        self.assertEqual(row["paired_parents"],1)

    def test_non_2000_bootstrap_refused(self):
        config=_config();config["bootstrap_samples"]=100
        with self.assertRaisesRegex(ValueError,"2000"):
            paired_headroom(_paired_parents(config),frozen_cohort(config),config)

    def test_absent_final_files_remain_not_run_and_do_not_read_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            partial=root/"results/a28/headroom/PER_ITERATION_PARTIAL.csv"
            partial.parent.mkdir(parents=True);partial.write_text("partial fixture must not be read",encoding="utf-8")
            # Do not render fabricated science; every final source is absent.
            with patch("a28.artifacts._draw_distances",side_effect=lambda rows,config,out,m,b:m.append({"status":"NOT_RUN","name":"counterfactual_distances"})), \
                 patch("a28.artifacts._draw_posterior",side_effect=lambda rows,out,m,b:m.append({"status":"NOT_RUN","name":"kappa_conditional_risk"})), \
                 patch("a28.artifacts._draw_bp",side_effect=lambda rows,s,out,m,b:m.append({"status":"NOT_RUN","name":"bp_information_preservation"})):
                result=report_artifacts(root,_config())
            self.assertTrue(result["no_physics_run"])
            self.assertFalse(result["scientific_verdict_performed"])
            self.assertFalse(result["sources"]["headroom"]["available"])
            self.assertTrue(all(f["status"]=="NOT_RUN" for f in result["figures"]))
            self.assertFalse(list((root/"results/a28/artifacts").rglob("*.png")))
            saved=json.loads((root/"results/a28/artifacts/HEADROOM_PAIRED_STATISTICS.json").read_text())
            self.assertTrue(all(r["mean_gain"] is None for r in saved["summary_rows"]))


if __name__=="__main__":
    unittest.main()
