"""Saved-evidence screening numbers only; never opens a dataset or runs physics."""
import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from a28.budget import charged_job, write_json


def table(name):
    with (ROOT/'results/a28'/name).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, '') else None


def valid(row):
    return row.get('valid', 'True') == 'True'


def describe(values):
    a = np.asarray([v for v in values if v is not None], float)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                minimum=float(a.min()), maximum=float(a.max())) if len(a) else dict(count=0)


def evidence(config, book):
    pairs=table('anatomy/PAIR_METRICS.csv')
    cnn=table('bp/CNN_COUNTERFACTUAL_RESPONSE.csv')
    post=table('posterior/PER_DISTRIBUTION.csv')
    bp=table('bp/PAIR_INFORMATION.csv')
    baseline=table('anatomy/BASELINE_DIAGNOSTICS.csv')
    threshold=config['gate_structural_material_delta_min']
    clear=config['gate_F_distance_clear']; ambiguous=config['gate_F_distance_ambiguous']
    complex_families=set(config['complex_families'])
    counts=[]; clear_missed=set(); ambiguous_parents=set(); ambiguous_families=set()
    cnnindex={(r['pair_id']):r for r in cnn}
    for role in config['posterior_banks']:
        for family in ['ALL_FAMILIES','COMPLEX_FAMILIES',*config['families']]:
            local=[r for r in pairs if r['role']==role and r['edit']!='CONTRAST_ONLY'
                   and (family=='ALL_FAMILIES' or family=='COMPLEX_FAMILIES' and r['family'] in complex_families or r['family']==family)]
            good=[r for r in local if valid(r)]
            important=[r for r in good if number(r,'material_difference_delta_nrmse')>=threshold]
            clearly=[r for r in important if number(r,'F_white_distance')>=clear]
            unclear=[r for r in important if number(r,'F_white_distance')<=ambiguous]
            counts.append(dict(role=role,group=family,planned_structural_pairs=len(local),valid_pairs=len(good),
                important_pairs=len(important),F_clear_important_pairs=len(clearly),F_ambiguous_important_pairs=len(unclear),
                F_clear_L_not_clear=sum(number(r,'L_white_distance')<clear for r in clearly),
                F_validation_error=describe([number(r,'binary_F_validation_error') for r in important]),
                L_on_F_validation_error=describe([number(r,'binary_L_on_F_validation_error') for r in important]),
                F_white_distance=describe([number(r,'F_white_distance') for r in important]),
                L_white_distance=describe([number(r,'L_white_distance') for r in important]),
                descriptive_pair_counts_not_independent_scenes=True))
            if family!='COMPLEX_FAMILIES':continue
            for r in clearly:
                cr=cnnindex.get(r['case_id']+'_'+role+'_'+r['edit'])
                if cr and (cr['CNN_nearest_pair_correct_a']!='True' or cr['CNN_nearest_pair_correct_b']!='True'):
                    clear_missed.add(int(r['parent_id']))
            for r in unclear:
                # Declared uniform prior is .2/.25 per labelled hypothesis;
                # this is a finite-bank witness, not a calibrated class prior.
                ambiguous_parents.add(int(r['parent_id']));ambiguous_families.add(r['family'])
    groups=table('artifacts/HEADROOM_GROUPS.csv')
    stats=table('artifacts/HEADROOM_PAIRED_STATISTICS.csv')
    decisions=table('artifacts/POSTERIOR_DECISION_PAIRED.csv')
    headprimary=[r for r in stats if r['group']=='COMPLEX_FAMILIES']
    bankprimary=[r for r in decisions if r['group']=='COMPLEX_FAMILIES' and r['role']=='CNN_CENTERED_DATA_ONLY']
    qualified_post=[r for r in post if r['bank']=='CNN_CENTERED_DATA_ONLY' and r['distribution']=='F_LIKELIHOOD']
    nondeg=[r for r in qualified_post if number(r,'distinct_material_ESS')>=config['gate_kappa_nondegenerate_ESS_min']]
    kappa_useful=[r for r in nondeg if number(r,'ratio_bias_delta_scale')>=config['gate_kappa_ratio_bias_material_scale_min']]
    identities=('constitutive_mean_identity','field_mean_identity','closed_mean_identity','data_mean_identity',
                'ratio_identity_material_scaled','forward_commutator_identity','risk_excess_identity_absolute',
                'mean_feedback_risk_identity_absolute')
    kappa=dict(conditions=len(qualified_post),distinct_material_ESS=describe([number(r,'distinct_material_ESS') for r in qualified_post]),
        nondegenerate_conditions=len(nondeg),nondegenerate_and_material_effective_conditions=len(kappa_useful),
        ratio_bias_delta_scale=describe([number(r,'ratio_bias_delta_scale') for r in qualified_post]),
        uniform_CNN_ratio_bias_delta_scale=describe([number(r,'ratio_bias_delta_scale') for r in post
                if r['bank']=='CNN_CENTERED_DATA_ONLY' and r['distribution']=='UNIFORM']),
        identity_maxima={k:max((number(r,k) for r in post if number(r,k) is not None),default=None) for k in identities},
        mean_feedback_accepted=sum(r['mean_feedback_accepted']=='True' for r in qualified_post),
        mean_feedback_risk_increase=describe([number(r,'mean_feedback_risk_increase') for r in qualified_post]))
    bp_summary=[]
    for bg in config['backgrounds']:
        for cutoff in ('machine','1e-12','1e-9','1e-6'):
            local=[r for r in bp if r['background_id']==bg and r['cutoff_id']==cutoff]
            bp_summary.append(dict(background_id=bg,cutoff=cutoff,rows=len(local),
                retained_rank=int(local[0]['retained_rank']) if local else None,
                information_preserved_fraction=describe([number(r,'information_preserved_fraction') for r in local]),
                actual_FP32_measurement_reconstruction_relative_error=describe([number(r,'actual_BP_feature_measurement_reconstruction_relative_error') for r in local])))
    # Gains are authoritative parent-paired artifacts, never ratios of pooled
    # candidate conditions. No posterior-oracle gain enters this headroom gate.
    possible=[r for r in headprimary if r['comparison']=='F6_vs_N3' and r['metric']=='delta_nrmse']
    possible += [r for r in bankprimary if r['comparison']=='F_mean_vs_N3']
    headroom=any(r['complete']=='True' and number(r,'mean_gain')>=config['gate_E2_material_headroom_min']
                 and number(r,'ci95_low')>0 for r in possible)
    # POSTHOC, cached-only explanatory diagnostics. Neither changes a frozen
    # gate nor constitutes a new deployable candidate/score.
    bias_rows=[];coverage=[]
    for meta in config['cohort']:
        book.check()
        with np.load(ROOT/'results/a28/anatomy'/(meta['case_id']+'.npz'),allow_pickle=False) as archive:
            chi=archive['CNN_CENTERED_DATA_ONLY_chi'];truth=archive['truth'];background=archive['background']
            err=np.linalg.norm(chi-truth,axis=1)/np.linalg.norm(truth-background)
            coverage.append(dict(**meta,N3_error=float(err[0]),offline_best_legal_bank_error=float(err.min()),
                offline_oracle_bank_gain=float((err[0]-err.min())/err[0]),
                offline_best_index=int(err.argmin()),online_algorithm_claim=False))
            for role in config['posterior_banks']:
                fs,ls=archive[role+'_F'],archive[role+'_L'];names=archive[role+'_names']
                for j in range(1,len(fs)):
                    if np.array_equal(fs[0],fs[j]):continue
                    df,dl=fs[j]-fs[0],ls[j]-ls[0]
                    nf,nl=np.linalg.norm(df),np.linalg.norm(dl)
                    if not nf or not nl:continue
                    den=meta['sigma']*nl/np.sqrt(2)
                    dot=float(np.vdot(dl,df).real)
                    midpoint=(ls[0]+ls[j])/2
                    analytic_L_error=.5*(ndtr(float(np.vdot(dl,fs[0]-midpoint).real)/den)+
                        ndtr(-float(np.vdot(dl,fs[j]-midpoint).real)/den))
                    # F-center or F-midpoint calibration is OFFLINE ORACLE.
                    # It isolates intercept bias/orientation, never a baseline.
                    center_aligned=.5*(ndtr(-.5*nl*nl/den)+ndtr((.5*nl*nl-dot)/den))
                    midpoint_aligned=ndtr(-.5*dot/den)
                    bias_rows.append(dict(**meta,role=role,edit=str(names[j]),
                        nonlinear_linear_secant_cosine=dot/(nf*nl),
                        analytic_L_on_F_binary_error=float(analytic_L_error),
                        OFFLINE_F_center_aligned_L_error=float(center_aligned),
                        OFFLINE_F_midpoint_aligned_L_error=float(midpoint_aligned),
                        online_algorithm_claim=False,posthoc_not_gate_input=True))
    for name, rows in [('POSTHOC_PAIR_BIAS_DIAGNOSTIC.csv',bias_rows),('OFFLINE_LEGAL_BANK_COVERAGE.csv',coverage)]:
        with (ROOT/'results/a28'/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    byparent={}
    for row in coverage:byparent.setdefault(row['parent_id'],[]).append(row)
    coverage_gain=[]
    for parent, local in byparent.items():
        if local[0]['family'] in complex_families:
            base=np.mean([r['N3_error'] for r in local]);best=np.mean([r['offline_best_legal_bank_error'] for r in local])
            coverage_gain.append((base-best)/base)
    out=dict(scope='EXPOSED_DEVELOPMENT_FINITE_CANDIDATES',pair_counts=counts,
        clear_missed_complex_parents=sorted(clear_missed),ambiguous_complex_parents=sorted(ambiguous_parents),
        ambiguous_complex_families=sorted(ambiguous_families),
        E2_data_only_material_headroom_gate=headroom,
        E3_important_ambiguity_gate=len(ambiguous_parents)>=config['gate_E3_ambiguous_complex_parents_min'] and
                 len(ambiguous_families)>=config['gate_E3_complex_families_min'],
        kappa_effective_gate=len(kappa_useful)>=config['gate_kappa_nondegenerate_conditions_min'],
        headroom_primary=headprimary,legal_bank_primary=bankprimary,
        final_family_metrics=[r for r in groups if r['outer_attempt'] in ('0','6')],
        kappa=kappa,BP=bp_summary,
        baseline_F_white_distance=describe([number(r,'F_white_distance') for r in baseline]),
        baseline_background_strong_error_fraction=describe([number(r,'background_strong_error_fraction') for r in baseline]),
        OFFLINE_legal_bank_coverage_gain_complex=describe(coverage_gain),
        POSTHOC_bias_diagnostic=dict(
            secant_cosine=describe([r['nonlinear_linear_secant_cosine'] for r in bias_rows]),
            L_on_F_error=describe([r['analytic_L_on_F_binary_error'] for r in bias_rows]),
            OFFLINE_F_center_aligned_L_error=describe([r['OFFLINE_F_center_aligned_L_error'] for r in bias_rows]),
            OFFLINE_F_midpoint_aligned_L_error=describe([r['OFFLINE_F_midpoint_aligned_L_error'] for r in bias_rows]),
            posthoc_not_gate_input=True,online_algorithm_claim=False),
        future_execution_authorized=False,global_identifiability_fraction='NOT_IDENTIFIABLE_FROM_THIS_EXPERIMENT')
    write_json(ROOT/'results/a28/EVIDENCE_SUMMARY.json',out)
    print(json.dumps({k:out[k] for k in ('clear_missed_complex_parents','ambiguous_complex_parents',
        'E2_data_only_material_headroom_gate','E3_important_ambiguity_gate','kappa_effective_gate')},ensure_ascii=False))
    return out


if __name__=='__main__':
    config=json.loads((ROOT/'a28_config.json').read_text())
    with charged_job(ROOT,sys.argv[1],config=config,metadata={'stage':'evidence-report'}) as book:
        book.deadline-=book.started_wall-START_WALL
        book.started_wall,book.started_cpu=START_WALL,START_CPU
        evidence(config,book)
