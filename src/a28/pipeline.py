"""A28 E1 sequential stages; no learning and no locked data access."""
from pathlib import Path
import json, math, time, runpy, shutil
import numpy as np
import torch
from a24.contracts import NumericalFailure
from a25.torch_physics import mass_pack, mass_unpack, pack_data
from a26.data import load_split, online_from_split, _safe_member
from a26.encoder import load_encoders
from a26.evaluation import clean, write_csv, CachedFullPhysics
from a26.training import load_trained_model, load_feature_cache
from a26.metrics import image_metrics
from a27.correction import CPUProxMeter, _value
from a25.prox import ProxProblem, solve_prox
from a28.budget import write_json, budget_history, BudgetExceeded
from .physics import CountedMaxwell, material_mass, numpy_state
from .posterior import finite_weights, moments, expected_risk


def _out(root):
    p=Path(root)/'results/a28';p.mkdir(parents=True,exist_ok=True);return p


def _inputs(root,data_root,a26_root,config,book,device,*,raw=False):
    old=json.loads((Path(root)/'configs/a26/pilot.json').read_text())
    with book.span('a28_verified_exposed_development_import'):
        labels=load_split(data_root,'validation');a=online_from_split(labels)
        path=Path(a26_root).resolve()/'results/a26/evaluation/261009/PREDICTIONS.npz'
        if any('locked' in part.lower() for part in path.parts):raise ValueError('LOCKED_INPUT_FORBIDDEN')
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['case_id'],a['case_id']):raise ValueError('A26_CASE_ORDER_MISMATCH')
            pred={k:z[k].copy() for k in ('N3','N1','BASE_N3','COMMON_BP_INITIAL')}
    enc=load_encoders(data_root,old,book,device=device)
    models={b:CountedMaxwell(e.reference) for b,e in enc.items()}
    if raw:
        path=_safe_member(data_root,'validation.npz',allow_locked=False)
        with book.span('a28_verified_offline_clean_current_import'):
            with np.load(path,allow_pickle=False) as z:
                for k in ('parent_id','split','case_id'):
                    if not np.array_equal(z[k],a[k]):raise ValueError('RAW_LABEL_IDENTITY_MISMATCH')
                rawarrays={k:z[k].copy() for k in ('current','clean')}
            if rawarrays['current'].shape!=(128,576,8) or rawarrays['clean'].shape!=(128,32,8):raise ValueError('OLD_CLEAN_LAYOUT_MISMATCH')
            for k,v in rawarrays.items():
                if v.dtype!=np.complex128 or not np.isfinite(v).all():raise ValueError('OLD_CLEAN_DTYPE_OR_FINITE_FAILURE')
        labels.update(rawarrays)
    return old,labels,a,enc,pred,models


def _indices(a,config):
    parents=sorted(set(map(int,a['parent_id'])))
    chosen=[]
    for f in config['families']:
        fp=sorted({int(a['parent_id'][i]) for i in range(len(a['data'])) if str(a['family'][i])==f})
        chosen+=fp[:int(config['parents_per_family'])]
    chosen=set(chosen)
    ii=[i for i in range(len(a['data'])) if int(a['parent_id'][i]) in chosen and
        float(a['amplitude'][i])==float(config['amplitude']) and float(a['sigma'][i])>0]
    ii.sort(key=lambda i:(int(a['parent_id'][i]),config['backgrounds'].index(str(a['background_id'][i]))))
    if len(ii)!=len(chosen)*2 or any(str(a['background_id'][i]) not in config['backgrounds'] for i in ii):raise ValueError('FIXED_COHORT_INCOMPLETE')
    return ii


def _meta(a,i):
    return {'case_id':str(a['case_id'][i]),'parent_id':int(a['parent_id'][i]),'family':str(a['family'][i]),
            'background_id':str(a['background_id'][i]),'amplitude':float(a['amplitude'][i]),'sigma':float(a['sigma'][i])}


def _frozen(root, config=None):
    p=_out(root)/'FROZEN_PROTOCOL.json'
    if not p.exists():raise ValueError('A28_FREEZE_REQUIRED_BEFORE_SCIENTIFIC_RUN')
    frozen = json.loads(p.read_text())
    if config is not None and frozen != config:
        raise ValueError('A28_FROZEN_PROTOCOL_CONFIG_MISMATCH')
    return frozen


def _cache(root,key):return _out(root)/'exact_states'/(str(key)+'.npz')
def _sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
def _rel(a,b):
    den=float(np.linalg.norm(b));return float(np.linalg.norm(a-b)/den) if den else None


def stage_theory(root,data_root,a26_root,config,book,device):
    job=_out(root)/'jobs'/book.metadata['job'];p=job/'A28_theory_checks.py'
    p.write_bytes((Path(root)/'theory/a28/A28_theory_checks.py').read_bytes())
    with book.span('a28_existing_theory_verification'):
        result=runpy.run_path(str(p))['results']
    count=sum(isinstance(v,dict) and v.get('passed') is True for v in result.values())
    return {'status':'PASS','identities':count,'scope':'finite algebra; not Maxwell imaging evidence','original_theory_unchanged':True}


def stage_audit(root,data_root,a26_root,config,book,device):
    old,labels,a,enc,pred,models=_inputs(root,data_root,a26_root,config,book,device,raw=True)
    ii=_indices(a,config);checks=[];rawchecks=[]
    with torch.no_grad():
        for i in ii:
            ref=enc[str(a['background_id'][i])].reference
            c=torch.tensor(labels['current'][i][None],device=device)
            e=ref.incident+ref.green(c);y=ref.observe(c)
            chi=torch.tensor(labels['chi'][i][None],device=device)
            ce=float(torch.linalg.vector_norm(c-chi[...,None]*e)/torch.linalg.vector_norm(c))
            de=_rel(y[0].cpu().numpy(),labels['clean'][i])
            # Clean labels are reused, never rebuilt with a new full solve.
            rawchecks.append({**_meta(a,i),'constitutive_identity':ce,'clean_receiver_identity':de,
                              'source_count':8,'clean_state_reused':True})
            if max(ce,de)>1e-10:raise NumericalFailure('FROZEN_CLEAN_FIELDS_INCONSISTENT')
        for bg in config['backgrounds']:
            i=next(i for i in ii if str(a['background_id'][i])==bg)
            ref=enc[bg].reference;m=models[bg];x=material_mass(pred['N3'][i],ref)
            st=m.cached(x,_cache(root,a['case_id'][i]))
            # Independent frozen SciPy execution; paid separately, same equations.
            # CachedFullPhysics.full_state owns a full_forward_calls span.
            # The repaired ledger reserves that span before its real solve;
            # do not reserve twice. Historical audit receipts stay unchanged.
            original=CachedFullPhysics(ref,old,book).full_state(pred['N3'][i])
            err=_rel(st.data[0].cpu().numpy(),original.data)
            rng=np.random.default_rng(config['primary_seed']+int(i))
            d=torch.tensor(rng.normal(size=(1,1152)),dtype=torch.float64,device=device)
            ch=st.chi[0];free=torch.cat((torch.minimum(ch.real+.5,4.-ch.real)>1e-3,torch.minimum(ch.imag,.5-ch.imag)>1e-3))
            d*=free[None];d/=torch.linalg.vector_norm(d)
            load=torch.tensor(rng.normal(size=(1,512)),dtype=torch.float64,device=device);load/=torch.linalg.vector_norm(load)
            j=m.jvp(x,d,st);v=m.adjoint(x,load,st)
            scale=max(float(torch.linalg.vector_norm(j)*torch.linalg.vector_norm(load)),float(torch.linalg.vector_norm(v)*torch.linalg.vector_norm(d)))
            dot=float(abs((j*load).sum()-(v*d).sum()))/scale
            diffs=[]
            for eps in (1e-4,1e-5,1e-6,1e-7):
                plus=m(x+eps*d);minus=m(x-eps*d)
                fd=pack_data(plus.data-minus.data)/(2*eps)
                diffs.append({'epsilon':eps,'derivative_relative_error':float(torch.linalg.vector_norm(fd-j)/torch.linalg.vector_norm(j))})
            row={'background':bg,'old_backend_relative_error':err,'real_mass_adjoint_defect':dot,
                 'full_backward_residual':float(st.residual.max()),'finite_differences':diffs}
            if max(err,dot)>1e-9 or min(r['derivative_relative_error'] for r in diffs)>1e-5:raise NumericalFailure('A28_T0_IDENTITY_FAILED')
            checks.append(row)
        cp=load_trained_model(Path(a26_root)/'results/a26/learning/261009/N3/best.pt',device=device,config=old)
        cache=load_feature_cache(Path(a26_root)/'results/a26/cache/validation_N3.npz',role='validation',method='N3',config=old)
        values=[]
        for start in range(0,128,4):
            f=torch.tensor(cache['features'][start:start+4],dtype=torch.float32,device=device)
            b=torch.tensor(cache['base_mass'][start:start+4],dtype=torch.float64,device=device)
            values.append(mass_unpack(cp.predict_mass(f,b),1/576).cpu().numpy())
        cp_error=_rel(np.concatenate(values),pred['N3'])
        if cp_error>1e-9:raise ValueError('FROZEN_N3_REPRODUCTION_CONFLICT')
    result={'T0':'PASS','checks':checks,'clean_current_checks':rawchecks,'N3_checkpoint_rows':128,'N3_prediction_relative_error':cp_error,
            'locked_opened':False,'new_training_labels':0,'clean_material_evaluations_reused':len(ii)}
    write_json(_out(root)/'AUDIT.json',result);return {k:v for k,v in result.items() if k!='clean_current_checks'}


def _train_rules(data_root,enc,book):
    from .edits import calibrate_train_rules
    with book.span('a28_TRAIN_only_structure_rule_calibration'):
        t=load_split(data_root,'train')
        backgrounds=np.stack([enc[str(b)].reference.chi_b.cpu().numpy() for b in t['background_id']])
        rules,audit=calibrate_train_rules(t['chi'],backgrounds,parent_ids=t['parent_id'],split='train',amplitudes=t['amplitude'],grid=(24,24))
    return rules,audit


def _single_update(x,obs,sigma,model,fullmodel,full,config,owner,damping=1.):
    """Measurement-only same TV update. At most ONE paid candidate audit.

    Rejection consumes one outer attempt; next attempt halves damping.
    No truth, geometry labels, family or reference direction enters this call.
    """
    ref=fullmodel.reference;_sync(x.device);start=time.perf_counter();before=dict(ref.book.counts)
    bg=mass_pack(ref.chi_b,ref.volume)[None];rms=math.sqrt(float(ref.yb_norm2)/(ref.q*ref.s))
    lam=.01*(1+float(sigma)/rms)*float(config['tv_multiplier'])
    old,olddata,oldtv=_value(full,obs,x,bg,lam,ref)
    st=full if model is fullmodel else model(x,recover_current=False)
    g=model.data_gradient(x,obs,st);g2=float(g.square().sum())
    if g2<=0:return x,full,{'accepted':False,'reason':'ZERO_GRADIENT','wall_seconds':time.perf_counter()-start,'lambda':lam}
    jg=model.jvp(x,g,st);curv=float(jg.square().sum()/ref.yb_norm2)
    if curv<=0 or not math.isfinite(curv):raise NumericalFailure('NONZERO_GRADIENT_ZERO_CURVATURE')
    step=float(damping)*min(.8*g2/curv,float(config['step_cap_background_multiple'])*float(ref.step))
    p=ProxProblem((x-step*g).cpu().numpy()[0],bg.cpu().numpy()[0],step*lam,(24,24),ref.volume,
                  config=config,owner_id=owner,algorithm='fgp')
    prox=solve_prox(p,budget=CPUProxMeter(ref.book))
    if not prox.converged:raise NumericalFailure('UNCERTIFIED_COMMON_PROX')
    candidate=torch.tensor(prox.x[None],dtype=torch.float64,device=x.device)
    disp=candidate-x;dn=float(torch.linalg.vector_norm(disp))
    meta={'old_objective':old,'old_data_loss':olddata,'old_tv':oldtv,'step':step,'damping':damping,'lambda':lam,
          'prox_iterations':prox.metadata['iterations'],'prox_kkt':prox.metadata['kkt_residual'],
          'prox_gap':prox.metadata['normalized_gap'],'proposed_mass_norm':dn,'gradient_mass_norm':math.sqrt(g2),
          'truth_access':False,'line_trials':1}
    if dn>float(config['trust_radius_mass']):
        accepted=False;newfull=full;result=x;meta.update(reason='TRUST_REJECT')
    else:
        audit=fullmodel(candidate);new,newdata,newtv=_value(audit,obs,candidate,bg,lam,ref)
        allowance=1e-12*max(1.,abs(old));accepted=bool(new<=old-float(config['armijo'])*dn**2/(2*step)+allowance and newdata<=olddata+allowance)
        result,newfull=(candidate,audit) if accepted else (x,full)
        meta.update(reason='ACCEPTED' if accepted else 'FULL_OBJECTIVE_OR_DATA_REJECT',proposed_objective=new,proposed_data_loss=newdata,proposed_tv=newtv)
    _sync(x.device)
    meta.update(accepted=accepted,wall_seconds=time.perf_counter()-start,
                actions={k:int(v-before.get(k,0)) for k,v in ref.book.counts.items() if v!=before.get(k,0)})
    return result,newfull,meta


def stage_micro(root,data_root,a26_root,config,book,device):
    if json.loads((_out(root)/'AUDIT.json').read_text())['T0']!='PASS':raise ValueError('AUDIT_REQUIRED')
    old,labels,a,enc,pred,models=_inputs(root,data_root,a26_root,config,book,device)
    ii=_indices(a,config);rules,train_audit=_train_rules(data_root,enc,book);rows=[]
    for bg in config['backgrounds']:
        i=next(i for i in ii if str(a['background_id'][i])==bg);ref=enc[bg].reference
        x=material_mass(pred['N3'][i],ref);m=models[bg];st=m.cached(x,_cache(root,a['case_id'][i]))
        obs=torch.tensor(a['data'][i][None],dtype=torch.complex128,device=device)
        for repeat in range(3):
            _sync(device);t=time.perf_counter();value=m(x);_sync(device)
            rows.append({'background':bg,'role':'fresh_full_state','repeat':repeat,'wall_seconds':time.perf_counter()-t,'backward_residual':float(value.residual.max())})
        for method in ('L_TV','F_TV'):
            model=m if method=='F_TV' else enc[bg].models['L']
            value,state,meta=_single_update(x,obs,float(a['sigma'][i]),model,m,st,config,'micro:'+method+bg)
            rows.append({'background':bg,'role':method,'wall_seconds':meta['wall_seconds'],'accepted':meta['accepted'],
                         'certificate':{k:meta.get(k) for k in ('prox_kkt','prox_gap','prox_iterations')},'scientific_material_score_computed':False})
    result={'status':'COMPLETE','rows':rows,'TRAIN_rules':rules,'TRAIN_audit':train_audit,
            'proposed_parents':16,'proposed_conditions':32,'scientific_material_score_computed':False,
            'planned_primary_new_state_upper_bound':736,'micro_state_evaluations_inclusive':True}
    write_json(_out(root)/'MICRO.json',result);return {k:v for k,v in result.items() if k not in ('TRAIN_rules','TRAIN_audit')}


def stage_freeze(root,data_root,a26_root,config,book,device):
    if (_out(root)/'FROZEN_PROTOCOL.json').exists():raise ValueError('PROTOCOL_ALREADY_FROZEN')
    audit=json.loads((_out(root)/'AUDIT.json').read_text());micro=json.loads((_out(root)/'MICRO.json').read_text())
    if audit['T0']!='PASS' or micro['status']!='COMPLETE':raise ValueError('AUDIT_MICRO_REQUIRED')
    # Cost-only cohort choice, before any scientific new material scores.
    fulltime=max(r['wall_seconds'] for r in micro['rows'] if r['role']=='fresh_full_state')
    updatetime=max(r['wall_seconds'] for r in micro['rows'] if r['role'] in ('F_TV','L_TV'))
    prediction=1.3*(320*fulltime+384*updatetime+180.)
    h=budget_history(root);available=min(config['gpu_limit_seconds']-h['gpu_seconds']-300.,config['cpu_limit_seconds']-h['cpu_seconds']-300.)
    if prediction>available:config['parents_per_family']=max(1,int(4*available/prediction))
    old,labels,a,enc,pred,models=_inputs(root,data_root,a26_root,config,book,device)
    ii=_indices(a,config);config.update(frozen=True,protocol_frozen=True,edit_rules=micro['TRAIN_rules'],
        cohort=[_meta(a,i) for i in ii],cohort_indices=ii,predicted_upper_gpu_seconds=prediction,
        freeze_selection='TRAIN rules + cost-only family-balanced prefix; no new scientific scores inspected')
    write_json(Path(root)/'a28_config.json',config);write_json(_out(root)/'FROZEN_PROTOCOL.json',config)
    return {'status':'FROZEN','conditions':len(ii),'parents':len(ii)//2,'prediction_seconds':prediction,'full_wave_cap':768,'new_training':False,'locked_opened':False}


def _binary_noise_check(fa,fb,la,lb,sigma,seed,repeats):
    """Fresh independent validation noise AFTER candidates are fixed.

    Theory L error is only under a world generated by L. Empirical L error
    here deliberately evaluates L likelihood decisions on F-generated data.
    """
    rng=np.random.default_rng(seed);shape=(int(repeats),)+fa.shape
    errors=[];raw=[]
    for source in (fa,fb):
        n=(rng.normal(size=shape)+1j*rng.normal(size=shape))*sigma/np.sqrt(2)
        ys=source[None]+n
        ff=np.sum(np.abs(ys-fa)**2,axis=(1,2))-np.sum(np.abs(ys-fb)**2,axis=(1,2))
        ll=np.sum(np.abs(ys-la)**2,axis=(1,2))-np.sum(np.abs(ys-lb)**2,axis=(1,2))
        truth_index=0 if source is fa else 1
        errors.append((float(np.mean((ff>0)!=truth_index)),float(np.mean((ll>0)!=truth_index))))
        raw.append({'F_errors':int(np.count_nonzero((ff>0)!=truth_index)),'L_errors':int(np.count_nonzero((ll>0)!=truth_index)), 'samples':repeats})
    return {'binary_F_validation_error':float(np.mean([e[0] for e in errors])),
            'binary_L_on_F_validation_error':float(np.mean([e[1] for e in errors])),
            'validation_noise_seed':int(seed),'validation_noise_samples_per_hypothesis':int(repeats),
            'validation_error_counts':raw,'search_noise_reused':False}


def _pair_record(meta,name,center,edited,fa,fb,la,lb,sigma,background,config,seed):
    dF=float(np.sqrt(2)*np.linalg.norm(fb-fa)/sigma)
    dL=float(np.sqrt(2)*np.linalg.norm(lb-la)/sigma)
    diff=(fb-fa)-(lb-la)
    dn=float(np.linalg.norm(center-background))
    metric=image_metrics(edited,center,background)
    return {**meta,'edit':name,'material_difference_delta_nrmse':float(np.linalg.norm(edited-center)/dn) if dn else None,
            'F_white_distance':dF,'L_white_distance':dL,'F_minus_L_pair_distance':dF-dL,
            'F_over_L_pair_distance':dF/dL if dL>0 else None,
            'finite_feedback_secant_norm':float(np.linalg.norm(diff)),
            'finite_feedback_secant_white_norm':float(np.sqrt(2)*np.linalg.norm(diff)/sigma),
            'feedback_secant_fraction':float(np.linalg.norm(diff)/np.linalg.norm(fb-fa)) if np.linalg.norm(fb-fa)>0 else None,
            'F_theory_equal_prior_binary_error':.5*math.erfc(dF/(2*np.sqrt(2))),
            'L_theory_binary_error_IF_L_generated_world':.5*math.erfc(dL/(2*np.sqrt(2))),
            'support_iou_pair':metric['support_iou'],'edge_f1_pair':metric['edge_f1'],
            'structural_material_difference_min':config['gate_structural_material_delta_min'],
            **_binary_noise_check(fa,fb,la,lb,sigma,seed,config['binary_noise_replicates'])}


def stage_anatomy(root,data_root,a26_root,config,book,device):
    _frozen(root,config);from .edits import make_structural_edits
    _,labels,a,enc,pred,models=_inputs(root,data_root,a26_root,config,book,device,raw=True)
    out=_out(root)/'anatomy';out.mkdir(exist_ok=True);rows=[];failures=[];casepaths=[];baseline_rows=[]
    for ordinal,i in enumerate(config['cohort_indices']):
        book.check();meta=_meta(a,i);bg=meta['background_id'];ref=enc[bg].reference;m=models[bg];lin=enc[bg].models['L']
        background=ref.chi_b.cpu().numpy();center=pred['N3'][i]
        cnn_state=m.cached(material_mass(center,ref),_cache(root,a['case_id'][i]));cs=numpy_state(cnn_state)
        truth_c=torch.tensor(labels['current'][i],device=device);truth_e=ref.incident+ref.green(truth_c[None])[0]
        sources={
            'TRUTH_CENTERED_ORACLE':{'chi':labels['chi'][i],'current':labels['current'][i],'field':truth_e.cpu().numpy(),'data':labels['clean'][i]},
            'CNN_CENTERED_DATA_ONLY':cs}
        lc=lin(material_mass(center,ref),recover_current=False).data[0].cpu().numpy()
        lt=lin(material_mass(labels['chi'][i],ref),recover_current=False).data[0].cpu().numpy()
        br=_pair_record(meta,'N3_VS_TRUE_ENDPOINT',labels['chi'][i],center,labels['clean'][i],cs['data'],lt,lc,
                        meta['sigma'],background,config,config['validation_noise_seed']+1000*i+900)
        # A local background spectral diagnostic is not posterior uncertainty.
        err=material_mass(center-labels['chi'][i],ref)[0]
        sig=material_mass(labels['chi'][i]-background,ref)[0]
        strong=enc[bg].background_spectrum/enc[bg].background_spectrum[0]>.05
        vv=enc[bg].background_right[strong]
        br.update(background_strong_error_fraction=float((vv@err).square().sum()/err.square().sum()),
                  background_strong_truth_energy_fraction=float((vv@sig).square().sum()/sig.square().sum()),
                  background_strong_rank=int(strong.sum()),local_diagnostic_not_global_information_fraction=True)
        baseline_rows.append(br)
        payload={};editmeta={};baseline_score=image_metrics(center,labels['chi'][i],background)
        for role,st in sources.items():
            offline=role.startswith('TRUTH');provenance='OFFLINE_TRUTH_DIAGNOSTIC' if offline else 'ONLINE_CNN_PREDICTION'
            edits=make_structural_edits(st['chi'],background,config['edit_rules'],center_provenance=provenance,include_contrast=offline)
            candidates=[st['chi']];currents=[st['current']];fields=[st['field']];data=[st['data']]
            lin0=lin(material_mass(st['chi'],ref),recover_current=False).data[0].cpu().numpy();ldata=[lin0];names=['CENTER']
            for j,edit in enumerate(edits):
                r={**meta,'role':role,'edit':edit.method_id,
                   'edit_status':edit.status,'edit_audit':edit.summary(),'truth_centered':offline,'online_algorithm_claim':False}
                if edit.status.startswith('INVALID'):
                    # A failed structural edit is retained as a duplicate point
                    # in the labelled diagnostic bank, not silently resampled.
                    nextst=st.copy();lf=lin0.copy();r.update(valid=False,error=edit.status)
                    failures.append(r.copy())
                else:
                    try:
                        with book.span('a28_counterfactual_'+role, counterfactual_materials=1):
                            nextst=numpy_state(m(material_mass(edit.chi,ref)))
                            lf=lin(material_mass(edit.chi,ref),recover_current=False).data[0].cpu().numpy()
                        r.update(valid=True,**_pair_record(meta,r['edit'],st['chi'],edit.chi,st['data'],nextst['data'],lin0,lf,meta['sigma'],background,config,
                              config['validation_noise_seed']+1000*i+100*j+int(offline)))
                        # Identity (1), using the already stored c0/e0. No need
                        # to refactor/re-solve the cached truth material state.
                        rr=nextst['current']-st['current']
                        rhs=(edit.chi-st['chi'])[:,None]*st['field']
                        lhs=rr-edit.chi[:,None]*(ref.G.cpu().numpy()@rr)
                        scale=max(np.linalg.norm(rhs),np.linalg.norm(st['current'])*1e-12)
                        r['finite_secant_identity_residual']=float(np.linalg.norm(lhs-rhs)/scale)
                        if r['finite_secant_identity_residual']>1e-9:raise NumericalFailure('FULL_WAVE_FINITE_SECANT_IDENTITY')
                    except BudgetExceeded:raise
                    except Exception as exc:
                        r.update(valid=False,error=type(exc).__name__+':'+str(exc));failures.append(r.copy());rows.append(r)
                        write_json(out/'FAILURES_PARTIAL.json',clean(failures));raise
                candidates.append(edit.chi if not edit.status.startswith('INVALID') else st['chi'])
                currents.append(nextst['current']);fields.append(nextst['field']);data.append(nextst['data']);ldata.append(lf);names.append(r['edit'])
                if not offline and r['valid']:
                    # Scores are AFTER candidates fixed. No truth used to rank.
                    r['offline_candidate_material_error']=image_metrics(edit.chi,labels['chi'][i],background)['delta_nrmse']
                    r['N3_material_error']=baseline_score['delta_nrmse']
                rows.append(r)
            payload.update({role+'_chi':np.asarray(candidates),role+'_current':np.asarray(currents),
                            role+'_field':np.asarray(fields),role+'_F':np.asarray(data),role+'_L':np.asarray(ldata),role+'_names':np.asarray(names)})
            editmeta[role]=[e.summary() for e in edits]
        payload.update(case_id=np.asarray(meta['case_id']),observation=a['data'][i],truth=labels['chi'][i],background=background,sigma=np.asarray(meta['sigma']))
        path=out/(meta['case_id']+'.npz');np.savez_compressed(path,**payload);casepaths.append(path.name)
        write_json(out/(meta['case_id']+'.json'),{'meta':meta,'edits':clean(editmeta),'posterior_prior':'uniform labelled bank, including explicitly flagged zero-edit duplicates'})
        write_csv(out/'PAIR_METRICS_PARTIAL.csv',clean(rows))
        print(json.dumps({'stage':'anatomy','conditions':ordinal+1,'rows':len(rows),'invalid_edits':len(failures)}),flush=True)
    write_csv(out/'PAIR_METRICS.csv',clean(rows));write_json(out/'FAILURES.json',clean(failures))
    write_csv(out/'BASELINE_DIAGNOSTICS.csv',baseline_rows)
    result={'status':'COMPLETE','conditions':len(config['cohort_indices']),'pairs':len(rows),'invalid_edits':len(failures),'case_files':casepaths,
            'truth_bank_role':'OFFLINE IDENTIFIABILITY ONLY','online_bank_role':'data-only candidate construction; no truth geometry','search_validation_noise_independent':True}
    write_json(out/'SUMMARY.json',result);return result


def _score_row(meta,method,k,chi,truth,bg,state,obs,elapsed,accepted,extra=None):
    metrics=image_metrics(chi,truth,bg)
    return {**meta,'method':method,'outer_attempt':k,'accepted':accepted,**metrics,
            'full_data_residual_absolute':float(torch.linalg.vector_norm(state.data-obs)),
            'full_backward_residual':float(state.residual.max()),'cumulative_wall_seconds':elapsed,**(extra or {})}


def stage_headroom(root,data_root,a26_root,config,book,device):
    _frozen(root,config);_,labels,a,enc,pred,models=_inputs(root,data_root,a26_root,config,book,device)
    out=_out(root)/'headroom';out.mkdir(exist_ok=True);rows=[];traces=[];failures=[];outputs={}
    for ordinal,i in enumerate(config['cohort_indices']):
        meta=_meta(a,i);bg=meta['background_id'];ref=enc[bg].reference;m=models[bg];background=ref.chi_b.cpu().numpy()
        initial=material_mass(pred['N3'][i],ref);full=m.cached(initial,_cache(root,a['case_id'][i]));obs=torch.tensor(a['data'][i][None],device=device)
        for method in config['headroom_methods']:
            x=initial.clone();state=full;elapsed=0.;damping=1.;accepted_count=0;recorded=[pred['N3'][i].copy()]
            scale=float(torch.linalg.vector_norm(obs-ref.yb))
            row0=_score_row(meta,method,0,pred['N3'][i],labels['chi'][i],background,state,obs,elapsed,False,{'valid':True})
            row0['full_data_relative_residual']=row0['full_data_residual_absolute']/scale if scale else None
            row0['full_data_normalization_denominator']=scale
            rows.append(row0)
            for k in range(1,int(config['headroom_outer_attempts'])+1):
                book.check();model=m if method=='F_TV' else enc[bg].models['L']
                try:
                    x,state,diag=_single_update(x,obs,meta['sigma'],model,m,state,config,method+':'+meta['case_id']+':'+str(k),damping)
                    elapsed+=diag['wall_seconds'];accepted_count+=int(diag['accepted'])
                    damping=damping if diag['accepted'] else damping/2.
                    chi=mass_unpack(x,ref.volume)[0].cpu().numpy();recorded.append(chi)
                    row=_score_row(meta,method,k,chi,labels['chi'][i],background,state,obs,elapsed,diag['accepted'],
                                   {'accepted_total':accepted_count,'valid':True,'damping_next':damping,'reason':diag['reason']})
                    row['full_data_relative_residual']=row['full_data_residual_absolute']/scale if scale else None
                    row['full_data_normalization_denominator']=scale
                    rows.append(row);traces.append({**meta,'method':method,'outer_attempt':k,**diag})
                except BudgetExceeded:raise
                except Exception as exc:
                    fail={**meta,'method':method,'outer_attempt':k,'valid':False,'error':type(exc).__name__+':'+str(exc)}
                    failures.append(fail);rows.append(fail);write_json(out/'FAILURES_PARTIAL.json',clean(failures));break
            outputs[meta['case_id']+'_'+method]=np.asarray(recorded)
        write_csv(out/'PER_ITERATION_PARTIAL.csv',rows);write_json(out/'TRACES_PARTIAL.json',clean(traces))
        print(json.dumps({'stage':'headroom','conditions':ordinal+1,'rows':len(rows),'failures':len(failures)}),flush=True)
    write_csv(out/'PER_ITERATION.csv',rows);write_json(out/'TRACES.json',clean(traces));write_json(out/'FAILURES.json',failures)
    np.savez_compressed(out/'TRAJECTORIES.npz',**outputs)
    result={'status':'COMPLETE','conditions':len(config['cohort_indices']),'rows':len(rows),'numerical_failures':len(failures),
            'max_outer_attempts':6,'ordinary_TV_baseline_alias':'F_TV','truth_in_candidate_or_acceptance':False,'full_F_acceptance_cost_paid_for_L':True}
    write_json(out/'SUMMARY.json',result);return result


def stage_posterior(root,data_root,a26_root,config,book,device):
    _frozen(root,config);_,labels,a,enc,pred,models=_inputs(root,data_root,a26_root,config,book,device)
    out=_out(root)/'posterior';out.mkdir(exist_ok=True);rows=[];decision_rows=[];failures=[]
    for ordinal,i in enumerate(config['cohort_indices']):
        book.check();meta=_meta(a,i);ref=enc[meta['background_id']].reference;m=models[meta['background_id']]
        with np.load(_out(root)/'anatomy'/(meta['case_id']+'.npz'),allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
        obs=bank['observation'];truth=bank['truth'];bg=bank['background'];payload={}
        for role in config['posterior_banks']:
            chi,c,e,f,l=[bank[role+'_'+k] for k in ('chi','current','field','F','L')]
            pf,lf=finite_weights(f,obs,meta['sigma']);pl,ll=finite_weights(l,obs,meta['sigma'])
            distributions={'UNIFORM':np.ones(len(chi))/len(chi),'F_LIKELIHOOD':pf,'L_LIKELIHOOD':pl}
            for name,p in distributions.items():
                stats=moments(chi,c,e,f,p,G=ref.G.cpu().numpy(),incident=ref.incident.cpu().numpy(),S=ref.S.cpu().numpy(),volume=ref.volume)
                scalar=stats['scalars'];mu=stats['mu_chi'];ratio=stats['ratio'];row={**meta,'bank':role,'distribution':name,
                    **scalar,'candidate_count':len(chi),'declared_prior':'uniform labelled finite bank; not calibrated global posterior',
                    'truth_centered':role.startswith('TRUTH'),'CNN_centered_prior_is_data_dependent':not role.startswith('TRUTH'),
                    'decision_material_nrmse':image_metrics(mu,truth,bg)['delta_nrmse'],
                    'ratio_material_nrmse':image_metrics(ratio,truth,bg)['delta_nrmse'],
                    'N3_material_nrmse':image_metrics(pred['N3'][i],truth,bg)['delta_nrmse'],
                    'ratio_bias_delta_scale':float(np.sqrt(ref.volume)*np.linalg.norm(ratio-mu)/(np.sqrt(ref.volume)*np.linalg.norm(truth-bg))),
                    'physical_mean_audit':'NOT_RUN_OTHER_DISTRIBUTION'}
                for k,v in stats.items():
                    if k!='scalars':payload[role+'__'+name+'__'+k]=v
                # Identities use material/current scale for tiny kappa, not a
                # tiny posterior variance denominator to manufacture a pass.
                if max(scalar[k] for k in ('constitutive_mean_identity','field_mean_identity','closed_mean_identity','data_mean_identity','ratio_identity_material_scaled'))>1e-9:
                    raise NumericalFailure('A28_POSTERIOR_MOMENT_IDENTITY_FAILED')
                if role=='CNN_CENTERED_DATA_ONLY' and name=='F_LIKELIHOOD':
                    try:
                        meanmass=material_mass(mu,ref);st=m(meanmass)
                        propagated,_=m._solve(st,torch.tensor(stats['kappa'][None],device=device),role='posterior_covariance_source')
                        rhs=-ref.observe(propagated)[0].cpu().numpy();lhs=st.data[0].cpu().numpy()-stats['mu_data']
                        err=float(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(stats['mu_data']),np.finfo(float).tiny))
                        if err>1e-9:raise NumericalFailure('A28_FORWARD_MEAN_COMMUTATOR_FAILED')
                        row.update(physical_mean_audit='PASS',forward_commutator_identity=err,
                            forward_commutator_absolute=float(np.linalg.norm(lhs)),forward_commutator_white=float(np.sqrt(2)*np.linalg.norm(lhs)/meta['sigma']),
                            mean_full_data_residual=float(np.linalg.norm(st.data[0].cpu().numpy()-obs)))
                        ratio_metrics=image_metrics(ratio,truth,bg);row['ratio_feasible']=ratio_metrics['feasible']
                        if ratio_metrics['feasible']:
                            rstate=m(material_mass(ratio,ref));row['ratio_full_data_residual']=float(np.linalg.norm(rstate.data[0].cpu().numpy()-obs))
                        else:row['ratio_full_data_residual']=None;row['ratio_physical_audit']='ILLEGAL_RATIO_NO_FALLBACK'
                        # Optional single correction of the finite material
                        # mean. This is an actual measured risk/data tradeoff,
                        # not an assertion that a CNN is a posterior mean.
                        new,newstate,diag=_single_update(meanmass,torch.tensor(obs[None],device=device),meta['sigma'],m,m,st,config,'finite-mean:'+meta['case_id'])
                        action=mass_unpack(new,ref.volume)[0].cpu().numpy()
                        risk=expected_risk(action,chi,p,ref.volume)
                        row.update(mean_feedback_accepted=diag['accepted'],mean_feedback_expected_risk=risk,
                                   mean_feedback_risk_increase=risk-scalar['risk_mean'],
                                   mean_feedback_material_nrmse=image_metrics(action,truth,bg)['delta_nrmse'],
                                   mean_feedback_full_data_residual=float(np.linalg.norm(newstate.data[0].cpu().numpy()-obs)),
                                   mean_feedback_risk_identity_absolute=abs(risk-scalar['risk_mean']-ref.volume*np.linalg.norm(action-mu)**2))
                        payload[role+'__'+name+'__mean_feedback_material']=action
                        payload[role+'__'+name+'__mean_feedback_data']=newstate.data[0].cpu().numpy()
                    except BudgetExceeded:raise
                    except Exception as exc:
                        fail={**meta,'bank':role,'distribution':name,'error':type(exc).__name__+':'+str(exc)};failures.append(fail)
                        row['physical_mean_audit']='FAILED';row['error']=fail['error']
                        if 'IDENTITY' in fail['error'] or 'COMMUTATOR' in fail['error']:raise
                rows.append(row)
            mf_mu=np.einsum('k,kn->n',pf,chi);ml_mu=np.einsum('k,kn->n',pl,chi)
            ef=image_metrics(mf_mu,truth,bg)['delta_nrmse'];el=image_metrics(ml_mu,truth,bg)['delta_nrmse'];en=image_metrics(pred['N3'][i],truth,bg)['delta_nrmse']
            fselect=int(np.argmax(pf));lselect=int(np.argmax(pl))
            decision_rows.append({**meta,'bank':role,'F_mean_error':ef,'L_mean_error':el,'N3_error':en,
                    'F_vs_L_material_gain':(el-ef)/el if el>0 else None,
                    'F_vs_L_gain_denominator':el,'F_vs_L_gain_undefined':el<=0,
                    'F_vs_N3_material_gain':(en-ef)/en if en>0 else None,
                    'F_vs_N3_gain_denominator':en,'F_vs_N3_gain_undefined':en<=0,
                    'F_mean_risk_under_F':expected_risk(mf_mu,chi,pf,ref.volume),
                    'L_mean_risk_under_F':expected_risk(ml_mu,chi,pf,ref.volume),
                    'F_map_index':fselect,'L_map_index':lselect,'F_MAP_material_error':image_metrics(chi[fselect],truth,bg)['delta_nrmse'],
                    'L_MAP_material_error':image_metrics(chi[lselect],truth,bg)['delta_nrmse'],
                    'F_ESS':float(1/np.sum(pf*pf)),'L_ESS':float(1/np.sum(pl*pl)),
                    'F_log_likelihood':lf.tolist(),'L_log_likelihood':ll.tolist(),
                    'oracle_center_excluded_from_algorithm_headroom':role.startswith('TRUTH')})
        np.savez_compressed(out/(meta['case_id']+'.npz'),**payload)
        write_csv(out/'PER_DISTRIBUTION_PARTIAL.csv',clean(rows));write_csv(out/'DECISIONS_PARTIAL.csv',clean(decision_rows))
        print(json.dumps({'stage':'posterior','conditions':ordinal+1,'rows':len(rows),'failures':len(failures)}),flush=True)
    write_csv(out/'PER_DISTRIBUTION.csv',clean(rows));write_csv(out/'DECISIONS.csv',clean(decision_rows));write_json(out/'FAILURES.json',failures)
    result={'status':'COMPLETE','rows':len(rows),'conditions':len(config['cohort_indices']),'physical_failures':len(failures),
            'finite_diagnostic_prior_only':True,'kappa_network':'NOT_RUN','post_mean_correction_is_one_gradient_TV_update':True}
    write_json(out/'SUMMARY.json',result);return result


def stage_bp(root,data_root,a26_root,config,book,device):
    _frozen(root,config);from .bp import audit_bp
    old,labels,a,enc,pred,models=_inputs(root,data_root,a26_root,config,book,device)
    out=_out(root)/'bp';out.mkdir(exist_ok=True);pairs={b:[] for b in config['backgrounds']};cnnrows=[];feature_payload={}
    cp=load_trained_model(Path(a26_root)/'results/a26/learning/261009/N3/best.pt',device=device,config=old)
    for i in config['cohort_indices']:
        meta=_meta(a,i);e=enc[meta['background_id']];ref=e.reference
        with np.load(_out(root)/'anatomy'/(meta['case_id']+'.npz'),allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
        for role in config['posterior_banks']:
            response=bank[role+'_F'];chi=bank[role+'_chi'];names=bank[role+'_names']
            # Fixed clean response pair feature trace. Covariance handles noise;
            # it is not inferred from full nonlinear channel Euclidean norms.
            with torch.no_grad():
                feats=[e.encode(np.asarray(y,np.complex128),meta['sigma'],'N3') for y in response]
                images=[mass_unpack(cp.predict_mass(v['features'],v['base_mass']),ref.volume)[0].cpu().numpy() for v in feats]
            for j in range(1,len(response)):
                if np.array_equal(chi[j],chi[0]):continue
                key=meta['case_id']+'_'+role+'_'+str(names[j])
                pair={**meta,'pair_id':key,'candidate_kind':role+':'+str(names[j]),'data_a':response[0],'data_b':response[j],
                      'actual_features_a':feats[0]['features'][0].cpu().numpy(),'actual_features_b':feats[j]['features'][0].cpu().numpy()}
                pairs[meta['background_id']].append(pair)
                dchi=chi[j]-chi[0];dcnn=images[j]-images[0];den=np.linalg.norm(dchi)*np.linalg.norm(dcnn)
                cnnrows.append({**meta,'role':role,'edit':str(names[j]),'pair_id':key,
                    'candidate_a_CNN_material_nrmse':image_metrics(images[0],chi[0],bank['background'])['delta_nrmse'],
                    'candidate_b_CNN_material_nrmse':image_metrics(images[j],chi[j],bank['background'])['delta_nrmse'],
                    'CNN_material_change_over_true_change':float(np.linalg.norm(dcnn)/np.linalg.norm(dchi)),
                    'CNN_change_material_alignment':float(np.vdot(dchi,dcnn).real/den) if den>0 else None,
                    'CNN_nearest_pair_correct_a':bool(np.linalg.norm(images[0]-chi[0])<np.linalg.norm(images[0]-chi[j])),
                    'CNN_nearest_pair_correct_b':bool(np.linalg.norm(images[j]-chi[j])<np.linalg.norm(images[j]-chi[0])),
                    'noise':'clean diagnostic feature trace; existing CNN unchanged','truth_used_online':False})
                feature_payload[key+'__CNN_a']=images[0];feature_payload[key+'__CNN_b']=images[j]
    summaries={};rows=[]
    for bg in config['backgrounds']:
        with book.span('a28_actual_N3_BP_information_SVD', BP_offline_SVDs=1):result=audit_bp(enc[bg].reference,data_pairs=pairs[bg])
        np.savez_compressed(out/(bg+'_BP.npz'),**result['arrays']);summaries[bg]=result['summary'];rows+=result['rows']
    write_csv(out/'PAIR_INFORMATION.csv',clean(rows));write_csv(out/'CNN_COUNTERFACTUAL_RESPONSE.csv',cnnrows)
    np.savez_compressed(out/'CNN_PAIR_IMAGES.npz',**feature_payload);write_json(out/'SUMMARY.json',clean(summaries))
    return {'status':'COMPLETE','BP_pairs':sum(len(v) for v in pairs.values()),'cutoff_rows':len(rows),'CNN_pair_rows':len(cnnrows),
            'new_full_material_states':0,'actual_12_channel_trace':True,'CNN_retrained':False,'formal_global_identifiability':'NOT_ESTABLISHED'}


def stage_report(root,data_root,a26_root,config,book,device):
    _frozen(root,config)
    from .artifacts import report_artifacts
    return report_artifacts(root,config,book=book)
