"""Small algebraic checks for A28. No training and no full-wave simulation.
Requires NumPy only. Run: python A28_theory_checks.py
Outputs A28_theory_checks.json next to this script.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

rng = np.random.default_rng(281010)
results: dict[str, object] = {}

def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-15))

def check(name: str, a: np.ndarray, b: np.ndarray, tol: float = 1e-11) -> None:
    error = rel(np.asarray(a), np.asarray(b))
    assert error < tol, (name, error)
    results[name] = {"relative_error": error, "passed": True}

def complex_random(shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape) + 1j*rng.normal(size=shape)

n, ns, m = 7, 3, 4
G = .035*complex_random((n,n))
G = (G+G.T)/2
S = complex_random((m,n))
inc = complex_random((n,ns))
chi0 = rng.uniform(.1,.5,n)+1j*rng.uniform(.01,.08,n)
chi1 = chi0 + rng.uniform(-.06,.12,n)+1j*rng.uniform(-.008,.012,n)

def state(chi: np.ndarray):
    L = np.eye(n)-chi[:,None]*G
    c = np.linalg.solve(L,chi[:,None]*inc)
    e = inc+G@c
    return L,c,e

L0,c0,e0 = state(chi0)
L1,c1,e1 = state(chi1)
d = chi1-chi0
check("finite_secant", c1-c0, np.linalg.solve(L1,d[:,None]*e0))
check("opposite_anchor_secant", c1-c0, np.linalg.solve(L0,d[:,None]*e1))
tangent = np.linalg.solve(L0,d[:,None]*e0)
remainder = S@np.linalg.solve(L1,d[:,None]*(G@tangent))
check("exact_feedback_secant_remainder", S@(c1-c0-tangent), remainder)

# Exact current-to-material pullback and cross-source decomposition.
dc = .03*complex_random((n,ns))
chat, ehat = c0+dc, e0+G@dc
D = np.sum(abs(ehat)**2,axis=1)
ratio = np.sum(ehat.conj()*chat,axis=1)/D
rr = L0@dc
pullback = np.sum(ehat.conj()*rr,axis=1)/D
check("current_material_pullback",ratio-chi0,pullback)
perp = rr-ehat*pullback[:,None]
check("cross_source_pythagoras",np.sum(abs(rr)**2,axis=1),D*abs(ratio-chi0)**2+np.sum(abs(perp)**2,axis=1))

# Real, strictly convex fixed-quadratic GN defect identity.
p, q = 6, 9
JA = rng.normal(size=(q,p)); E = .12*rng.normal(size=(q,p)); JF=JA+E
rA=rng.normal(size=q); dr=.1*rng.normal(size=q); rF=rA+dr
Lam=np.diag(rng.uniform(.3,1.5,p)); prior_gradient=rng.normal(size=p)
HA=JA.T@JA+Lam; HF=JF.T@JF+Lam
sA=-np.linalg.solve(HA,JA.T@rA+prior_gradient)
sF=-np.linalg.solve(HF,JF.T@rF+prior_gradient)
zeta=JF.T@(dr+E@sA)+E.T@(rA+JA@sA)
check("material_update_defect",sA-sF,np.linalg.solve(HF,zeta))

def quad(s):
    return .5*np.linalg.norm(rF+JF@s)**2+.5*s@Lam@s+prior_gradient@s
check("quadratic_excess_risk",np.array(quad(sA)-quad(sF)),np.array(.5*zeta@np.linalg.solve(HF,zeta)))

# Data compression: exact lost Gaussian log likelihood ratio.
U,_=np.linalg.qr(rng.normal(size=(9,3)))
P=U@U.T; fa=rng.normal(size=9); fb=rng.normal(size=9); yy=rng.normal(size=9)
ell=(yy-(fa+fb)/2)@(fa-fb)
ellp=(P@(yy-(fa+fb)/2))@(P@(fa-fb))
r=(np.eye(9)-P)@(fa-fb)
check("lost_log_likelihood_ratio",np.array(ell-ellp),np.array((yy-(fa+fb)/2)@r))
results["lost_log_likelihood_under_hypothesis_a"]={"mean":float(.5*r@r),"variance":float(r@r),"lost_KL":float(.5*r@r)}

# Reciprocal two-cell VIE algebra counterexample, NOT a geometric EM simulation.
g=.8
G2=np.array([[0.,g],[g,0.]])
CHI=np.array([[.1,1.],[1.,.1]])
C=np.stack([np.linalg.solve(np.eye(2)-x[:,None]*G2,x) for x in CHI])
EE=1+C@G2.T
f=C.sum(axis=1)
mu=CHI.mean(axis=0); mc=C.mean(axis=0); me=EE.mean(axis=0)
kappa=((CHI-mu)*(EE-me)).mean(axis=0)
check("two_cell_equal_measurements",np.array(f[0]),np.array(f[1]))
check("posterior_constitutive_covariance",mc-mu*me,kappa)
cmu=np.linalg.solve(np.eye(2)-mu[:,None]*G2,mu)
check("posterior_forward_commutator",cmu-mc,-np.linalg.solve(np.eye(2)-mu[:,None]*G2,kappa))
qmean=mc/me
cq=np.linalg.solve(np.eye(2)-qmean[:,None]*G2,qmean)
check("projected_mean_fits_data",np.array(cq.sum()),np.array(f.mean()))
risk_mean=float(np.mean(np.sum((CHI-mu)**2,axis=1)))
risk_ratio=float(np.mean(np.sum((CHI-qmean)**2,axis=1)))
check("mean_projection_excess_risk",np.array(risk_ratio-risk_mean),np.array(np.linalg.norm(qmean-mu)**2))
# True-field weighting changes the conditional estimand.
field_weighted=(EE**2*CHI).mean(axis=0)/(EE**2).mean(axis=0)
risk_weighted=float(np.mean(np.sum((CHI-field_weighted)**2,axis=1)))
results["two_cell_counterexample"]={
    "scope":"reciprocal finite VIE algebra, not a full-wave geometric simulation",
    "materials":CHI.tolist(),"measured_data":f.tolist(),
    "material_conditional_mean":mu.tolist(),"mean_current":mc.tolist(),
    "mean_internal_field":me.tolist(),"constitutive_covariance":kappa.tolist(),
    "forward_of_mean_material":float(cmu.sum()),
    "ratio_of_mean_currents":qmean.tolist(),
    "forward_of_ratio_material":float(cq.sum()),
    "material_MSE_sum_over_cells":risk_mean,
    "after_exact_data_consistency_MSE":risk_ratio,
    "relative_MSE_increase":risk_ratio/risk_mean-1,
    "true_field_weighted_MSE_minimizer":field_weighted.tolist(),
    "true_field_weighted_minimizer_material_MSE":risk_weighted,
    "weighted_loss_relative_material_MSE_increase":risk_weighted/risk_mean-1,
    "max_constitutive_matrix_condition":float(max(np.linalg.cond(np.eye(2)-x[:,None]*G2) for x in CHI))}

# Prior correlation couples an unmeasured coordinate to the measured coordinate.
rho=.99; noise_var=.01
prior=np.array([[1.,rho],[rho,1.]])
a=np.array([[1.,0.]])
post=prior-prior@a.T@np.linalg.solve(a@prior@a.T+noise_var*np.eye(1),a@prior)
results["prior_coupling_example"]={"rho":rho,"noise_variance":noise_var,"unmeasured_coordinate_prior_variance":1.,"unmeasured_coordinate_posterior_variance":float(post[1,1])}

# Harmless current error in a fixed, non-dark internal field (G=0).
ee=np.array([1.,1.]); dc2=np.array([.4,-.4])
check("nonzero_current_error_zero_material_error",np.array(ee@dc2),np.array(0.))
results["weak_direction_update_counterexample"]={"J":[[1.,0.]],"truth":[0.,1.],"estimate":[0.,0.],"data_residual":0.,"GN_step":0.,"squared_material_error":1.}

# A27 beta is a norm ratio, not a binary detectability threshold.
beta=.168546
complex_data_count=256
mahalanobis=math.sqrt(2*complex_data_count)*beta
results['beta_detectability_example']={
    'scope':'two fixed Gaussian mean hypotheses, not material identifiability',
    'beta':beta,
    'complex_data_count':complex_data_count,
    'real_mahalanobis_distance':mahalanobis,
    'equal_prior_binary_error':.5*math.erfc(mahalanobis/(2*math.sqrt(2)))}

out=Path(__file__).with_suffix('.json')
out.write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
