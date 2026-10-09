"""Finite-bank conditional moments, never claimed to be a calibrated posterior."""
import numpy as np


def finite_weights(response, observation, sigma, prior=None):
    """CN(0,sigma²I): log L=-||y-f||²/sigma² (real variance sigma²/2)."""
    response = np.asarray(response, np.complex128)
    observation = np.asarray(observation, np.complex128)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('POSITIVE_NOISE_REQUIRED_FOR_FINITE_LIKELIHOOD')
    k = len(response)
    prior = np.ones(k)/k if prior is None else np.asarray(prior, float)
    if prior.shape != (k,) or np.any(prior<=0) or not np.isfinite(prior).all() or not np.isclose(prior.sum(),1):
        raise ValueError('DECLARED_POSITIVE_NORMALIZED_PRIOR_REQUIRED')
    log = np.log(prior)-np.sum(np.abs(response-observation)**2,axis=tuple(range(1,response.ndim)))/float(sigma)**2
    weights = np.exp(log-log.max()); weights /= weights.sum()
    return weights, log


def moments(chi, current, field, response, weights, *, G, incident, S, volume):
    """Equation 8/9/11/12 in the actual sqrt(area) current/field coordinates.

    Kappa has NO conjugation: constitutive multiplication is chi*field.
    Source fields and currents are mass-normalized; chi is physical material.
    """
    chi, current, field, response = [np.asarray(x,np.complex128) for x in (chi,current,field,response)]
    p = np.asarray(weights,float)
    if p.shape!=(len(chi),) or np.any(p<0) or not np.isclose(p.sum(),1) or not np.isfinite(p).all():
        raise ValueError('FINITE_NORMALIZED_WEIGHT_VECTOR_REQUIRED')
    mu = np.einsum('k,kn->n',p,chi)
    mc = np.einsum('k,kns->ns',p,current); me=np.einsum('k,kns->ns',p,field)
    mf = np.einsum('k,kqs->qs',p,response)
    kappa = np.einsum('k,kn,kns->ns',p,chi-mu,field-me)
    den=np.sum(np.abs(me)**2,axis=1)
    if np.any(den<=0) or not np.isfinite(den).all():
        raise ValueError('DARK_MEAN_FIELD_RATIO_UNDEFINED')
    A=np.sum(me.conj()*mc,axis=1); K=np.sum(me.conj()*kappa,axis=1)
    ratio=A/den; bias=K/den
    def norm(x):return float(np.linalg.norm(x))
    def identity(x,y,scale):
        return norm(x-y)/max(norm(scale),np.finfo(float).tiny)
    risk=lambda a:float(volume*np.einsum('k,kn->',p,np.abs(chi-a)**2))
    risk_mu, risk_ratio = risk(mu), risk(ratio)
    excess=float(volume*norm(ratio-mu)**2)
    scale_material=max(norm(mu),1.)
    # Failed/zero edits remain labelled hypotheses with their declared weights.
    # Aggregate exact duplicate materials separately: labelled ESS must never
    # be used to claim actual physical posterior ambiguity.
    distinct, inverse = np.unique(chi, axis=0, return_inverse=True)
    material_weights = np.bincount(inverse, weights=p, minlength=len(distinct))
    positive = material_weights[material_weights > 0]
    scalar={
        'ESS':float(1/np.sum(p*p)), 'maximum_weight':float(p.max()),
        'entropy':float(-np.sum(p[p>0]*np.log(p[p>0]))),
        'distinct_material_count':int(len(distinct)),
        'duplicate_label_count':int(len(chi)-len(distinct)),
        'distinct_material_ESS':float(1/np.sum(material_weights**2)),
        'distinct_material_entropy':float(-np.sum(positive*np.log(positive))),
        'distinct_material_maximum_weight':float(material_weights.max()),
        'kappa_norm_mass':norm(kappa),
        'constitutive_mean_identity':identity(mc,mu[:,None]*me+kappa,mc),
        'field_mean_identity':identity(me,incident+G@mc,me),
        'closed_mean_identity':identity(mc-mu[:,None]*(G@mc),mu[:,None]*incident+kappa,mc),
        'data_mean_identity':identity(mf,S@mc,mf),
        'ratio_identity_absolute':norm((ratio-mu)-bias),
        'ratio_identity_material_scaled':norm((ratio-mu)-bias)/scale_material,
        'risk_mean':risk_mu,'risk_ratio':risk_ratio,'risk_excess':risk_ratio-risk_mu,
        'risk_excess_identity_absolute':abs((risk_ratio-risk_mu)-excess),
        'risk_excess_relative':(risk_ratio-risk_mu)/risk_mu if risk_mu>1e-20 else None,
        'ratio_bias_mass':float(np.sqrt(volume)*norm(ratio-mu)),
        'mean_field_energy_min':float(den.min()),'mean_field_energy_max':float(den.max()),
        'risk_identity_scale_note':'tiny posterior variances/commutators use absolute material/current scales, no floor used to pass a scientific gate',
    }
    return {'mu_chi':mu,'mu_current':mc,'mu_field':me,'mu_data':mf,'kappa':kappa,
            'ratio':ratio,'ratio_bias':bias,'field_denominator':den,'A':A,'K':K,
            'weights':p,'scalars':scalar}


def expected_risk(action, chi, weights, volume):
    return float(volume*np.einsum('k,kn->',np.asarray(weights,float),np.abs(np.asarray(chi)-action)**2))
