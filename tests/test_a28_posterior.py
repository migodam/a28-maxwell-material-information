import unittest
import numpy as np
from a28.posterior import finite_weights, moments


class PosteriorTests(unittest.TestCase):
    def test_real_noise_factor(self):
        w,_=finite_weights(np.array([[[0j]],[[1+0j]]]),np.zeros((1,1),complex),2.)
        self.assertAlmostEqual(w[1]/w[0],np.exp(-.25))

    def test_geometric_mass_moments_no_conjugation(self):
        rng=np.random.default_rng(281010);n=6;s=3
        G=.04*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
        S=rng.normal(size=(4,n))+1j*rng.normal(size=(4,n))
        inc=(rng.normal(size=(n,s))+1j*rng.normal(size=(n,s)))/6
        chi=rng.uniform(.1,.8,(4,n))+1j*rng.uniform(.01,.15,(4,n))
        c=np.array([np.linalg.solve(np.eye(n)-x[:,None]*G,x[:,None]*inc) for x in chi])
        e=inc+np.einsum('ij,kjs->kis',G,c);f=np.einsum('qi,kis->kqs',S,c)
        a=moments(chi,c,e,f,np.array([.1,.2,.3,.4]),G=G,incident=inc,S=S,volume=1/36)
        for k in ('constitutive_mean_identity','field_mean_identity','closed_mean_identity','data_mean_identity','ratio_identity_material_scaled','risk_excess_identity_absolute'):
            self.assertLess(a['scalars'][k],1e-12,k)
        mu=a['mu_chi'];mf=S@np.linalg.solve(np.eye(n)-mu[:,None]*G,mu[:,None]*inc)
        propagated=-S@np.linalg.solve(np.eye(n)-mu[:,None]*G,a['kappa'])
        np.testing.assert_allclose(mf-a['mu_data'],propagated,atol=1e-13)
        conjugated=np.einsum('k,kn,kns->ns',a['weights'],chi-mu,(e-a['mu_field']).conj())
        self.assertGreater(np.linalg.norm(conjugated-a['kappa']),1e-6)

    def test_degenerate_posterior_stays_explicit(self):
        w,_=finite_weights(np.array([[[0j]],[[100j]]]),np.zeros((1,1),complex),.01)
        np.testing.assert_array_equal(w,[1.,0.])

    def test_forbid_zero_noise_likelihood(self):
        with self.assertRaises(ValueError):finite_weights(np.zeros((2,1,1),complex),np.zeros((1,1),complex),0.)

    def test_duplicate_labels_do_not_create_material_ambiguity(self):
        chi=np.full((3,2),.1+.02j);G=np.zeros((2,2),complex)
        inc=np.ones((2,1),complex);S=np.eye(2,dtype=complex)
        current=chi[:,:,None]*inc;field=np.broadcast_to(inc,(3,2,1));response=current.copy()
        a=moments(chi,current,field,response,np.ones(3)/3,G=G,incident=inc,S=S,volume=.5)
        self.assertAlmostEqual(a['scalars']['ESS'],3.)
        self.assertEqual(a['scalars']['distinct_material_count'],1)
        self.assertEqual(a['scalars']['duplicate_label_count'],2)
        self.assertAlmostEqual(a['scalars']['distinct_material_ESS'],1.)
