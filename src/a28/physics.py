"""Reuse the frozen Maxwell adapter; count every new attempted material state."""
from pathlib import Path
import numpy as np
import torch
from a27.exact import ExactMaxwell, ExactState
from a24.contracts import NumericalFailure
from a25.torch_physics import mass_pack


class CountedMaxwell(ExactMaxwell):
    def __call__(self, mass, *, recover_current=False):
        # E1 reservations are per material, not per batched solver invocation.
        # Keep this driver explicitly single-state so a future caller cannot
        # accidentally undercount a batch at the hard 768-state boundary.
        if mass.ndim != 2 or mass.shape[0] != 1:
            raise ValueError('A28_SINGLE_MATERIAL_STATE_REQUIRED')
        self.book.reserve('full_wave_material_states', 768)
        return super().__call__(mass, recover_current=recover_current)

    def cached(self, mass, path):
        """Reuse only exact same state/LU, with equation and source guards.

        This is an A28 cache of a previously PAID evaluation, not a new state
        solve. Cache load/assembly/backward audits remain billed. No pseudoinverse.
        """
        path = Path(path)
        if mass.ndim == 1:
            mass = mass[None]
        if not path.exists():
            state = self(mass)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {name: getattr(state, name).detach().cpu().numpy()
                       for name in ('mass', 'chi', 'current', 'field', 'w', 'data',
                                    'lu', 'pivots', 'residual')}
            payload.update(background=self.reference.chi_b.cpu().numpy(),
                           cache_protocol=np.asarray('A28-EXACT-STATE-v1'))
            with self.book.span('a28_exact_state_cache_write', exact_cache_writes=1):
                np.savez_compressed(path, **payload)
            return state
        ref = self.reference
        with self.book.span('a28_exact_state_cache_load', exact_cache_loads=1):
            with np.load(path, allow_pickle=False) as archive:
                a = {k: archive[k].copy() for k in archive.files}
            if (str(a['cache_protocol']) != 'A28-EXACT-STATE-v1'
                    or not np.array_equal(a['background'], ref.chi_b.cpu().numpy())
                    or not np.array_equal(a['mass'], mass.cpu().numpy())):
                raise ValueError('A28_FOREIGN_OR_STALE_STATE_CACHE')
            t = {k: torch.as_tensor(v, device=mass.device) for k, v in a.items()
                 if k not in ('background', 'cache_protocol')}
            matrix = torch.eye(ref.n, dtype=torch.complex128, device=mass.device)-t['chi'][...,None]*ref.G
            state = ExactState(t['mass'], t['chi'], t['current'], t['field'], t['w'], t['data'],
                               matrix, t['lu'], t['pivots'], t['residual'], id(self))
            rhs = state.chi[...,None]*ref.incident
            checks = ((matrix@state.current-rhs, rhs),
                      (state.field-ref.incident-ref.green(state.current), state.field),
                      (state.data-ref.observe(state.current), state.data))
            for residual, scale in checks:
                den = float(torch.linalg.vector_norm(scale))
                err = float(torch.linalg.vector_norm(residual)) / den if den else float(torch.linalg.vector_norm(residual))
                if not np.isfinite(err) or err > 1e-10:
                    raise NumericalFailure('A28_CACHED_EQUATION_MISMATCH')
            # This paid backward solve tests the actual saved LU, not just its
            # saved current. It introduces no new material state evaluation.
            check, _ = self._solve(state, rhs, role='cached_LU_audit')
            if float(torch.linalg.vector_norm(check-state.current)/torch.linalg.vector_norm(state.current)) > 1e-10:
                raise NumericalFailure('A28_CACHED_LU_MISMATCH')
        return state


def material_mass(chi, ref):
    return mass_pack(torch.as_tensor(chi, dtype=torch.complex128, device=ref.G.device), ref.volume)[None]


def numpy_state(state):
    return {name: getattr(state, name)[0].detach().cpu().numpy()
            for name in ('chi', 'current', 'field', 'data')}
