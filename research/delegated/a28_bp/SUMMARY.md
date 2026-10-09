# A28-E1 BP information audit handoff

Status: implementation complete; numerical tests/real audits **NOT_RUN by worker**.
The parent will execute the 14 tiny tests and the real cached-background audit
inside its paid A28 driver. No Maxwell material-state solve, labels, data-split
load, GPU job, plots, model training or publication was performed here.

Owned files:

- `src/a28/bp.py`
- `tests/test_a28_bp.py`
- this short handoff

## Exact source mapping

Read A28 Part I §5, Part II §14 and E1; read the actual A26
`FrozenEncoder._prepare_operators/encode` and A25 `mass_pack/mass_unpack/pack_data`.
The actual N3 feature channels 2 and 3 are the **unboxed** physical material BP
increment, subsequently cast to float32. Their map is

\[
 B_{physical}=\frac{step}{\sqrt{v}\,\|y_b\|^2}J_b^T,
 \quad BP(y)=B_{physical}\,pack(y-y_b),
 \quad B_{mass}=\sqrt v B_{physical}.
\]

Here `J_b` is the background Jacobian in real MASS material coordinates. Data
packing is source-major `[Re q, Im q]`; materials use `[Re n, Im n]`. The map is
1152×512 for the real 24×24/32-receiver/8-source experiment. It is derived only
from existing `Db`, `eb`, `yb`, volume and frozen step. Neither full truth-state
J nor a saved oracle field is accessed.

## API

```python
from a28.bp import audit_bp

# One invocation per cached background; collect all same-background pairs to
# reuse its one SVD. The caller owns resource accounting and output files.
result = audit_bp(encoder_or_reference, [
    {
        "pair_id": "frozen_candidate_pair_id",
        "parent_id": parent_id,
        "family": family_for_offline_grouping,
        "data_a": cached_candidate_response_a,  # complex128 (32,8)
        "data_b": cached_candidate_response_b,
        "sigma": declared_common_complex_noise_sigma,
        # Optional already computed real A26 pre-normalizer features:
        "actual_features_a": features_a,       # (12,24,24)
        "actual_features_b": features_b,
    },
])
# result["summary"] and result["rows"] are JSON/CSV scalar records.
# result["arrays"] is a caller-owned NPZ payload.
```

Alternatively pass `actual_bp_a/b` extracted channels 2/3 (complex n or real
`[2,n]`); do not pass both extracted and complete features. If actual features
are absent the float32 audit is explicitly a simulation of the exact BP cast,
not a trace of the full encoder. Parent may use `build_bp_map`,
`factor_bp_matrix`, and `candidate_information` directly for finer caching.

## Statistical and precision boundaries

- Four fixed relative B-singular-value thresholds: `max(shape)*eps64`, 1e−12,
  1e−9 and 1e−6. Save the entire singular spectrum and last 16 tiny values.
  A full *numerical* rank is not a proof of mathematical injectivity.
- Proper complex noise has per-real-coordinate variance `sigma²/2`. BP noise
  covariance is `(sigma²/2) B Bᵀ`. Its declared OFFLINE covariance pseudoinverse
  is evaluated in SVD coordinates, not by a physical matrix pseudoinverse.
- Candidate BP Mahalanobis distance is `sqrt(2)/sigma` times the projection of
  the original data difference onto retained B right-singular coordinates.
  Naive Euclidean BP norms are reported and explicitly not likelihood distances.
- Save projected/lost energies, lost pair KL, orthogonal energy identity,
  FP64 and FP32 data reconstruction errors, plus actual BP-channel reconstruction
  if existing features are provided. Absolute image-pair rounding is used; just
  rounding a difference could hide cancellation loss.
- Whole 12-channel feature differences are retained descriptively. Other
  channels include boxes/nonlinearities; their covariance is **NOT_ESTIMATED**.
  BP-only conditioning does not establish information loss of the whole encoder
  or inability of a finite CNN to use information.
- `sigma=0`, zero pairs and zero BP maps have explicit undefined statistics;
  there is no denominator floor used to certify preservation.
- General SPD-noise helper `covariance_information` verifies coordinate/noise
  transformations via Cholesky whitening. No covariance pinv is formed.

## Test coverage (14 tiny array tests; parent execution pending)

Exact Torch/A26 packing, conjugation, real adjoint, mass scale, source
permutation, refused normalization conflict/teacher/full-J property access,
actual FP32/extracted and complete feature traces, fixed tiny singular cutoffs,
induced-noise likelihood rather than Euclidean BP, map rescaling invariance,
correlated-noise coordinate invariance, float32 cancellation counterexample,
zero/noise-free cases and invalid layouts/noise/covariance.

Uncertainties requiring parent evidence: actual weak/ring singular spectra,
candidate information loss and real 12-channel/FP32 traces. No verdict has been
assigned. No additional worker numerical computation should be inferred from
this implementation handoff.
