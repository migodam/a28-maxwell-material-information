# A28-E1 structural editing handoff

Implemented `src/a28/edits.py` and `tests/test_a28_edits.py`. No dataset was opened, no Maxwell action was executed, and no labels were generated. All tests use synthetic 12×12 fixtures and are numerical/unit evidence only.

## API

```python
from a28.edits import calibrate_train_rules, make_structural_edits

rules, train_audit = calibrate_train_rules(
    train_chi, aligned_known_background,
    split="train", parent_ids=train_parent_ids,
    amplitudes=train_amplitudes, grid=(24, 24),
    reference_amplitude=1.2,
)
# Parent writes rules/audit into the immutable pre-development protocol.
edits = make_structural_edits(
    center_chi, known_background, rules,
    center_provenance="ONLINE_CNN_PREDICTION",
)
```

Use `OFFLINE_TRUTH_DIAGNOSTIC` for truth-centered counterfactuals. Changing this role **does not change the produced candidates**; it only records permissions. Inputs are absolute physical complex χ, flat native C order or two-dimensional grid, never mass coordinates. Native A2 axes are `meshgrid(indexing='ij')`. Background must be aligned with each material row during TRAIN calibration.

Returns four `MaterialEdit` objects in fixed order: `CAVITY`, `BOUNDARY`, `COMPONENT`, `CONTRAST_ONLY`. Fields are `.chi`, `.change`, `.primary_change`, `.compensation_change`, `.primary_mask`, `.compensation_mask`, `.status`, `.metadata`, and `.valid`; `.summary()` is JSON safe. `.change`/`.delta` mean candidate minus center, **not** candidate minus background. Output arrays are flat. `include_contrast=False` gives only three structural candidates.

## Frozen conventions

- TRAIN-only parent whitelist 250001–250064, rejected development/locked roles, no file loader or physics inside this module.
- Default fixed phase `(1+.2j)/abs(1+.2j)`; scalar support is the center perturbation projected along that phase.
- Scale: median peak amplitude per TRAIN parent, then median across parents; if amplitudes are passed, only prespecified amplitude 1.2 is used. Threshold is `0.25×scale`.
- Cavity: fill the largest existing enclosed threshold-support hole; otherwise open a radius-one interior disk at the largest distance-transform site, deterministic C-order ties. No eligible interior means `INVALID_ZERO_EDIT`.
- Boundary: fixed radius-one dilation. A preregistered erosion option exists but is not selected based on reconstruction outcomes.
- Component: remove the smallest disconnected component of at most five cells; otherwise add a separated radius-one component nearest a fixed `+first-axis` offset from support centroid. Geometry determines this frozen branch; no family, truth boundary or error-direction input exists.
- Structural edits receive separately logged common-support contrast compensation. It cannot touch/undo primary edit cells. Uniform scalar compensation is limited by `0.25×scale` and the exact absolute box. Primary and compensation changes are separately returned.
- Contrast-only: fixed factor `.85` on the complete background perturbation; identifiability-only diagnostic and never eligible as an online primary candidate.
- Absolute material box matches A24–A27: `Re χ∈[-.5,4]`, `Im χ∈[0,.5]`; tolerance `1e-10`. Fixed-phase centers/edits preserve phase. Arbitrary legal CNN complex centers are also accepted; replacing primary pixels uses the known frozen phase, while untouched CNN residuals are retained.
- Empty/zero/box-blocked edits remain explicit invalid rows with no resampling. Parent should retain these in denominators and report missing candidate support rather than silently dropping objects.
- Metadata scalar sums are **unweighted cell sums**; multiply by the physical cell area for actual integrals. Unit tests and builder do not treat these as mass-coordinate norms.

## Verification and cost

Command:

```bash
PYTHONPATH=src:tests ../../A25/a25-adaptive-om-feedback/.venv_a25/bin/python -m unittest tests.test_a28_edits -v
```

Fourteen tests passed: TRAIN isolation, parent-balanced freezing, online/offline rule identity, deterministic C order, exact changes, compensation masking, fixed phase, hole fill/open, dilation/erosion, component add/remove, empty zero-edit rejection, absent cavity sites, compensation cap/box, source-copy immutability, and illegal/unfrozen inputs.

Measured with `/usr/bin/time -p`: wall **0.35 s**, user **0.17 s**, system **0.04 s**. Charge **0.35 CPU s** conservatively. Tool-reported wall was 0.2393 s; the explicit process timing is retained as the conservative source. No GPU, physics or labels.

## Parent review needed

These are candidate conventions, not an identifiability theorem or an optimized inverse solver. The parent must write the TRAIN calibration before development, approve the fixed parameters scientifically, audit the actual role-specific dataset loader, and decide how CNN-centered candidates enter a common F/L search. Existing finite candidate coverage can be poor (e.g. small/thin objects with no interior cavity); invalid candidates must not trigger a search for a better edit. The compensation may hit its cap, and this is deliberately recorded rather than pretending all edits conserve scalar contrast exactly. No conclusion about E1 Go/No-Go is delegated here.
