# E1 saved-evidence artifacts handoff

Implementation: `src/a28/artifacts.py`.
Entry: `report_artifacts(root, config, book=None)`; the parent already calls it
from `pipeline.stage_report`. No physics, dataset import, checkpoint execution,
training, publication or scientific verdict is performed.

**Execution status:** artifact worker did not launch numerical jobs or tests.
Ten tiny aggregation tests in `tests/test_a28_artifacts.py` await the parent's
paid test driver. BP's previous 14 tests remain unchanged.

## Inputs and outputs

Only final files in `results/a28/{anatomy,headroom,posterior,bp}` are read.
`*_PARTIAL.csv` is never substituted for a missing final file. `config['cohort']`
must be the frozen metadata list; grouping supports the cost-only balanced
prefix if used. Alias `bank` is normalized to `role`; max weight accepts either
`max_weight` or `maximum_weight`.

Outputs are under `results/a28/artifacts/`:

- preserved raw CSVs, strict parent/group tables for headroom, counterfactuals,
  baseline spectra/errors, posterior distributions/decisions and CNN response;
- `HEADROOM_PAIRED_STATISTICS.json/.csv`, per-parent gains, and finite-bank
  `POSTERIOR_DECISION_PAIRED.json/.csv` with oracle/data-only roles separated;
- `SUMMARY.json`, `FIGURE_MANIFEST.json`, raw plot CSVs, PNG and SVG figures.

Main figures: counterfactual distances vs structural tasks; complex-family
paired gains and update-quality/cost; kappa ratio bias vs conditional risk/ESS;
BP spectrum and covariance-whitened distance preservation; real/imag actual
truth/N3/L6/F6 images. Images use the first frozen parent of every family and
both backgrounds. All panels share their row's material scale. Missing k6 is
marked missing; it is never replaced by a previous successful image.

## Statistical and cost limits

Two backgrounds are averaged within each parent before equal-parent means or
relative gains. F6/L6, F6/N3 and F6/F1 are explicitly compared; F1/N3 and L6/N3
are descriptive additional rows. Paired bootstrap is fixed at 2,000 draws over
parents, with complex families as primary. Missing, invalid or duplicated
conditions null the primary point and interval; no success-only mean is emitted.
Valid rejected updates remain in the trajectories.

Core moment quantities can remain valid when an optional tiny-risk ratio is
undefined. Optional means remain null if any required paired condition lacks
that quantity. No posterior calibration claim follows from moment identities.
Truth-centered finite-bank decisions remain **OFFLINE ORACLE ONLY**.

Headroom time is explicitly **cumulative update wall time**: background setup,
N3 starting prediction and shared initial-state preparation are excluded by the
pipeline and remain separately accounted in its cost ledger. The figures make
no 40 ms deployment or acceleration claim. All plotting costs belong to the
caller-owned report job.

## Tiny tests (not worker-run)

Frozen metadata/paths, condition averaging, invalid/duplicate/unexpected rows,
valid rejection and blank union-CSV flags, optional undefined risk metrics,
parent-first gain vs ratio-of-means, deterministic bootstrap, missing pair
nulls, fixed sample count and absent final files. Tests use temporary arithmetic
fixtures; they do not create synthetic scientific plots or run Maxwell.

No gate or final scientific interpretation is included. The parent must inspect
the real outputs, fixed images, failure records and actual cost before judging
E2/E3 eligibility.
