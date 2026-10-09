# A28 E1 infrastructure and read-only asset audit

Status: infrastructure complete; no physics jobs or locked input access by this worker.

## APIs and owned source

- `src/a28/budget.py`: independent `results/a28/jobs` restore, `charged_job`, `BudgetLedger`, `write_json`, `budget_history` and ownership-checked locks.
- `book.reserve("full_wave_material_states")` is required **before** every attempted exact material state. Global hard cap 768, durable fsync reservation, no refund after failures, retries inherit costs. Optional lower cap is `full_wave_material_states_max`. A caller cannot raise it. Cache reuse does not re-reserve an already paid state.
- Category aliases `full_wave_states`, `material_state_evaluations`, `full_wave_material_state` share the same global cap. New training-label aliases remain cap zero.
- Conservative cumulative CPU/GPU caps are 3600 seconds each, with 15/30-second stop reserves. CPU jobs charge max(wall, process CPU), CUDA jobs charge all occupied wall plus host process CPU. Foreign locks and unresolved interrupted manifests block new work. Receipt immutability is retained.
- `tools/a28_driver.py`: audit/micro/freeze/anatomy/headroom/posterior/bp/report/tests/theory. Default config `a28_config.json`; pipeline call signature `(root, data_root, a26_root, config, book, device)`.
- Built-in `theory` copies the supplied source into a new job directory and runs it there; the original package output is not overwritten. Built-in `tests` retains all inherited tests and explicit skips.
- `tools/a28_remote.py`: actions preflight/inventory/push/run/pull. Reuses existing private transport without logging, copying or printing credentials. Remote root `D:/AI/A28_E1_MATERIAL_INFORMATION`, shared GPU lock `D:/AI/A20_OPM_IMAGING/runs/gpu.lock`.
- Source and accepted A28 receipts only are uploaded; no train/locked payload transport. Draft config may change until `frozen`/`protocol_frozen` or FROZEN_PROTOCOL receipt is set. A frozen config mismatch then rejects deployment.
- Normal run: `python tools/a28_remote.py run --job a28-JOB --stage STAGE --device cuda --config a28_config.json --timeout SECONDS --transport-helper EXISTING_HELPER --private-config EXTERNAL_PRIVATE_CONFIG`. Push source first. CLI wrappers create paid local transport receipts; native jobs and controller tails are measured separately.

## Live environment, bounded read-only preflight

- Python 3.10.9; NumPy 2.2.6; SciPy 1.15.3; Torch 2.6.0+cu124.
- RTX 4060 Laptop, 8188 MiB total; 0 MiB occupied and 0% utilization at preflight.
- Queue query succeeded; no compute processes; shared lock absent.
- All declared A25/A26/A27 exposed assets existed. This confirms availability, not numerical correctness.
- Full preflight and native accounting: `results/a28/transport/a28-transport-preflight-577a23145172.json`, and matching `results/a28/jobs/a28-transport-preflight-*/` receipt.

## Read-only cache header/identity inventory

Inventory read NPY headers and identity metadata only. It did not materialize chi/current/clean arrays, run Maxwell, or read locked data.

| Asset | Actual rows / dimensions | Parent scope |
|---|---|---|
| A25 TRAIN | 512; data/clean [512,32,8], chi [512,576], current [512,576,8], all complex128 | 250001–250064, 64 TRAIN parents |
| A25 development validation | 128; data/clean [128,32,8], chi [128,576], current [128,576,8], complex128 | 250065–250080, 16 exposed parents |
| A26 N3 TRAIN cache | features [512,12,24,24] float32; base_mass [512,1152] float64 | same TRAIN parents |
| A26 N3 validation cache | features [128,12,24,24] float32; base_mass [128,1152] float64 | same exposed parents |
| A26 prediction archive | N1/N3/COMMON_BP_INITIAL [128,576] complex128, case/parent metadata present | same exposed parents |
| A27 corrected predictions | N1/N3 × NONE/L/F/TASK_R32 [128,576] complex128 | aligned via case IDs; inventory does not claim numerical alignment |

Both backgrounds weak/ring, amplitudes 0.3/1.2; family metadata is shell/asymmetric_piecewise/smooth/mixed_scale. Exact validation family ranges are shell 250065–250068, asymmetric 250069–250072, smooth 250073–250076, mixed-scale 250077–250080; eight observations per parent are present. Clean/current labels are reusable existing assets, not newly generated labels.

- Successful inventory: `results/a28/transport/a28-transport-inventory-eaea5813df2d.json`.
- Successful inventory measured remote wall 0.0825563 s and process CPU 0.0625 s, in addition to the paid local transport wall. These are header-audit costs, not imaging latency.
- Initial inventory result `a28-transport-inventory-999ab9d04fc9.json` returned sanitized FAILED_REMOTE_BOOTSTRAP: predictions archive contains parent metadata, and the generic "predictions" role was incorrectly rejected. Fixed to the explicit validation whitelist; bounded retry passed. Initial failure is preserved. Its old local receipt says COMPLETE because the pre-existing CLI did not escalate returned bootstrap status; the failed result artifact must remain in the failure ledger, not be silently excluded. CLI now escalates FAILED_REMOTE_BOOTSTRAP after retaining its sanitized result, so future native receipts are FAILED.

## Verification and limits

- `a28-infrastructure-tests-001`: 18 new budget/driver/transport tests passed, no skips. Covers durable cross-retry cap and failed attempted-state accounting, lower-cap and overshoot rejection, history independence, interruption receipts, ownership loss, timeout/deployment scopes, active inherited test ledger, and dispatch failures.
- `a28-theory-001`: 14 supplied algebra assertions passed. Output and source copy are in `results/a28/jobs/a28-theory-001/`; original theory package JSON unchanged. This is toy algebra, not Maxwell experimental evidence.
- No source matrices, field values, model accuracy or actual checkpoint predictions were compared by the header audit. Parent must perform paid backend/packing/adjoint and checkpoint checks.
- No additional remote scans or experiments will be launched by this worker. All scientific candidate generation, cohort freeze, gates and interpretation belong to the parent.
- No SHA256 computation, publication, model training or locked ID/OOD read occurred.
