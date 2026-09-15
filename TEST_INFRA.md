# Test Infrastructure Specification: SCAR M2-Pro

## 1. Test Philosophy

The SCAR M2-Pro test infrastructure implements a requirement-driven, opaque-box, 4-tier testing methodology designed to rigorously validate model behavior, numerical safety, interface contracts, and benchmark compliance for myocardial scar segmentation.

### Core Principles
1. **Opaque-Box Verification**: Tests validate observable external behaviors, inputs, outputs, schemas, and contract guarantees without coupling to internal private implementations.
2. **Requirement-Driven Derivation**: Every test case directly traces back to requirements R1-R4 and acceptance criteria AC1-AC2 in `ORIGINAL_REQUEST.md` and feature specifications in `PROJECT.md`.
3. **Authoritative Output Oracles**:
   - Canonical label mapping: derived from `ORIGINAL_REQUEST.md § R4` and `data_contract.py` (`0:bg, 1:normal_myo, 2:edema, 3:scar`).
   - Benchmark evaluation: derived from `myops380_voxel_v1` protocol and official reference heuristic (`I_MMSeg`).
   - Cryptographic data lock: authoritative SHA256 `3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc` for the fixed 76 3D test volumes.
   - Reporting schemas: exact 17-column `per_case.csv` and hierarchical `metrics.json` specifications.
4. **Progressive Testability & Contract Isolation**:
   - Tests are self-contained and isolated; each test sets up its own state and cleans up temporary directories.
   - For modules under active milestone development (`M2ProNet`, `M2ProLoss`), tests enforce the strict interface contracts defined in `PROJECT.md § Interface Contracts`. Tests dynamically adapt: when live modules are present, they are verified directly; when pending, interface contract compliance and baseline models are exercised.
5. **Adversarial & Numerical Robustness**:
   - Explicit guards against NaN/Inf loss propagation.
   - Extreme edge case validation: double empty masks, one-sided empty masks, all-zero predictions, all-foreground predictions, single-slice 3D volumes ($D=1$), and zero-size scars.

---

## 2. Feature Inventory Mapping

The test suite maps directly to the features defined in `PROJECT.md § Feature Inventory`:

| Feature # | Feature Name | Description | Test Tier | Test Module |
|---|---|---|---|---|
| F1 | SOTA Scar Mechanism Analysis | Knowledge synthesis and architectural grounding | Tier 1 | `test_tier1_feature_coverage.py` |
| F2 | Anatomy-Guided Skip Attention (AGSA) | High-resolution skip feature gating | Tier 1 & 3 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` |
| F3 | Detail-Aware Difference (DFE) | Lesion enhancement via multi-scale difference | Tier 1 & 3 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` |
| F4 | Decoupled Cross-Attention | Bottleneck cross-attention without modality averaging | Tier 1 & 3 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` |
| F5 | Deep Supervision Auxiliary Head | 1/4 resolution auxiliary prediction head | Tier 1 & 3 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` |
| F6 | M2ProNet Model Registration | Model registry binding and interface contract | Tier 1 | `test_tier1_feature_coverage.py` |
| F7 | Class-Weighted Cross Entropy | Background suppression weights [0.1, 1.0, 2.0, 3.0] | Tier 1 | `test_tier1_feature_coverage.py` |
| F8 | Scar-Heavy Present Dice | Dice evaluated strictly on slices with target presence | Tier 1 & 2 | `test_tier1_feature_coverage.py`, `test_tier2_boundary_corner.py` |
| F9 | Binary Scar Focal Loss | One-vs-rest focal loss for sparse positive scar pixels | Tier 1 & 2 | `test_tier1_feature_coverage.py`, `test_tier2_boundary_corner.py` |
| F10 | Pathology Inclusion Constraint | Penalty enforcing Scar $\le$ Edema $\le$ Myocardial Ring | Tier 1 & 3 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` |
| F11 | Auxiliary Deep Supervision Loss | Loss computation hook for 1/4 resolution auxiliary logits | Tier 1 & 3 | `test_tier1_feature_coverage.py`, `test_tier3_cross_feature.py` |
| F12 | M2-Pro YAML Configuration | Hyperparameters and configuration verification | Tier 1 | `test_tier1_feature_coverage.py` |
| F13 | SCAR Pipeline & Trainer Integration | AMP precision, gradient clipping, metrics logging | Tier 3 & 4 | `test_tier3_cross_feature.py`, `test_tier4_real_world.py` |
| F14 | Unit Test Verification Suite | Shapes, gradients, loss bounds, router gradients | Tiers 1-3 | All Tier test modules |
| F15 | Opaque-Box E2E Testing Suite | Requirements-driven 4-tier test runner | All | `tests/e2e/test_suite.py` |
| F16 | 76 3D Test Volumes Evaluation | Fixed 76 cases cohort locked with SHA256 | Tier 4 | `test_tier4_real_world.py` |
| F17 | Standardized Reporting Output | Exact 17-column `per_case.csv`, `metrics.json`, `metrics.csv` | Tier 4 | `test_tier4_real_world.py` |
| F18 | Quantitative Benchmark Acceptance | Threshold checking: Scar > 60%, Edema >= 73.5%, Myo >= 87.5% | Tier 4 | `test_tier4_real_world.py` |

---

## 3. Test Architecture

### Directory Layout
```
tests/
├── e2e/
│   ├── __init__.py
│   ├── test_tier1_feature_coverage.py     # Tier 1: Core feature coverage & contracts
│   ├── test_tier2_boundary_corner.py       # Tier 2: Extreme boundaries & numerical stability
│   ├── test_tier3_cross_feature.py         # Tier 3: Pairwise combinations & AMP / multi-task
│   ├── test_tier4_real_world.py           # Tier 4: End-to-end evaluation & reporting schemas
│   └── test_suite.py                      # Unified test runner with CLI & summary reporting
├── smoke_dpf_cuda.py
├── test_m3_dpf.py
└── test_run_dir.py
```

### Execution Framework
- **Test Engine**: Python `unittest` framework (standard library, zero external runner dependency).
- **Execution CLI**:
  - Full suite: `python tests/e2e/test_suite.py` or `python -m unittest discover tests/e2e`
  - Tier-by-tier: `python tests/e2e/test_suite.py --tier 1` (or `2`, `3`, `4`)
  - Verbose discovery: `python -m unittest -v tests/e2e/test_tier1_feature_coverage.py`
- **Isolation**: Each test operates on isolated synthetic memory buffers or `tempfile.TemporaryDirectory()`, leaving no artifacts or side effects on disk.

---

## 4. Tier 1-4 Coverage Goals

### Tier 1: Feature Coverage
- **Canonical Label Normalization**:
  - Verify `canonicalize_label` preserves canonical mapping `[0:bg, 1:normal_myo, 2:edema, 3:scar]`.
  - Verify `legacy` translation vector `[0, 1, 3, 2]` correctly flips scar and edema into canonical order.
  - Reject invalid label inputs (non-integers, negative class IDs, out-of-bound labels > 3).
  - Verify immutability of input segmentation arrays.
- **Multimodal Input Handling**:
  - Model forward pass accepts 3 separate tensors: `bSSFP`, `LGE`, `T2w`, each with shape `(B, 1, H, W)`.
  - Rejects misaligned shapes or missing modalities.
- **Model Output Tensor Shapes**:
  - Training mode: outputs `(main_logits, aux_logits)` where `main_logits` is `(B, 4, H, W)` and `aux_logits` is `(B, 2, H/4, W/4)` (edema + scar).
  - Eval mode: outputs `(B, 4, H, W)`.
- **Loss Computation on Valid & Empty Slices**:
  - Evaluates on valid multi-class targets and empty slices without zero-division.
  - Verifies finite gradients (`torch.isfinite(grad).all()`).
- **5-Region Benchmark Metrics**:
  - Verifies computation for all 5 canonical regions: `normal_myocardium`, `edema`, `scar`, `edema_inclusive`, `myocardial_ring`.
  - Verifies primary selection metric formula: $\text{avg\_pathology\_dice} = \frac{\text{mean\_dice}(\text{scar}) + \text{mean\_dice}(\text{edema})}{2}$.

### Tier 2: Boundary & Corner Cases
- **Extreme Empty Masks**:
  - Both prediction and target empty ($n_P = 0, n_T = 0$):
    - Primary benchmark: `status: "both_empty"`, `dice: None`, `iou: None`, `hd95: None`, `asd: None`. Excluded from primary case mean.
    - Official reference: `official_dice: 1.0`, `official_hd95_voxel: 0.0`.
  - Target positive, prediction empty ($n_P = 0, n_T > 0$):
    - Primary: `status: "prediction_empty"`, `dice: 0.0`, `iou: 0.0`, `precision: None`, `recall: 0.0`.
  - Target empty, prediction positive ($n_P > 0, n_T = 0$):
    - Primary: `status: "target_empty"`, `dice: 0.0`, `iou: 0.0`, `precision: 0.0`, `recall: None`.
- **All-Zero & All-Foreground Extreme Predictions**:
  - Complete background prediction: precision is `None`, recall is 0.0 for lesions.
  - Complete foreground prediction: recall is 1.0, precision is small positive fraction, Dice is bounded in $[0, 1]$.
- **Numerical Safety & NaN/Inf Protection**:
  - Input arrays containing `NaN` or `Inf` are rejected with `ValueError`.
  - Loss values are verified finite across empty microbatches.
  - Gradient norm clipping and skipped steps under overflow.
- **Zero-Size Scars & Sparse Lesions**:
  - Target volumes with 0 scar voxels evaluate cleanly without crashing distance transforms.
  - Isolated single-voxel lesions ($N=1$) calculate finite Hausdorff distances.
- **Single-Slice 3D Volumes ($D=1$)**:
  - Volume inference and surface distance metrics execute correctly on 2D-like volumes with shape `(H, W, 1)`.

### Tier 3: Cross-Feature Combinations
- **AMP Precision with Gradient Scaling**:
  - Mixed precision forward pass under `bfloat16` and `float16`.
  - Integration with `torch.amp.GradScaler`, testing gradient scaling, unscaling, and detection of non-finite gradients.
- **Multi-Term Loss with Deep Supervision**:
  - Joint optimization of class-weighted CE, present Dice, scar focal loss, inclusion penalty, and 1/4 resolution auxiliary loss.
  - Gradient propagation to both backbone skip connections and auxiliary head.
- **Decoupled Bottleneck + Anatomy-Guided Skip Connections**:
  - Multimodal inputs passing through decoupled cross-attention and high-resolution skip gating.
  - Gradient backpropagation verified across all 3 input modalities.
- **Microbatch Scaling & Gradient Accumulation**:
  - Loss normalization by accumulation steps (`accum_steps=2`, `4`).
  - Correct optimizer update cadence and step counting.

### Tier 4: Real-World Application Scenarios
- **End-to-End 3D Volume Evaluation Pipeline**:
  - Complete execution of volume prediction, resizing, and metric extraction on sample 3D test cases.
- **Schema Compliance for `per_case.csv`**:
  - Exactly 17 columns:
    `case,region,dice,iou,hd95,asd,status,precision,recall,prediction_voxels,target_voxels,hd95_unit,hd95_voxel,asd_voxel,official_dice,official_hd95_voxel,inference_seconds`.
  - Exactly 5 rows per evaluated case (one per region).
- **Schema Compliance for `metrics.json`**:
  - Valid JSON without NaN (`allow_nan=False`).
  - Contains all 5 region dictionaries, summary pathology metrics, protocol metadata, and timing info.
- **Incremental Training Metric Logging (`metrics.csv`)**:
  - Per-epoch row appending via `append_metrics_csv`.
  - Header schema auto-expansion on new metric keys without data corruption.
- **Official Benchmark Cryptographic Verification**:
  - Verification that the official 76 test patient IDs match SHA256 `3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc`.
  - Strict validation that cross-split patient overlap raises `ValueError`.

---

## 5. Execution and Verification Guide

### Quick Run Commands
```bash
# Run the entire E2E test suite
python tests/e2e/test_suite.py

# Run a specific tier
python tests/e2e/test_suite.py --tier 1
python tests/e2e/test_suite.py --tier 2
python tests/e2e/test_suite.py --tier 3
python tests/e2e/test_suite.py --tier 4

# Run via unittest discovery
python -m unittest discover tests/e2e
```
