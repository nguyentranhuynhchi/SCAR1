# Project: SCAR M2-Pro (High-Resolution & Anatomy-Guided Scar Attention)

## Architecture
SCAR M2-Pro addresses the fundamental limitation of M2/M2-Plus (scar signal attenuation at the 1/16 bottleneck) by introducing high-resolution anatomy-guided attention at skip connections (1/4 at 32x32, 1/8 at 16x16) and an asymmetric extreme-imbalance loss formulation.

```
Inputs:
  bSSFP (CINE)  ───► ResNet Encoder ───► Skip-0 (1/2, 64x64) ───► AGSA-0 (64x64) ─────┐
                                    ───► Skip-1 (1/4, 32x32) ───► AGSA-1 (32x32) ───┐ │
                                    ───► Skip-2 (1/8, 16x16) ───► AGSA-2 (16x16) ─┐ │ │
                                    ───► Bottleneck (8x8) ─────┐                  │ │ │
  LGE (PSIR)    ───► ResNet Encoder ───► Skip-0, Skip-1, Skip-2─┼─► Bottleneck ───┤ │ │
  T2w           ───► ResNet Encoder ───► Skip-0, Skip-1, Skip-2─┘   Decoupled CA  │ │ │
                                                                        │         │ │ │
                                                                        ▼         │ │ │
                                                               Decoder Block 3 ◄──┘ │ │
                                                                        │           │ │
                                            Aux Head (1/4) ◄── Decoder Block 2 ◄────┘ │
                                                                        │             │
                                                               Decoder Block 1 ◄──────┘
                                                                        │
                                                               Decoder Block 0 (1/1)
                                                                        │
                                                                        ▼
                                                          Segmentation Head (B, 4, 128, 128)
```

Key Architectural Principles:
1. **Anatomy-Guided Skip Attention (AGSA)**: Extracts a soft myocardial mask from CINE at skip levels (32x32, 16x16) to gate LGE, completely filtering out blood-pool and chest-wall false positives.
2. **Detail-Aware Difference / Lesion Enhancement (DFE)**: Applies multi-scale dilated convolutions to subtract local context from fine details, amplifying subtle scar hyper-intensities.
3. **Decoupled Cross-Attention Bottleneck**: LGE queries CINE anatomical keys without modality averaging, eliminating feature dilution.
4. **Deep Supervision**: 1/4 resolution auxiliary prediction head providing direct gradient flow to high-resolution skip layers.
5. **Loss Architecture (`M2ProLoss`)**: Class-weighted CE + Scar-heavy Present Dice + Binary Scar Focal Loss + Pathology Inclusion Penalty.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | SOTA Scar Mechanism Analysis | Synthesize Decoupled Experts, Anatomy ROI, DFE, SFAM, and Elasticity Field | M0 (Done) | Survey / R1 |
| 2 | Anatomy-Guided Skip Attention (AGSA) | CINE myocardial soft mask gating LGE features at 1/4 (32x32) and 1/8 (16x16) | M1 (Done) | Survey / R2 |
| 3 | Detail-Aware Difference Enhancement (DFE) | Multi-scale dilated conv subtraction (Local - Context) for scar edge enhancement | M1 (Done) | Survey / R2 |
| 4 | Decoupled Inverted Cross-Attention | Bottleneck CA with LGE query attending to CINE keys without feature averaging | M1 (Done) | Survey / R2 |
| 5 | Deep Supervision Auxiliary Head | Auxiliary 1/4 resolution segmentation head for high-res skip gradient flow | M1 (Done) | Survey / R2 |
| 6 | M2ProNet Model Registration | M2ProNet registered in `training/models/__init__.py` and `MODEL_REGISTRY` | M1 (Done) | Survey / R2 |
| 7 | Class-Weighted Cross Entropy | Background: 0.1, Myo: 1.0, Edema: 2.0, Scar: 3.0 to suppress background | M2 (Done) | Survey / R3 |
| 8 | Scar-Heavy Present Dice | Dice weights [0.15, 0.35, 0.50] evaluated only on slices with target presence | M2 (Done) | Survey / R3 |
| 9 | Binary Scar Focal Loss | One-vs-Rest Focal Loss (gamma=2.0, alpha=0.75) for sparse positive scar pixels | M2 (Done) | Survey / R3 |
| 10 | Pathology Inclusion Constraint | Penalty enforcing Scar <= Edema <= Myocardial Ring spatial containment | M2 (Done) | Survey / R3 |
| 11 | Auxiliary Deep Supervision Loss | Loss computation hook for 1/4 resolution auxiliary logits | M2 (Done) | Survey / R3 |
| 12 | M2-Pro YAML Configuration | `training/config/models/m2_pro.yaml` with optimized hyperparameters | M3 | Survey / R4 |
| 13 | SCAR Pipeline & Trainer Integration | Integration with AMP (bf16/fp16), gradient clipping, and metrics logging | M3 | Survey / R4 |
| 14 | Unit Test Verification Suite | Forward/backward passes, shapes, router grads, AMP safety, loss edge cases | M_TEST & M3 (Done) | Survey / AC2 |
| 15 | Opaque-Box E2E Testing Suite | Tiers 1-4 test suite derived from ORIGINAL_REQUEST requirements | M_TEST (Done)| Dual Track |
| 16 | 76 3D Test Volumes Evaluation | Voxel-grid evaluation on fixed 76 cases verified with SHA256 | M4 | Survey / R4 |
| 17 | Standardized Reporting Output | Generation of schema-compliant `metrics.csv`, `per_case.csv`, `metrics.json` | M4 | Survey / AC2 |
| 18 | Quantitative Benchmark Acceptance | Attain Scar > 60%, Edema >= 73.5%, Myo >= 87.5%, Avg Path >= 65-70% | M4 | Survey / AC1 |
| 19 | Adversarial Coverage Hardening | Tier 5 white-box edge case testing and robustness verification | M5 | Dual Track |
| 20 | Forensic Integrity Audit | Static analysis, runtime tracing, no cheating, no hardcoded results | M5 | Integrity Forensics |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M_TEST | E2E Testing Track | Requirement-driven test harness, Tiers 1-4 tests, TEST_READY.md | none | DONE |
| M1 | M2-Pro Architecture | AGSA, DFE, Decoupled CA, Aux Head, M2ProNet implementation | none | DONE |
| M2 | Extreme Imbalance Loss | M2ProLoss (Weighted CE, Present Dice, Scar Focal, Inclusion) | none | DONE |
| M3 | Pipeline & Config Integration | YAML config, trainer integration, unit test verification | M1, M2 | IN_PROGRESS |
| M4 | Benchmark Evaluation & AC | 76-volume evaluation, metrics.csv, per_case.csv, metrics.json, target scores | M3 | PLANNED |
| M5 | Adversarial Hardening & Audit | Tier 5 tests, Forensic Integrity Audit, Acceptance verification | M_TEST, M4 | PLANNED |

## Interface Contracts

### Model Architecture: `M2ProNet`
- **Inputs**:
  - `cine`: `torch.Tensor` of shape `(B, 1, H, W)` float32
  - `psir`: `torch.Tensor` of shape `(B, 1, H, W)` float32 (LGE)
  - `t2w`: `torch.Tensor` of shape `(B, 1, H, W)` float32
- **Outputs**:
  - Training mode: `M2ProOutput(main_logits, aux_logits)` where `main_logits` is `(B, 4, H, W)` and `aux_logits` is `(B, 2, H/4, W/4)` (edema + scar). Provides `.main_logits`, `.aux_logits`, dict indexing, and tuple unpacking.
  - Eval mode: `main_logits` of shape `(B, 4, H, W)`.
- **Label Order**: Canonical `[0:bg, 1:normal_myo, 2:edema, 3:scar]`

### Loss Function: `M2ProLoss`
- **Inputs**:
  - `logits`: `(B, 4, H, W)` float32 (or `M2ProOutput`)
  - `targets`: `(B, H, W)` int64 with values in `[0, 3]`
  - `aux_logits` (optional): `(B, 2, H/4, W/4)` float32
- **Outputs**:
  - `dict` containing `"loss"` (scalar `torch.Tensor`), `"ce"`, `"dice"`, `"focal_scar"`, `"inclusion"`, `"aux"`

### Benchmark Evaluation Contract
- **Input Cohort**: Fixed 76 cases from `preprocessing/splits/test_vol.txt` (SHA256: `3237b314...`)
- **Protocol**: `myops380_voxel_v1` on unit voxel grid (`voxelspacing=None`)
- **Outputs**:
  - `per_case.csv`: Exactly 380 rows, 17 columns
  - `metrics.json`: Hierarchical structure with 5 regions + summary metrics
  - `metrics.csv`: Per-epoch training/val progression

## Code Layout
- `training/models/modules/m2_pro.py`: Architecture components (`AGSA_Block`, `DFE_Block`, `DecoupledBottleneckFusion`, `DeepSupervisionAuxHead`, `M2ProOutput`) [DONE]
- `training/models/m2_pro.py`: `M2ProNet` class inheriting from `CMSPANet` [DONE]
- `training/models/__init__.py`: Model registry binding [DONE]
- `training/loss/m2_pro_loss.py`: `M2ProLoss` class [DONE]
- `training/config/models/m2_pro.yaml`: Model and loss configuration [M3]
- `tests/test_m2_pro.py`: Architecture unit tests [DONE, 15/15 passed]
- `tests/test_m2_pro_loss.py`: Loss unit tests [DONE, 15/15 passed]
- `tests/test_m2_adversarial_challenger.py`: Adversarial challenge tests [DONE, 20/20 passed]
- `tests/e2e/`: Opaque-box E2E test suite (Tiers 1-4) [DONE, 33/33 passed]
