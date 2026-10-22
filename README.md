# HighStateGen

State-directed SE(3) flow matching for protein–cyclic peptide conformations.

## Scope

This submission-stage source release contains the principal IL-23R–JNJ-2113 model, data preparation, training, sampling, and donor-graft/OpenMM refinement code. It is assembled from the available local source and the supplied reconstruction implementation. It has not been rerun end to end to reproduce the manuscript's hit rates or RMSDs.

The trajectories, parameterized topology, training arrays, fitted TICA transform, ordered state centers, and checkpoints have been located in the authors' server archive but are **not distributed in this package**. State ordering follows the archived `data_tica50000/basin_labels.json`. The full KRAS–LUNA18 adaptation exists separately and is not included here. The final manuscript evaluator and its version-matched reference structures, interface selections, and atom mappings are also outside this release.

## Files

| Path | Contents |
|---|---|
| `src/ipa_model_dual.py`, `src/topo_graph_encoder.py` | Conditioned dual-branch model, topology features, and interface attention |
| `src/frame_flow_matching.py`, `src/frame_utils.py` | Frame interpolation, velocity targets, and SE(3) operations |
| `src/physics_losses.py`, `src/lddt_loss.py`, `src/physics_constraints.py` | Structural losses and geometry utilities |
| `src/train.py`, `src/sample.py` | Training and Heun generation entry points |
| `src/data_protocol.py`, `src/frame_dataset_dual.py`, `src/configuration.py` | Data partition, normalization, and configuration loading |
| `analysis/` | TICA analysis and preparation of the fixed training/validation partition |
| `configs/jnj2113_manuscript.json` | JNJ-2113 model and method settings |
| `graft_and_minimize.py` | Donor selection, residue-wise atom transfer, mapping records, and OpenMM minimization |

## Method settings

For JNJ-2113, replicas 1, 3, and 4 supply 30,000 training frames and 7,500 validation frames; replica 2 contains 12,500 frames excluded from network optimization. Each development replica contributes its final 2,500 frames to validation. The fixed TICA transform uses the full ensemble, with lagged pairs restricted to individual replicas. Condition normalization and the coordinate scale use training frames only.

Network dropout is 0.2 and CFG condition dropout is 0.15. Generation geometry-guidance weights are 1.0 for protein translation, 1.0 for peptide translation, 0.5 for peptide rotation, and 10.0 for Cα–Cα distances.

The three ablation switches in the trainer disable TICA conditioning, topology bias, or interface attention. The sampler reads these switches from the checkpoint. Removing TICA conditioning retains flow-time encoding; removing topology bias retains geometry losses and refinement; removing interface attention retains both within-branch modules.

## Entry points

Use Python 3.11/3.12 and install `requirements.txt` with an appropriate PyTorch CPU/CUDA build. Run each entry point with `--help` for its input arguments:

```text
python analysis/prepare_training_data.py --help
python src/train.py --config configs/jnj2113_manuscript.json --help
python src/sample.py --help
python graft_and_minimize.py --help
```

Refinement requires one generated conformation per PDB, the matching dry topology and donor trajectory, and the archived TICA transform/alignment. The release entry point projects the **actual generated coordinates** and selects the nearest donor in the same TICA coordinates. It fits each mapped residue using N/Cα/C, transfers donor atoms, and retains donor coordinates for unmapped caps. Each run writes donor indices, residue/atom mappings, and failures. These new records do not replace the original experiment's mapping records, which are not included.

OpenMM uses OBC2/NoCutoff with constraints on bonds involving hydrogen atoms, Cα restraints of `0.5*k*distance²` with k = 10 then 2 kcal mol⁻¹ Å⁻², up to 1,000 iterations per stage, and a force tolerance of 10 kJ mol⁻¹ nm⁻¹. The supplied implementation's SG–SG restraint is retained. This refinement entry point does not calculate the manuscript's final QC or reference-based evaluation tables.

## Availability and rights

This package contains source code only. No open-source license has been designated for this submission-stage release. Any wider data, software, or licensing availability should be stated separately in the manuscript's Data and Software Availability Statement.
