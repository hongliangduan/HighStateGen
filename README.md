# HighStateGen

**State-Directed SE(3) Flow Matching for Ensemble Generation of Protein–Cyclic Peptide Complexes**

HighStateGen uses conformational-state conditions to guide the generation of protein–cyclic peptide complex ensembles. The model combines residue-frame flow matching, topology-aware representations and protein–peptide interface attention. Fixed TICA coordinates specify the target state, while geometry guidance supports structured sampling.

This source package contains the JNJ-2113 implementation, training and sampling entry points, analysis utilities and a portable inference exporter.

## Installation

Use Python 3.11 and a PyTorch build suitable for your CPU or CUDA environment:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Workflow

1. Define conformational states in a fixed TICA coordinate system.
2. Prepare training and validation inputs and train the conditional flow model.
3. Select a target state and generate complex structures.
4. Reconstruct all-atom structures, minimize and evaluate their conformations.

[Workflow commands](docs/WORKFLOW.md) cover preparation, training, inference export and sampling.

## Contents

| Path | Purpose |
| --- | --- |
| `src/` | Model, geometry losses, training and sampling |
| `analysis/` | TICA, data preparation and reference-distance calculations |
| `configs/jnj2113_manuscript.json` | JNJ-2113 implementation settings, including the 45,000/5,000 training/validation split |
| `tools/` | Portable inference export and sample splitting |
| `graft_and_minimize.py` | Donor-based all-atom reconstruction and restrained minimization |
| `examples/jnj2113/` | Topology, reference structure and closure-atom mapping |
| `tests/` | Numerical, data-split and inference-export checks |

## Training

The JNJ-2113 profile uses four trajectories with 12,500 frames each. The first 11,250 frames of each trajectory form the training set and the final 1,250 form the validation set. The profile uses coordinate-component RMS scaling, translation outputs in the aligned coordinate frame, and a topology graph spanning the complex, matching the released inference weights. The trainer includes state-weighted sampling, geometry losses, EMA tracking, validation-based checkpoint selection and compatible-checkpoint continuation. See [training commands](docs/WORKFLOW.md).

## Portable inference

An exported inference bundle packages the selected weights, coordinate and condition normalization, structural templates, topology and sampling settings. Subsequent sampling reads this bundle directly. Target selection and projection use the corresponding fixed TICA transform and state definitions.

The exporter records the weight dictionary, source hashes and model settings in `export_record.json`. Keep these records with the bundle to identify the model used for each run.
