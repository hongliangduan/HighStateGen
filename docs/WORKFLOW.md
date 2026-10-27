# JNJ-2113 workflow

Run commands from the repository root. Paths below identify inputs and output directories; replace them with your experiment paths. Training, sampling and analysis use the same topology, coordinate conventions and fixed state definitions.

## Prepare and train

Use four 12,500-frame trajectories. For JNJ-2113, each trajectory contributes its first 11,250 frames to training and its last 1,250 to validation.

`analysis/conformation_space.py fit` fits TICA using separate trajectory segments. `analysis/prepare_training_data.py` creates the 45,000-frame training and 5,000-frame validation inputs, condition statistics and split manifest. See each command's `--help` for input arguments.

```bash
python analysis/prepare_training_data.py --system jnj2113 \
  --coords-nm inputs/aligned_coords_nm.npy \
  --replica-ids inputs/replica_ids.npy --frame-indices inputs/frame_indices.npy \
  --tica-scores inputs/projection_scores.npy --tica-metadata inputs/metadata.json \
  --state-centers inputs/state_centers.npy --output-dir prepared
```

The aligned arrays and fixed-transform metadata can be produced by `analysis/conformation_space.py fit --export-training-inputs`. Supply the independent trajectories separately, or pass their lengths with `--trajectory-lengths` for concatenated input. The generated split manifest records frame membership and input hashes.

```bash
python src/train.py --config configs/jnj2113_manuscript.json \
  --data prepared/train_set.npz --val_data prepared/val_set.npz \
  --slow_feat_path prepared/condition_train_normalized.npy \
  --slow_feat_val_path prepared/condition_validation_normalized.npy \
  --train_basin_labels prepared/basin_labels_train.npy \
  --split_manifest prepared/split_manifest.json \
  --topology examples/jnj2113/complex_dry.prmtop \
  --epochs "$EPOCHS" --seed "$TRAIN_SEED" --output_dir runs/full
```

Set `EPOCHS` and `TRAIN_SEED` for the experiment. Training writes `best_model.pt`, saved checkpoints, arguments and run metadata. To continue a compatible checkpoint, add `--resume path/to/checkpoint.pt --finetune_extra_epochs "$EXTRA_EPOCHS"` and choose a new output directory. `--reset_scheduler` restarts the learning-rate schedule. The JNJ-2113 profile uses `translation_scale_mode=coordinate_rms`, `equivariant_translations=false`, and `topology_graph_scope=complex`, matching the released inference weights. The trainer checks these settings before training and when loading a resumed checkpoint. Ablation flags are `--disable_tica_conditioning`, `--disable_topology_bias` and `--disable_interface_attention`.

## Export the inference bundle

The complete matching training NPZ is read once to obtain fixed coordinate statistics and structural templates. Select the parameter dictionary used in the experiment: `model_state_dict`, `active_state_dict` or `ema_state_dict`.

```bash
python tools/export_inference_bundle.py \
  --checkpoint inputs/best_model.pt --weights-key model_state_dict \
  --data inputs/train_set.npz \
  --topology inputs/complex_dry.prmtop \
  --condition-stats inputs/tica_stats.npz \
  --sampling-config inputs/sampling_config.json \
  --tica-dir inputs/fixed_tica --output weights/jnj2113
```

Use the checkpoint's matching sampling configuration with the schema in `configs/jnj2113_manuscript.json`. The topology requires companion coordinates during export. The exporter validates model shapes and records input hashes. Older checkpoint formats use `--allow-legacy-checkpoint` for export and `--allow_legacy_checkpoint` for sampling, preserving their recorded model settings.

The output contains `inference.pt`, `inference_context.pt`, topology, condition statistics, sampling settings, supplied TICA files and an export manifest. The training NPZ and optimizer state remain outside the inference bundle.

## Generate a target-state ensemble

Set `TIC1` and `TIC2` to the selected target in the model's fixed TICA coordinates.

```bash
python src/sample.py \
  --checkpoint weights/jnj2113/inference.pt \
  --inference_context weights/jnj2113/inference_context.pt \
  --slow_feat "$TIC1" "$TIC2" --n_samples 100 --seed 42 \
  --output samples.pdb
python tools/split_samples.py samples.pdb --expected-count 100 --output-dir samples
```

The sampler reads topology, condition statistics and sampling settings beside the context file. It writes structures and a `.sampling.json` record. The original `--data` mode remains available for runs using the complete NPZ.

## Refine and evaluate

`graft_and_minimize.py` uses a donor trajectory, its frame-matched TICA scores, the fixed transform and topology to reconstruct and minimize all-atom structures. Use `--help` for its arguments. `analysis/conformation_space.py project` projects structures into the fixed state coordinates.

`analysis/reference_metrics.py` provides receptor-aligned reference-distance calculations with explicit alignment atoms, measured atoms and reference-frame membership. RMSD is the square root of the mean squared Euclidean distance per atom, expressed in the input coordinate units. Retain sample IDs through generation, refinement and evaluation.

## Tests

```bash
python -m unittest discover -s tests -v
```

The inference regression test exports a miniature model using synthetic frames, removes the NPZ input and compares the resulting PDB with direct-input sampling. Other tests check numerical behavior, frame membership, condition normalization and reference-distance definitions.
