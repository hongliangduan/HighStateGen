#!/usr/bin/env python3
"""Export a JNJ-2113 inference bundle using the matching complete training NPZ once."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]
import numpy as np
import torch
from data_protocol import sha256_file
from checkpoint_protocol import validate_checkpoint_protocol, verify_inference_inputs
from configuration import MODEL_KEYS, sampling_defaults
from inference_context import collect_context, load_context, _portable

def export_bundle(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new output directory: " + str(out))
    paths = [args.checkpoint, args.data, args.topology, args.sampling_config]
    if args.condition_stats:
        paths.append(args.condition_stats)
    for filename in paths:
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    if Path(args.topology).suffix.lower() != ".prmtop":
        raise ValueError("Provide the AMBER .prmtop topology used by this implementation")
    if args.tica_dir and not Path(args.tica_dir).is_dir():
        raise FileNotFoundError(args.tica_dir)
    sampling_defaults(args.sampling_config)
    # Checkpoints are trusted author files. Portable exported files contain tensors and builtins.
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = source.get("args")
    if model_args is None:
        config_path = args.model_args or str(Path(args.checkpoint).parent / "args.json")
        model_args = json.loads(Path(config_path).read_text(encoding="utf-8"))
    elif not isinstance(model_args, dict):
        model_args = vars(model_args)
    run_metadata = source.get("run_metadata") or {}
    status = validate_checkpoint_protocol(model_args, run_metadata, args.allow_legacy_checkpoint)
    if status == "manuscript_configured_run":
        verify_inference_inputs(run_metadata, args.data, args.topology, args.condition_stats)
    selected = source.get(args.weights_key)
    if not isinstance(selected, dict) or not selected:
        raise ValueError("Selected parameter dictionary is absent or empty")
    weights = {}
    for key, value in selected.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("Invalid model tensor: " + str(key))
        weights[key] = value.detach().cpu().clone()
    dim = int(model_args.get("slow_feat_dim", 0))
    if dim and not model_args.get("disable_tica_conditioning", False) and not args.condition_stats:
        raise ValueError("Provide the condition normalization used by this model")
    if args.condition_stats:
        with np.load(args.condition_stats, allow_pickle=False) as stats:
            mean, std = stats["mean"], stats["std"]
            if mean.shape != (dim,) or std.shape != (dim,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("Invalid condition normalization")
    from frame_dataset_dual import ProteinPeptideFrameDataset
    dataset = ProteinPeptideFrameDataset(
        args.data, args.topology,
        translation_scale_mode=model_args.get("translation_scale_mode", "coordinate_rms"))
    if (dataset.n_protein, dataset.n_peptide) != (294, 13):
        raise ValueError("This exporter supports the supplied JNJ-2113 implementation; use the KRAS adapter for LUNA18")
    context = collect_context(dataset, args.topology, bool(model_args.get("cyclic_peptide", False)))
    from ipa_model_dual import ProteinPeptideFrameDenoiser
    model = ProteinPeptideFrameDenoiser(
        n_protein=dataset.n_protein, n_peptide=dataset.n_peptide,
        **{k: model_args.get(k, v) for k, v in {
            "c_s":256,"c_z":64,"n_layers":6,"condition_dim":128,
            "n_heads":8,"slow_feat_dim":0,"n_topo_heads":2,"khop":3,
            "cyclic_peptide":False,"disable_tica_conditioning":False,
            "disable_topology_bias":False,"disable_interface_attention":False,
            "equivariant_translations":False,"topology_graph_scope":"complex"}.items()},
        dropout=0.0, topo_path=args.topology, peptide_residue_ids=dataset.peptide_residue_ids)
    model.load_state_dict(weights, strict=True)
    keep = set(MODEL_KEYS) | {
        "translation_scale_mode", "disable_tica_conditioning", "disable_topology_bias",
        "disable_interface_attention", "manuscript_protocol", "finetune_extra_epochs"}
    portable_args = {k: _portable(v) for k, v in model_args.items() if k in keep}
    if model_args.get("resume"):
        portable_args["resume"] = str(model_args["resume"]).replace("\\", "/").rsplit("/", 1)[-1]
    provenance = {
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "training_input_sha256": sha256_file(args.data),
        "weights_key": args.weights_key, "epoch": source.get("epoch"),
        "training_frame_count": len(dataset), "checkpoint_protocol_status": status,
        "original_initialization": "resumed_checkpoint" if model_args.get("resume") else "see_training_record",
        "source_code_sha256": {name: sha256_file(ROOT / "src" / name) for name in (
            "sample.py", "frame_dataset_dual.py", "frame_utils.py", "physics_losses.py",
            "physics_constraints.py", "ipa_model_dual.py", "inference_context.py")},
    }
    # All input/shape checks precede creation of the destination.
    out.mkdir(parents=True)
    checkpoint = out / "inference.pt"
    torch.save({
        "format": "highstategen_inference_weights_v1",
        "model_state_dict": weights, "args": portable_args,
        "run_metadata": _portable(run_metadata), "export_provenance": _portable(provenance),
    }, checkpoint)
    shutil.copy2(args.topology, out / "complex_dry.prmtop")
    companion = Path(args.topology).with_suffix(".pdb")
    if companion.is_file():
        shutil.copy2(companion, out / "complex_dry.pdb")
    shutil.copy2(args.sampling_config, out / "sampling_config.json")
    if args.condition_stats:
        shutil.copy2(args.condition_stats, out / "condition_stats.npz")
    if args.tica_dir:
        tica_dir = Path(args.tica_dir)
        if not tica_dir.is_dir():
            raise FileNotFoundError(tica_dir)
        (out / "tica").mkdir()
        for name in ("pca_model.pkl", "metadata.json", "align_ca_positions.npy",
                     "reference_ca.pdb", "state_centers.npy", "pca_scores.npy",
                     "projection_scores.npy", "basin_labels.json"):
            if (tica_dir / name).is_file():
                shutil.copy2(tica_dir / name, out / "tica" / name)
    context["input_sha256"] = {
        "checkpoint": sha256_file(checkpoint),
        "topology": sha256_file(out / "complex_dry.prmtop"),
        "condition_stats": sha256_file(out / "condition_stats.npz") if args.condition_stats else None,
    }
    context["source"] = _portable(provenance)
    torch.save(context, out / "inference_context.pt")
    load_context(out / "inference_context.pt", checkpoint, out / "complex_dry.prmtop",
                 out / "condition_stats.npz" if args.condition_stats else None, portable_args)
    (out / "export_record.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = [{"path": str(p.relative_to(out)).replace("\\", "/"),
                 "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in sorted(out.rglob("*")) if p.is_file()]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(out), "weights_key": args.weights_key,
                      "context_bytes": (out / "inference_context.pt").stat().st_size,
                      "training_frames_copied": 0}, indent=2))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weights-key", required=True,
                   choices=["model_state_dict", "active_state_dict", "ema_state_dict"])
    p.add_argument("--data", required=True, help="Matching complete training NPZ, used only during export")
    p.add_argument("--topology", required=True, help="Matching JNJ-2113 .prmtop and companion coordinates")
    p.add_argument("--condition-stats")
    p.add_argument("--sampling-config", required=True)
    p.add_argument("--model-args", help="JSON fallback when checkpoint has no args")
    p.add_argument("--tica-dir")
    p.add_argument("--allow-legacy-checkpoint", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    export_bundle(args)

if __name__ == "__main__":
    main()
