"""Fixed inference metadata exported once from the matching training inputs."""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import torch
from data_protocol import sha256_file

FORMAT = "highstategen_inference_context_v1"
TENSOR_FIELDS = (
    "protein_N_local", "protein_O_local", "peptide_N_local", "peptide_O_local",
    "complex_com", "protein_aa", "peptide_aa",
)
ID_FIELDS = ("protein_residue_ids", "peptide_residue_ids")

def _portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_portable(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported inference metadata type: {type(value)}")

def collect_context(dataset, topology_path, cyclic_peptide):
    """Compute exactly the same fixed metadata used by the dataset-based sampler."""
    from sample import infer_peptide_sulfur_templates
    from physics_losses import build_disulfide_loss
    attributes = {key: _portable(getattr(dataset, key)) for key in TENSOR_FIELDS}
    attributes.update({key: _portable(getattr(dataset, key)) for key in ID_FIELDS})
    attributes.update({
        "n_protein": int(dataset.n_protein), "n_peptide": int(dataset.n_peptide),
        "protein_residue_names": list(dataset.protein_residue_names),
        "peptide_residue_names": list(dataset.peptide_residue_names),
        "global_scale": float(dataset.global_scale),
        "translation_scale_mode": dataset.translation_scale_mode,
    })
    loss_data = None
    sulfur = (None, None, None, None, {}, {}, {}, {})
    if cyclic_peptide:
        loss = build_disulfide_loss(
            topology_path=str(topology_path),
            peptide_residue_ids=dataset.peptide_residue_ids,
            N_local=dataset.peptide_N_local, global_scale=dataset.global_scale,
            complex_com=dataset.complex_com,
            peptide_bb_indices=dataset.peptide_bb_indices, coords_nm=dataset.coords_nm)
        pairs = []
        if loss is not None:
            pairs = list(loss.disulfide_pairs)
            loss_data = {
                "disulfide_pairs_local": pairs,
                "ref_cbcb_dist": loss.ref_cbcb_dist, "ref_caca_dist": loss.ref_caca_dist,
                "k_cbcb": loss.k_cbcb, "k_caca": loss.k_caca,
            }
        sulfur = infer_peptide_sulfur_templates(
            dataset, disulfide_pairs_local=pairs, max_ref_frames=4096)
    context = {"format": FORMAT, "dataset": attributes,
               "cyclic_peptide": bool(cyclic_peptide),
               "disulfide": _portable(loss_data), "sulfur": _portable(sulfur)}
    validate_context(context)
    return context

def validate_context(context):
    if context.get("format") != FORMAT:
        raise ValueError("Unsupported inference context format")
    d = context["dataset"]
    if not math.isfinite(d["global_scale"]) or d["global_scale"] <= 0:
        raise ValueError("Invalid inference coordinate scale")
    if d["translation_scale_mode"] not in ("coordinate_rms", "radial_rms"):
        raise ValueError("Invalid coordinate scale convention")
    for branch in ("protein", "peptide"):
        n = d["n_" + branch]
        if not isinstance(n, int) or n < 1:
            raise ValueError("Invalid residue count")
        for suffix in ("_N_local", "_O_local"):
            x = d[branch + suffix]
            if not isinstance(x, torch.Tensor) or tuple(x.shape) != (n, 3) or not torch.isfinite(x).all():
                raise ValueError("Invalid backbone template: " + branch + suffix)
        ids, aa = d[branch + "_residue_ids"], d[branch + "_aa"]
        if ids.shape != (n,) or ids.dtype != torch.int64 or len(torch.unique(ids)) != n or (ids < 0).any():
            raise ValueError("Invalid residue identifiers")
        if aa.shape != (n,) or aa.dtype != torch.int64 or (aa < 0).any() or (aa > 20).any():
            raise ValueError("Unsupported residue vocabulary; use the system-specific exporter")
        if len(d[branch + "_residue_names"]) != n:
            raise ValueError("Residue names and counts disagree")
    if set(d["protein_residue_ids"].tolist()) & set(d["peptide_residue_ids"].tolist()):
        raise ValueError("Protein and peptide residue sets overlap")
    if d["complex_com"].shape != (3,) or not torch.isfinite(d["complex_com"]).all():
        raise ValueError("Invalid coordinate center")
    sulfur = context["sulfur"]
    if len(sulfur) != 8:
        raise ValueError("Invalid sulfur template record")
    for i in range(4):
        x = sulfur[i]
        if x is not None:
            expected = (d["n_peptide"], 3) if i in (0, 2) else (d["n_peptide"],)
            if tuple(x.shape) != expected or not torch.isfinite(x).all():
                raise ValueError("Invalid sulfur/CB template")
    def finite(value):
        if isinstance(value, dict):
            return all(finite(x) for x in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(x) for x in value)
        if isinstance(value, float):
            return math.isfinite(value)
        return True
    if not finite(context["disulfide"]) or not finite(sulfur[4:]):
        raise ValueError("Non-finite geometry reference")

def restore_dataset(context):
    validate_context(context)
    d = dict(context["dataset"])
    for key in ID_FIELDS:
        d[key] = d[key].cpu().numpy().copy()
    return SimpleNamespace(**d)

def restore_geometry(context, dataset, device="cpu"):
    from physics_losses import DisulfideGeometryLoss
    loss = None
    if context["disulfide"] is not None:
        loss = DisulfideGeometryLoss(
            N_local=dataset.peptide_N_local, global_scale=dataset.global_scale,
            complex_com=dataset.complex_com, **context["disulfide"]).to(device)
    return loss, context["sulfur"]

def load_context(path, checkpoint_path, topology_path, condition_stats_path, model_args):
    context = torch.load(path, map_location="cpu", weights_only=True)
    validate_context(context)
    sources = context.get("input_sha256", {})
    for name, filename in (("checkpoint", checkpoint_path), ("topology", topology_path)):
        if not sources.get(name) or sha256_file(filename) != sources[name]:
            raise ValueError("Inference context does not match " + name)
    if sources.get("condition_stats"):
        if not condition_stats_path or sha256_file(condition_stats_path) != sources["condition_stats"]:
            raise ValueError("Inference context does not match condition normalization")
    if context["dataset"]["translation_scale_mode"] != model_args.get("translation_scale_mode", "coordinate_rms"):
        raise ValueError("Checkpoint and inference context use different coordinate scales")
    if context["cyclic_peptide"] != bool(model_args.get("cyclic_peptide", False)):
        raise ValueError("Checkpoint and inference context disagree on cyclization")
    return context
