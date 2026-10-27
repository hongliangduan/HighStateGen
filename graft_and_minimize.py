#!/usr/bin/env python3
"""JNJ-2113 donor graft and restrained all-atom OpenMM minimization.

Core graft/minimization routines are retained from the supplied server script.
The release entry point projects each input with the archived TICA transform,
selects the nearest donor, and writes residue/atom mapping and failure records.
This script does not implement the manuscript's final structure evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Dict, List, Sequence, Tuple

import numpy as np

def kabsch_rt(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    H = (src - src_c).T @ (tgt - tgt_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = tgt_c - R @ src_c
    return R, t


def _res_atom_index_map(top: md.Topology) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}
    for r in top.residues:
        d: Dict[str, int] = {}
        for a in r.atoms:
            d[a.name] = a.index
        out[r.index] = d
    return out


def _find_receptor_peptide_chains(cand_top: md.Topology) -> Tuple[md.core.topology.Chain, md.core.topology.Chain]:
    chains = list(cand_top.chains)
    if len(chains) != 2:
        raise ValueError("JNJ-2113 candidate PDB must contain exactly 2 chains (294 receptor and 13 peptide residues).")
    counts = sorted([(len(list(ch.residues)), ch) for ch in chains], key=lambda x: x[0], reverse=True)
    receptor = counts[0][1]
    peptide = counts[-1][1]
    return receptor, peptide


def _find_ace_index(donor_top: md.Topology) -> int:
    for r in donor_top.residues:
        if r.name.upper() == "ACE":
            return int(r.index)
    raise ValueError("ACE residue not found in donor topology.")


def _build_residue_mapping(candidate_top: md.Topology, donor_top: md.Topology) -> Dict[int, int]:
    """
    Returns map: donor_res_idx -> candidate_res_idx
    """
    receptor_chain, peptide_chain = _find_receptor_peptide_chains(candidate_top)
    cand_receptor = list(receptor_chain.residues)
    cand_peptide = list(peptide_chain.residues)
    donor_res = list(donor_top.residues)
    if len(cand_receptor) != 294 or len(cand_peptide) != 13:
        raise ValueError("JNJ-2113 mapping requires 294 receptor and 13 peptide residues; truncated inputs are rejected")
    if len(donor_res) != 308 or donor_res[294].name.upper() != "ACE":
        raise ValueError("JNJ-2113 donor must have 294 receptor residues, one ACE, and 13 peptide residues")

    # Receptor mapping by order
    if len(cand_receptor) > len(donor_res):
        raise ValueError("Candidate receptor residue count exceeds donor residue count.")
    mapping: Dict[int, int] = {}
    for i, rc in enumerate(cand_receptor):
        rd = donor_res[i]
        if rc.name != rd.name:
            raise ValueError(
                f"Receptor residue name mismatch at order {i}: candidate={rc.name} donor={rd.name}"
            )
        mapping[int(rd.index)] = int(rc.index)

    # Peptide mapping by order after ACE
    ace_idx = _find_ace_index(donor_top)
    pep_start = ace_idx + 1
    if pep_start + len(cand_peptide) > len(donor_res):
        raise ValueError("Peptide mapping exceeds donor residue range.")
    for j, rp in enumerate(cand_peptide):
        rd = donor_res[pep_start + j]
        if rp.name != rd.name:
            raise ValueError(
                f"Peptide residue name mismatch at order {j}: candidate={rp.name} donor={rd.name}"
            )
        mapping[int(rd.index)] = int(rp.index)

    return mapping


def _graft_positions(
    donor_traj: md.Trajectory,
    candidate_traj: md.Trajectory,
    donor_to_candidate_res: Dict[int, int],
) -> Tuple[np.ndarray, Dict[str, int]]:
    donor_top = donor_traj.topology
    cand_top = candidate_traj.topology
    donor_xyz = np.asarray(donor_traj.xyz[0], dtype=np.float64)  # nm
    cand_xyz = np.asarray(candidate_traj.xyz[0], dtype=np.float64)  # nm
    out_xyz = donor_xyz.copy()

    donor_res_atom = _res_atom_index_map(donor_top)
    cand_res_atom = _res_atom_index_map(cand_top)

    stats = {
        "n_res_total_donor": int(donor_top.n_residues),
        "n_res_mapped": int(len(donor_to_candidate_res)),
        "n_res_kabsch": 0,
        "n_res_fallback": 0,
    }

    for r_donor in donor_top.residues:
        d_idx = int(r_donor.index)
        if d_idx not in donor_to_candidate_res:
            # e.g. ACE cap in donor; keep donor coordinates
            continue
        c_idx = donor_to_candidate_res[d_idx]
        d_atoms = donor_res_atom[d_idx]
        c_atoms = cand_res_atom[c_idx]

        need = ("N", "CA", "C")
        has_d = all(k in d_atoms for k in need)
        has_c = all(k in c_atoms for k in need)

        if has_d and has_c:
            src = donor_xyz[[d_atoms["N"], d_atoms["CA"], d_atoms["C"]], :]
            tgt = cand_xyz[[c_atoms["N"], c_atoms["CA"], c_atoms["C"]], :]
            R, t = kabsch_rt(src, tgt)
            all_idx = [a.index for a in r_donor.atoms]
            pts = donor_xyz[all_idx, :]
            out_xyz[all_idx, :] = (R @ pts.T).T + t
            stats["n_res_kabsch"] += 1
        else:
            # Fallback: copy candidate atoms that exist, keep donor for others
            for a in r_donor.atoms:
                nm = a.name
                if nm in c_atoms:
                    out_xyz[a.index, :] = cand_xyz[c_atoms[nm], :]
            stats["n_res_fallback"] += 1

    return out_xyz, stats


def _run_openmm_minimize_subprocess(
    script_path: Path,
    openmm_python: str,
    input_pdb: Path,
    topology: Path,
    output_pdb: Path,
    ca_indices: Sequence[int],
    ring_indices: Tuple[int, int],
    stage1_k: float,
    stage2_k: float,
    max_iter: int,
    timeout_s: int,
    platform: str,
) -> Tuple[bool, str]:
    cmd = [
        openmm_python,
        str(script_path),
        "_minimize_one",
        "--input-pdb",
        str(input_pdb),
        "--topology",
        str(topology),
        "--output-pdb",
        str(output_pdb),
        "--stage1-k",
        str(stage1_k),
        "--stage2-k",
        str(stage2_k),
        "--max-iter",
        str(max_iter),
        "--platform",
        platform,
        "--ca-indices",
        ",".join(str(int(x)) for x in ca_indices),
        "--ring-indices",
        f"{int(ring_indices[0])},{int(ring_indices[1])}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            return False, f"openmm_failed(rc={proc.returncode}): {msg[:1000]}"
        return True, (proc.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return False, f"openmm_timeout(>{timeout_s}s)"
    except Exception as e:
        return False, f"openmm_exception: {type(e).__name__}: {e}"


def _minimize_one_cli(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-pdb", required=True)
    p.add_argument("--topology", required=True)
    p.add_argument("--output-pdb", required=True)
    p.add_argument("--stage1-k", type=float, default=10.0)
    p.add_argument("--stage2-k", type=float, default=2.0)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--platform", default="CPU", choices=["CPU", "CUDA", "OpenCL", "Reference"])
    p.add_argument("--ca-indices", required=True)
    p.add_argument("--ring-indices", required=True)
    args = p.parse_args(list(argv))

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    input_pdb = Path(args.input_pdb).resolve()
    prmtop_path = Path(args.topology).resolve()
    out_pdb = Path(args.output_pdb).resolve()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    ca_idx = [int(x) for x in args.ca_indices.split(",") if x.strip()]
    ring = tuple(int(x) for x in args.ring_indices.split(","))
    if len(ring) != 2:
        raise ValueError("--ring-indices must provide exactly two indices.")

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    pdb = app.PDBFile(str(input_pdb))
    if pdb.topology.getNumAtoms() != prmtop.topology.getNumAtoms():
        raise ValueError(
            f"Atom count mismatch: pdb={pdb.topology.getNumAtoms()} vs prmtop={prmtop.topology.getNumAtoms()}"
        )

    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        implicitSolvent=app.OBC2,
    )

    expected = [(a.residue.index, a.name) for a in prmtop.topology.atoms()]
    actual = [(a.residue.index, a.name) for a in pdb.topology.atoms()]
    if actual != expected:
        raise ValueError("PDB atom order or residue mapping does not match the AMBER topology")
    if len(set(ring)) != 2 or min(ring) < 0 or max(ring) >= len(expected):
        raise ValueError("Invalid zero-based ring atom indices")
    if not np.isfinite(np.asarray(pdb.positions.value_in_unit(unit.nanometer))).all():
        raise ValueError("Nonfinite input coordinates")

    # CA positional restraint around grafted coordinates
    restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", float(args.stage1_k) * 418.4)  # kcal/mol/A^2 -> kJ/mol/nm^2
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    pos_nm = np.array(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    for idx in ca_idx:
        xyz = pos_nm[idx]
        restraint.addParticle(int(idx), [float(xyz[0]), float(xyz[1]), float(xyz[2])])
    system.addForce(restraint)

    # SG-SG harmonic bond restraint
    hb = mm.HarmonicBondForce()
    hb.addBond(int(ring[0]), int(ring[1]), 0.205, 50000.0)  # nm, kJ/mol/nm^2
    system.addForce(hb)

    integrator = mm.LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picoseconds,
    )
    platform = mm.Platform.getPlatformByName(args.platform)
    sim = app.Simulation(prmtop.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    # stage 1
    sim.minimizeEnergy(
        maxIterations=int(args.max_iter),
        tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
    )
    # stage 2
    sim.context.setParameter("k", float(args.stage2_k) * 418.4)
    sim.minimizeEnergy(
        maxIterations=int(args.max_iter),
        tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
    )

    st = sim.context.getState(getPositions=True)
    final_pos = st.getPositions(asNumpy=True)
    arr = np.array(final_pos.value_in_unit(unit.nanometer), dtype=float)
    if not np.isfinite(arr).all():
        raise RuntimeError("NaN/Inf detected in minimized coordinates.")

    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(prmtop.topology, final_pos, fh, keepIds=True)
    print(f"saved {out_pdb}")
    return 0


def project_actual_tica(traj, model, ref_ca, align_positions, exclude_resnames):
    """Project C-alpha coordinates using the fixed transform's saved alignment."""
    if not np.isfinite(traj.xyz).all():
        raise ValueError("Nonfinite candidate coordinates")
    excluded = {name.upper() for name in exclude_resnames}
    ca_indices = [a.index for a in traj.topology.atoms
                  if a.name == "CA" and not a.residue.is_water
                  and a.residue.name.upper() not in excluded]
    if len(ca_indices) != ref_ca.n_atoms:
        raise ValueError("Candidate/reference C-alpha counts differ; check atom mapping")
    selected = traj.atom_slice(ca_indices)
    names = [a.residue.name for a in selected.topology.atoms]
    ref_names = [a.residue.name for a in ref_ca.topology.atoms]
    if names != ref_names:
        raise ValueError("Candidate/reference C-alpha residue order differs")
    selected.superpose(ref_ca, frame=0, atom_indices=align_positions,
                       ref_atom_indices=align_positions)
    values = np.asarray(model.transform(selected.xyz.reshape(selected.n_frames, -1)))
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values[:, :2]).all():
        raise ValueError("The archived transform did not produce finite TIC1/TIC2")
    return values[:, :2]


def nearest_donor(scores, actual_tica):
    scores = np.asarray(scores, dtype=float)
    actual_tica = np.asarray(actual_tica, dtype=float)
    if (scores.ndim != 2 or scores.shape[1] != 2 or not len(scores)
            or actual_tica.shape != (2,) or not np.isfinite(scores).all()
            or not np.isfinite(actual_tica).all()):
        raise ValueError("Supply finite donor scores and one actual TIC1/TIC2 coordinate")
    squared = np.sum((scores - actual_tica[None, :]) ** 2, axis=1)
    index = int(np.argmin(squared))
    return index, float(np.sqrt(squared[index]))


def mapping_record(donor_top, candidate_top, mapping):
    donor_atoms = _res_atom_index_map(donor_top)
    candidate_atoms = _res_atom_index_map(candidate_top)
    records = []
    for donor_residue in donor_top.residues:
        donor_index = int(donor_residue.index)
        candidate_index = mapping.get(donor_index)
        record = {
            "donor_residue_index": donor_index,
            "donor_residue_name": donor_residue.name,
            "donor_atom_indices": [a.index for a in donor_residue.atoms],
            "candidate_residue_index": candidate_index,
        }
        if candidate_index is None:
            record["operation"] = "retain_donor_coordinates"
        else:
            donor_names, candidate_names = donor_atoms[donor_index], candidate_atoms[candidate_index]
            record["candidate_residue_name"] = candidate_top.residue(candidate_index).name
            anchors = [name for name in ("N", "CA", "C")
                       if name in donor_names and name in candidate_names]
            record["operation"] = "N_CA_C_rigid_fit" if len(anchors) == 3 else "copy_shared_atoms"
            record["anchor_atoms"] = [
                {"name": name, "donor_atom_index": donor_names[name],
                 "candidate_atom_index": candidate_names[name]} for name in anchors]
            if len(anchors) != 3:
                record["copied_atoms"] = [
                    {"name": name, "donor_atom_index": index,
                     "candidate_atom_index": candidate_names[name]}
                    for name, index in donor_names.items() if name in candidate_names]
        records.append(record)
    return {"index_base": 0, "residues": records}


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_resources(args):
    global md
    import mdtraj as md
    import joblib
    sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
    import conformation_space as legacy_models
    # Both names occur in archived joblib files. Load only trusted model files.
    sys.modules.setdefault("pca_conformation_space", legacy_models)
    globals().update(NumpyTICAModel=legacy_models.NumpyTICAModel,
                     NumpyPCAModel=legacy_models.NumpyPCAModel)
    model = legacy_models._load_model(Path(args.tica_model))
    if not hasattr(model, "eigenvalues_") or not hasattr(model, "lagtime_"):
        raise ValueError("Supply the archived TICA model, not a PCA projection")
    reference = Path(args.tica_ref_dir)
    ref_ca = md.load(str(reference / "reference_ca.pdb"))
    align = np.load(reference / "align_ca_positions.npy", allow_pickle=False)
    if (align.ndim != 1 or len(align) < 3 or not np.issubdtype(align.dtype, np.integer)
            or align.min() < 0 or align.max() >= ref_ca.n_atoms
            or len(np.unique(align)) != len(align)):
        raise ValueError("Invalid archived C-alpha alignment indices")
    metadata_file = reference / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    excluded = metadata.get("exclude_resnames", ["Na+", "Cl-", "NA", "CL", "K+", "K", "MG", "MG2+", "CA", "ZN", "ZN2+"])
    scores = np.load(args.donor_tica_scores, allow_pickle=False)
    with md.open(args.trajectory) as handle:
        count = len(handle)
    if scores.shape != (count, 2) or count < 1 or not np.isfinite(scores).all():
        raise ValueError("Donor TIC1/TIC2 rows must match every trajectory frame in order")
    donor0 = md.load_frame(args.trajectory, 0, top=args.topology)
    atoms = list(donor0.topology.atoms)
    ring = tuple(args.ring_indices)
    if len(set(ring)) != 2 or min(ring) < 0 or max(ring) >= len(atoms):
        raise ValueError("Invalid zero-based ring atom indices")
    if set(atoms[index].residue.index for index in ring) != {295, 300}:
        raise ValueError("JNJ-2113 closure must use peptide residues 296 and 301 (one-based), not receptor cysteines")
    if any(atoms[index].name != "SG" for index in ring):
        raise ValueError("JNJ-2113 closure indices must select the two SG atoms")
    # Catch normalized-vs-raw scores, frame ordering and transform mismatches.
    for index in sorted({0, count // 2, count - 1}):
        donor = donor0 if index == 0 else md.load_frame(args.trajectory, index, top=args.topology)
        projected = project_actual_tica(donor, model, ref_ca, align, excluded)[0]
        if not np.allclose(projected, scores[index], rtol=1e-3, atol=1e-3):
            raise ValueError(f"Donor scores disagree with the fixed TICA projection at frame {index}; check units, normalization and row order")
    return model, ref_ca, align, excluded, scores, donor0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="One raw conformation per PDB")
    parser.add_argument("--topology", required=True, help="Matching dry AMBER prmtop")
    parser.add_argument("--trajectory", required=True, help="All-atom donor trajectory")
    parser.add_argument("--tica_model", required=True, help="Trusted archived TICA pickle/joblib file")
    parser.add_argument("--tica_ref_dir", required=True, help="Archived reference_ca.pdb and align_ca_positions.npy")
    parser.add_argument("--donor_tica_scores", required=True, help="[n_frames, 2], same fixed TICA transform and units")
    parser.add_argument("--ring_indices", type=int, nargs=2, required=True,
                        help="Zero-based SG atom indices in the dry prmtop")
    parser.add_argument("--output_dir", required=True, help="New output directory")
    parser.add_argument("--stage1_k", type=float, default=10.0)
    parser.add_argument("--stage2_k", type=float, default=2.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--openmm_python", default=sys.executable)
    parser.add_argument("--openmm_platform", choices=["CPU", "CUDA", "OpenCL", "Reference"], default="CPU")
    parser.add_argument("--minimize_timeout_s", type=int, default=300)
    return parser.parse_args()


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if min(args.stage1_k, args.stage2_k, args.max_iter, args.minimize_timeout_s) <= 0:
        raise ValueError("Positive restraint, iteration and timeout settings are required")
    records = [{"sample_id": number, "input": str(Path(path).resolve()), "status": "pending"}
               for number, path in enumerate(args.inputs, 1)]
    settings = vars(args).copy()
    settings.update(restraint_expression="0.5*k*distance_squared",
                    tolerance_kJ_mol_nm=10.0, donor_selection="nearest_actual_generated_TICA",
                    implicit_solvent="OBC2", constraints="HBonds")
    provenance = {}

    def write_log():
        payload = {"initial_sample_count": len(records), "settings": settings,
                   "n_complete": sum(row["status"] == "complete" for row in records),
                   "n_failed": sum(row["status"] == "failed" for row in records),
                   "input_artifacts": provenance, "samples": records}
        (output / "reconstruction_log.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    write_log()
    try:
        model, ref_ca, align, excluded, scores, donor0 = load_resources(args)
        for name, source in (("topology", args.topology), ("tica_model", args.tica_model),
                             ("donor_tica_scores", args.donor_tica_scores),
                             ("reference_ca", Path(args.tica_ref_dir) / "reference_ca.pdb"),
                             ("alignment", Path(args.tica_ref_dir) / "align_ca_positions.npy")):
            provenance[name] = {"path": str(Path(source).resolve()), "sha256": _sha256(source)}
        trajectory = Path(args.trajectory)
        provenance["trajectory"] = {"path": str(trajectory.resolve()),
                                    "bytes": trajectory.stat().st_size, "frames": len(scores)}
        ca_indices = [a.index for a in donor0.topology.atoms if a.name == "CA"]
        if not ca_indices:
            raise ValueError("No C-alpha atoms in donor topology")
    except Exception as error:
        for record in records:
            record.update(status="failed", stage="input_validation", error=f"{type(error).__name__}: {error}")
        write_log()
        print(f"Input validation failed: {error}", file=sys.stderr)
        return 1

    for record in records:
        number = record["sample_id"]
        prefix = f"{number:04d}_{Path(record['input']).stem}"
        grafted = output / f"{prefix}_grafted.pdb"
        minimized = output / f"{prefix}_minimized.pdb"
        try:
            candidate = md.load(record["input"])
            if candidate.n_frames != 1:
                raise ValueError("Supply one raw conformation per input PDB")
            actual = project_actual_tica(candidate, model, ref_ca, align, excluded)[0]
            index, distance = nearest_donor(scores, actual)
            donor = donor0 if index == 0 else md.load_frame(args.trajectory, index, top=args.topology)
            mapping = _build_residue_mapping(candidate.topology, donor.topology)
            xyz, statistics = _graft_positions(donor, candidate, mapping)
            if not np.isfinite(xyz).all():
                raise ValueError("Nonfinite grafted coordinates")
            mapped = mapping_record(donor.topology, candidate.topology, mapping)
            mapped.update(donor_frame=index, actual_generated_tica=actual.tolist(),
                          donor_tica=scores[index].tolist(), donor_tica_distance=distance,
                          input_sha256=_sha256(record["input"]), graft_stats=statistics)
            mapping_file = output / f"{prefix}_mapping.json"
            mapping_file.write_text(json.dumps(mapped, indent=2) + "\n", encoding="utf-8")
            record.update(actual_generated_tica=actual.tolist(), donor_frame=index,
                          donor_tica_distance=distance, mapping=mapping_file.name)
            md.Trajectory(xyz=xyz[None].astype(np.float32), topology=donor.topology).save_pdb(str(grafted))
            record["grafted_pdb"] = grafted.name
            ok, message = _run_openmm_minimize_subprocess(
                Path(__file__).resolve(), args.openmm_python, grafted, Path(args.topology), minimized,
                ca_indices, tuple(args.ring_indices), args.stage1_k, args.stage2_k,
                args.max_iter, args.minimize_timeout_s, args.openmm_platform)
            record["minimize_log"] = message
            if not ok or not minimized.is_file():
                raise RuntimeError(message or "OpenMM did not write an output structure")
            record.update(status="complete", minimized_pdb=minimized.name)
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc(limit=4))
        write_log()
        print(f"[{number}/{len(records)}] {record['status']}: {Path(record['input']).name}")
    return 0 if all(record["status"] == "complete" for record in records) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_minimize_one":
        raise SystemExit(_minimize_one_cli(sys.argv[2:]))
    raise SystemExit(run(parse_args()))
