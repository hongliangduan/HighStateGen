#!/usr/bin/env python3
"""Split sampler PDB models without changing atom names, serials or coordinates."""
import argparse
import json
from pathlib import Path


def split_models(text):
    lines = text.splitlines(keepends=True)
    if not any(line.startswith("MODEL ") for line in lines):
        if not any(line.startswith(("ATOM  ", "HETATM")) for line in lines):
            raise ValueError("PDB contains no atoms")
        return [text]
    models, active, outside = [], None, []
    for line in lines:
        if line.startswith("MODEL "):
            if active is not None:
                raise ValueError("Nested MODEL record")
            active = []
        elif line.startswith("ENDMDL"):
            if active is None:
                raise ValueError("ENDMDL without MODEL")
            if not any(x.startswith(("ATOM  ", "HETATM")) for x in active):
                raise ValueError("Empty PDB model")
            models.append(active)
            active = None
        elif active is not None:
            active.append(line)
        elif line.startswith(("ATOM  ", "HETATM")):
            raise ValueError("Atoms outside MODEL records")
        elif line.startswith(("CONECT", "CRYST1")):
            outside.append(line)
    if active is not None:
        raise ValueError("Unterminated MODEL record")
    return ["".join(model + outside) + "END\n" for model in models]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    models = split_models(args.input.read_text(encoding="utf-8"))
    if len(models) != args.expected_count:
        raise ValueError(f"Found {len(models)} models; expected {args.expected_count}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, text in enumerate(models):
        name = f"sample_{index:04d}.pdb"
        (args.output_dir / name).write_text(text, encoding="utf-8")
        entries.append({"initial_sample_id": index, "file": name})
    (args.output_dir / "samples.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(models)} unchanged coordinate models")


if __name__ == "__main__":
    main()
