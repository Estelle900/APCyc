#!/usr/bin/env python3
"""Add APCyc property metrics without changing legacy structural metrics.

Run from APCyc repo root:
 python -m evaluation.score_properties --samples OUT/apcyc_predictions.jsonl --out OUT/properties
Or with original cal_metrics input JSONL:
 python -m evaluation.score_properties --results OUT/results.jsonl --out OUT/properties

Optional external predictions can be supplied as exported CSV tables; see README.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.property_metrics import (
    PROTEASES, affinity_proxy, aggregate_target, bigmhc_score,
    cyclization_ca_distance, extract_chain_pdb, fit_stats, load_csv_rows,
    mhc_windows, parse_rosetta_scorefile, parse_vina, permeability_proxy,
    physicochemical_descriptors, protease_resistance, read_chain, windows,
)

METRICS = ["vina_score", "rosetta_dG", "rosetta_total", "camsol_score",
           "immunogenicity", "protease_resistance", "permeability", "affinity_proxy"]


def write_csv(path: Path, records: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="") as handle:
        wr = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(records)


def _key(row):
    return str(row["id"]), str(int(row["number"]))


def read_manifest(path, is_sampling, use_rosetta):
    root = Path(path).resolve().parent
    samples = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if is_sampling:
                target = str(rec.get("receptor_id") or rec["sample_id"])
                number = int(rec.get("sample_idx", 0))
                pdb = Path(rec["pdb"])
                if not pdb.is_absolute() and not pdb.exists():
                    pdb = root / pdb
            else:
                target = str(rec["id"])
                number = int(rec["number"])
                suffix = "_rosetta" if use_rosetta else ""
                pdb = Path(rec.get("pdb") or root / "candidates" / target /
                           f"{target}_gen_{number}{suffix}.pdb")
                if not pdb.is_absolute() and not pdb.exists():
                    pdb = root / pdb
            samples.append({"id": target, "number": number, "pdb": str(pdb),
                            "pair_hat": rec.get("pair_hat"),
                            "sequence_hint": rec.get("gen_seq"),
                            "rec_chain": rec.get("rec_chain", "R"),
                            "lig_chain": rec.get("lig_chain", "L")})
    keys = [_key(row) for row in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Non-unique (target id, sample number); inspect manifest")
    return samples


def csv_index(path, required, kind):
    rows = load_csv_rows(path)
    if rows and not set(required) <= set(rows[0]):
        raise ValueError(f"{kind} CSV requires {required}; got {list(rows[0])}")
    index = defaultdict(list)
    for rec in rows:
        index[_key(rec)].append(rec)
    return index


def only_scalar(index, key, field):
    vals = index.get(key, [])
    if not vals:
        return None
    if len(vals) != 1:
        raise ValueError(f"Duplicate {field} for {key}")
    val = vals[0].get(field, "")
    return float(val) if val not in (None, "") else None


def read_normalization(path):
    if path is None:
        return {}
    stats = json.loads(Path(path).read_text())
    for key, obj in stats.items():
        if not isinstance(obj, dict) or not {"mean", "std"} <= set(obj):
            raise ValueError(f"Normalization for {key} requires mean and std")
    return stats


def score_vina(pdbqt_dir, pdb, rec_chain, lig_chain, vina_bin, coords, size_padding):
    """Uses pre-generated PDBQT files, avoids silently guessing charges/types."""
    stem = Path(pdb).name.replace(".pdb.gz", "").removesuffix(".pdb")
    folder = Path(pdbqt_dir)
    receptor = folder / f"{stem}_{rec_chain}.pdbqt"
    ligand = folder / f"{stem}_{lig_chain}.pdbqt"
    if not receptor.is_file() or not ligand.is_file():
        raise FileNotFoundError(f"Prepare {receptor} and {ligand} with MGLTools first")
    minimum = [min(a[i] for a in coords) for i in range(3)]
    maximum = [max(a[i] for a in coords) for i in range(3)]
    center = [(lo + hi) / 2 for lo, hi in zip(minimum, maximum)]
    size = [max(hi - lo + size_padding, 1.0) for lo, hi in zip(minimum, maximum)]
    args = [vina_bin, "--receptor", str(receptor), "--ligand", str(ligand)]
    for axis, c, s in zip("xyz", center, size):
        args += [f"--center_{axis}", str(c), f"--size_{axis}", str(s)]
    args += ["--score_only"]
    process = subprocess.run(args, capture_output=True, text=True, timeout=240, check=False)
    if process.returncode:
        raise RuntimeError(f"Vina exited {process.returncode}: {(process.stderr + process.stdout)[-1200:]}")
    return parse_vina(process.stdout + "\n" + process.stderr)


def score_rosetta(rosetta_bin, pdb, xml_path):
    """No relaxation; dg filter must appear in score.sc as column dg."""
    with tempfile.TemporaryDirectory(prefix="apcyc_rosetta_") as tmp:
        outfile = Path(tmp) / "rosetta.sc"
        args = [rosetta_bin, "-s", str(Path(pdb).resolve()),
                "-parser:protocol", str(Path(xml_path).resolve()),
                "-out:file:scorefile", str(outfile), "-out:nooutput", "true",
                "-overwrite", "true"]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=900, check=False)
        if proc.returncode:
            raise RuntimeError(f"Rosetta exited {proc.returncode}: {(proc.stderr + proc.stdout)[-1200:]}")
        return parse_rosetta_scorefile(outfile)


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.samples or args.results, bool(args.samples), args.rosetta_structures)
    camsol = csv_index(args.camsol_csv, ("id", "number", "camsol_score"), "CamSol")
    bigmhc = csv_index(args.bigmhc_csv, ("id", "number", "window", "el", "im"), "BigMHC")
    proteases = csv_index(args.prosperous_csv, ("id", "number", "protease", "pro"), "ProsperousPlus")
    externals = csv_index(args.scores_csv, ("id", "number"), "external scores")
    stats = read_normalization(args.normalization)

    if args.rosetta_bin and not args.rosetta_xml:
        raise ValueError("--rosetta-bin requires --rosetta-xml")
    if args.vina_bin and not args.pdbqt_dir:
        raise ValueError("--vina-bin requires --pdbqt-dir (preprepared PDBQT files)")

    records = []
    mhc_inputs, protease_inputs, camsol_inputs = [], [], []
    for candidate in manifest:
        rec = {"id": candidate["id"], "number": candidate["number"],
               "pdb": candidate["pdb"], "errors": []}
        key = _key(rec)
        pdb = candidate["pdb"]
        peptide_residues = []
        try:
            seq, peptide_residues, peptide_block = read_chain(pdb, args.lig_chain)
            rec["sequence"] = seq
            if candidate.get("sequence_hint") and candidate["sequence_hint"] != seq:
                rec["errors"].append("PDB sequence differs from results JSONL gen_seq")
            desc = physicochemical_descriptors(peptide_block, seq)
            rec.update(desc)
            rec["cyc_distance_CA"] = cyclization_ca_distance(peptide_residues, candidate["pair_hat"])
            rec["cyc_success_CA_proxy"] = (
                3.0 <= rec["cyc_distance_CA"] <= 8.0
                if rec["cyc_distance_CA"] is not None else None
            )
            camsol_inputs.append({"id": rec["id"], "number": rec["number"], "sequence": seq})
            for win in mhc_windows(seq):
                mhc_inputs.append({"id": rec["id"], "number": rec["number"],
                                   "window": win, "allele": args.allele})
            for i, win in enumerate(windows(seq)):
                protease_inputs.append({"id": rec["id"], "number": rec["number"],
                                        "window_index": i, "window": win})
        except (ValueError, OSError, ImportError) as exc:
            rec["errors"].append(f"PDB/descriptor error: {exc}")
            rec.setdefault("cyc_distance_CA", None)
            rec.setdefault("cyc_success_CA_proxy", None)

        # External predictor outputs are intentionally ingested, NOT fabricated.
        if key in camsol:
            rec["camsol_score"] = only_scalar(camsol, key, "camsol_score")
        if key in bigmhc:
            # Require complete prediction coverage; using a subset silently
            # inflates/deflates the top-K score for long sequences.
            expected = mhc_windows(rec.get("sequence", ""))
            received = [x.get("window") for x in bigmhc[key]]
            if not expected or Counter(expected) != Counter(received):
                rec["errors"].append(
                    f"BigMHC window coverage mismatch: expected {len(expected)}, "
                    f"got {len(received)} (including duplicates)"
                )
            else:
                pair_scores = [(float(x["el"]), float(x["im"])) for x in bigmhc[key]]
                rec["immunogenicity"] = bigmhc_score(pair_scores, top_k=args.mhc_top_k)
        if key in proteases:
            grouped = defaultdict(list)
            for x in proteases[key]:
                grouped[x["protease"]].append(float(x["pro"]))
            rec["protease_resistance"] = protease_resistance(grouped)
            if rec["protease_resistance"] is None:
                rec["errors"].append("Incomplete ProsperousPlus five-protease panel")
        if key in externals:
            if len(externals[key]) != 1:
                raise ValueError(f"Duplicate external scalar row {key}")
            for field in ("vina_score", "rosetta_dG", "rosetta_total", "camsol_score",
                          "immunogenicity", "protease_resistance", "cyc_distance_A"):
                value = externals[key][0].get(field)
                if value not in (None, ""):
                    rec[field] = float(value)

        if args.vina_bin and peptide_residues:
            xyz = [xyz for residue in peptide_residues for atom, xyz in residue["atoms"].items()
                   if not atom.startswith("H")]
            try:
                rec["vina_score"] = score_vina(args.pdbqt_dir, pdb, args.rec_chain,
                                               args.lig_chain, args.vina_bin, xyz,
                                               args.vina_padding)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                rec["errors"].append(f"Vina error: {exc}")
        if args.rosetta_bin:
            try:
                rec.update(score_rosetta(args.rosetta_bin, pdb, args.rosetta_xml))
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                rec["errors"].append(f"Rosetta error: {exc}")

        # Actual cyclization atom-pair distance from external data overrides CA proxy.
        # CA proxy is used ONLY if explicitly requested by the user.
        cyc = rec.get("cyc_distance_A")
        if cyc is None and args.allow_ca_validity:
            cyc = rec.get("cyc_distance_CA")
        rec["cyclization_success"] = 3.0 <= cyc <= 8.0 if cyc is not None else None
        rec["energy_success"] = rec.get("rosetta_dG", 0) < 0 if rec.get("rosetta_dG") is not None else None
        if rec["cyclization_success"] is None or rec["energy_success"] is None:
            rec["valid"] = None
        else:
            rec["valid"] = rec["cyclization_success"] and rec["energy_success"]
        records.append(rec)

    if args.fit_normalization:
        stats = fit_stats(records)
        (out / "fitted_normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
        warnings.warn("LOCAL-fit normalization is NOT comparable to APCyc's published reference stats")
    for rec in records:
        rec["permeability"] = permeability_proxy(rec, stats)
        rec["affinity_proxy"] = affinity_proxy(rec, stats)

    # Predictor input templates to simplify reproducible external batch processing.
    write_csv(out / "camsol_input.csv", camsol_inputs, ["id", "number", "sequence"])
    write_csv(out / "bigmhc_input.csv", mhc_inputs, ["id", "number", "window", "allele"])
    write_csv(out / "prosperous_input.csv", protease_inputs,
              ["id", "number", "window_index", "window"])

    for rec in records:
        rec["errors"] = "; ".join(rec["errors"])
    write_csv(out / "properties.csv", records)
    target_means = aggregate_target(records, METRICS, valid_only=not args.all_candidates)
    write_csv(out / "target_means.csv", target_means)
    counts = {"n_candidates": len(records),
              "n_valid": sum(r["valid"] is True for r in records),
              "n_unknown_validity": sum(r["valid"] is None for r in records),
              "n_targets": len(target_means)}
    summary = {"coverage": counts, "method": "means over valid candidates, then means over targets",
               "normalization": ("local_fit_NOT_PAPER_COMPARABLE" if args.fit_normalization
                                 else (str(args.normalization) if args.normalization else None)),
               "metrics": {m: (sum(float(r[m]) for r in target_means if r[m] is not None) /
                               sum(r[m] is not None for r in target_means)
                               if any(r[m] is not None for r in target_means) else None)
                           for m in METRICS},
               "n_targets_scored": {m: sum(r[m] is not None for r in target_means) for m in METRICS},
               "validity_policy": "CA proxy" if args.allow_ca_validity else "explicit linkage distance required"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out / 'properties.csv'} and {out / 'target_means.csv'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--samples", help="apcyc_sample.py apcyc_predictions.jsonl")
    inputs.add_argument("--results", help="Legacy cal_metrics.py input JSONL")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--rec-chain", default="R")
    p.add_argument("--lig-chain", default="L")
    p.add_argument("--rosetta-structures", action="store_true", help="Use _rosetta.pdb filenames with --results")
    p.add_argument("--normalization", help="Author's exact dataset stats JSON for comparability")
    p.add_argument("--fit-normalization", action="store_true", help="Fit local population stats (NOT comparable to paper)")
    p.add_argument("--bigmhc-csv", help="CSV: id,number,window,el,im")
    p.add_argument("--prosperous-csv", help="CSV: id,number,protease,pro")
    p.add_argument("--camsol-csv", help="CSV: id,number,camsol_score")
    p.add_argument("--scores-csv", help="Scalar scores CSV: id,number,vina_score,rosetta_dG,...")
    p.add_argument("--allele", default="HLA-A*02:01")
    p.add_argument("--mhc-top-k", type=int, default=3)
    p.add_argument("--vina-bin", help="Vina executable (requires --pdbqt-dir)")
    p.add_argument("--pdbqt-dir", help="Prepared {PDB_stem}_{R/L}.pdbqt files")
    p.add_argument("--vina-padding", type=float, default=10.0)
    p.add_argument("--rosetta-bin", help="rosetta_scripts executable")
    p.add_argument("--rosetta-xml", help="RosettaScripts XML defining dg filter")
    p.add_argument("--allow-ca-validity", action="store_true", help="CA distance as approximate cyclization validity (not exact paper definition)")
    p.add_argument("--all-candidates", action="store_true", help="Summarize all candidates, even when validity unknown; not paper methodology")
    args = p.parse_args(argv)
    if args.fit_normalization and args.normalization:
        p.error("Choose only one of --fit-normalization and --normalization")
    return args


if __name__ == "__main__":
    main(parse_args())
