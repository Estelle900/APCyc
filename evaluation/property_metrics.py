"""APCyc property helpers. Requires only stdlib; RDKit for PDB descriptors.

Definitions follow Zhao et al., APCyc (2026), Appendix C. Values obtained
without the identical third-party models/settings and normalization population
are not guaranteed numerically identical to the paper.
"""
from __future__ import annotations

import csv
import gzip
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROTEASES = ("S01.001", "A01.001", "C01.060", "M10.003", "M10.008")
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AA3 = dict(zip(
    ("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL").split(),
    ("A R N D C Q E G H I L K M F P S T W Y V").split(),
))
AA3.update({"MSE": "M", "SEC": "C", "HID": "H", "HIE": "H", "HIP": "H"})


def read_chain(pdb: str | Path, chain: str = "L") -> tuple[str, list[dict], str]:
    """Return (sequence, residues with atoms, peptide PDB text).

    Residue index is PDB order (0-based), matching sampling `pair_hat`.
    """
    pdb = Path(pdb)
    opener = gzip.open if pdb.suffix == ".gz" else open
    lines, residues, seen = [], [], {}
    with opener(pdb, "rt") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
                continue
            if line[21] != chain or line[16] not in (" ", "A", "1"):
                continue
            lines.append(line.rstrip("\n"))
            key = (line[22:26], line[26])
            if key not in seen:
                seen[key] = len(residues)
                residues.append({"name": line[17:20].strip(), "atoms": {}})
            atom = line[12:16].strip()
            try:
                xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
            except ValueError:
                continue
            residues[seen[key]]["atoms"].setdefault(atom, xyz)
    if not residues:
        raise ValueError(f"No residues found on chain {chain!r} in {pdb}")
    seq = "".join(AA3.get(r["name"], "X") for r in residues)
    return seq, residues, "\n".join(lines + ["END", ""])


def extract_chain_pdb(complex_pdb: str | Path, chain: str, out: str | Path) -> None:
    _, _, block = read_chain(complex_pdb, chain)
    Path(out).write_text(block)


def distance(a, b):
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def cyclization_ca_distance(residues, pair_hat) -> float | None:
    """CA-to-CA distance (proxy); not a linkage-specific bond measurement."""
    if not isinstance(pair_hat, (list, tuple)) or len(pair_hat) != 2:
        return None
    try:
        i, j = map(int, pair_hat)
        a, b = residues[i]["atoms"].get("CA"), residues[j]["atoms"].get("CA")
        return distance(a, b) if a and b and i >= 0 and j >= 0 else None
    except (ValueError, TypeError, IndexError):
        return None


def physicochemical_descriptors(peptide_pdb_block: str, sequence: str) -> dict:
    """RDKit PDB-derived logP/TPSA; GRAVY from chain sequence.

    No substitution with a linear SMILES: preserves bonds RDKit infers from
    the PDB, albeit missing CONECT/linkage info can change calculated values.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors

    if not sequence or any(a not in HYDROPATHY for a in sequence):
        raise ValueError("GRAVY requires only canonical amino acids (or supported mappings)")
    mol = Chem.MolFromPDBBlock(peptide_pdb_block, removeHs=False, proximityBonding=True)
    if mol is None:
        raise ValueError("RDKit could not construct a valid peptide molecule from PDB")
    heavy_atoms = mol.GetNumHeavyAtoms()
    if not heavy_atoms:
        raise ValueError("Peptide has no heavy atoms")
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return {
        "logP": float(Crippen.MolLogP(mol)),
        "GRAVY": sum(HYDROPATHY[aa] for aa in sequence) / len(sequence),
        "TPSA": float(tpsa),
        "rTPSA": float(tpsa / heavy_atoms),
        "heavy_atoms": heavy_atoms,
    }


def fit_stats(rows: list[dict]) -> dict:
    stats = {}
    for metric in ("logP", "GRAVY", "TPSA", "rTPSA", "vina_score", "rosetta_dG"):
        numbers = [float(row[metric]) for row in rows
                   if row.get(metric) is not None and math.isfinite(float(row[metric]))]
        if numbers:
            mean = statistics.mean(numbers)
            sd = statistics.pstdev(numbers)
            stats[metric] = {"mean": mean, "std": sd, "n": len(numbers)}
    return stats


def zscore(value, entry):
    if value is None or entry is None:
        return None
    sd = float(entry["std"])
    return (float(value) - float(entry["mean"])) / sd if sd > 0 else None


def permeability_proxy(row: dict, stats: dict) -> float | None:
    z = {key: zscore(row.get(key), stats.get(key)) for key in ("logP", "GRAVY", "TPSA", "rTPSA")}
    if any(value is None for value in z.values()):
        return None
    return z["logP"] + z["GRAVY"] - z["TPSA"] - z["rTPSA"]


def affinity_proxy(row: dict, stats: dict) -> float | None:
    rv, rr = zscore(row.get("vina_score"), stats.get("vina_score")), zscore(row.get("rosetta_dG"), stats.get("rosetta_dG"))
    return -0.5 * (rv + rr) if rv is not None and rr is not None else None


def windows(sequence: str, length: int = 8, stride: int = 2, max_windows: int = 30) -> list[str]:
    """ProsperousPlus input windows, including C terminus (no invented padding)."""
    sequence = "".join(a if a in HYDROPATHY else "-" for a in sequence.upper())
    if len(sequence) < length:
        return []
    starts = list(range(0, len(sequence) - length + 1, stride))
    terminal = len(sequence) - length
    if starts[-1] != terminal:
        starts.append(terminal)
    if len(starts) > max_windows:
        # Even subsampling, guaranteed to keep start and C-terminal window.
        starts = [starts[round(i * (len(starts) - 1) / (max_windows - 1))]
                  for i in range(max_windows)]
    return [sequence[i:i + length] for i in starts]


def mhc_windows(sequence: str) -> list[str]:
    if len(sequence) < 8:
        return []
    if 8 <= len(sequence) <= 11:
        return [sequence]
    return [sequence[i:i + n] for n in (8, 9, 10, 11)
            for i in range(len(sequence) - n + 1)]


def bigmhc_score(pairs: list[tuple[float, float]], top_k: int = 3, eps: float = 1e-12) -> float | None:
    """BigMHC EL/IM independently top-K averaged, then log-summed."""
    if not pairs:
        return None
    els, ims = [float(p[0]) for p in pairs], [float(p[1]) for p in pairs]
    if any(not 0 <= v <= 1 or not math.isfinite(v) for v in els + ims):
        raise ValueError("BigMHC EL and IM scores must be probabilities in [0, 1]")
    el = statistics.mean(sorted(els, reverse=True)[:top_k])
    im = statistics.mean(sorted(ims, reverse=True)[:top_k])
    return math.log(max(el, eps)) + math.log(max(im, eps))


def protease_resistance(probabilities: dict[str, list[float]]) -> float | None:
    """Negative sum of 5 protease-wise mean cleavage probabilities."""
    if any(not probabilities.get(p) for p in PROTEASES):
        return None
    for vals in probabilities.values():
        if any(not 0 <= v <= 1 or not math.isfinite(v) for v in vals):
            raise ValueError("ProsperousPlus probabilities must be in [0, 1]")
    return -sum(statistics.mean(probabilities[p]) for p in PROTEASES)


def parse_vina(output: str) -> float:
    """Parse AutoDock Vina --score_only affinity, never first arbitrary number."""
    patterns = [r"Affinity:\s*([-+]?\d+(?:\.\d+)?)\s*\(kcal/mol\)",
                r"Estimated Free Energy of Binding\s*:\s*([-+]?\d+(?:\.\d+)?)"]
    for pattern in patterns:
        found = re.search(pattern, output, re.I)
        if found:
            return float(found.group(1))
    raise ValueError("Vina affinity not found in score-only stdout/stderr")


def parse_rosetta_scorefile(path: str | Path) -> dict:
    """Parse final score.sc SCORE header and row (Ddg filter name: dg)."""
    headings, values = None, None
    for line in Path(path).read_text().splitlines():
        if not line.startswith("SCORE:"):
            continue
        fields = line.split()[1:]
        if "description" in fields:
            headings = fields
        elif headings and len(fields) == len(headings):
            values = dict(zip(headings, fields))
    if not values:
        raise ValueError(f"No Rosetta score row in {path}")
    result = {}
    if "total_score" in values:
        result["rosetta_total"] = float(values["total_score"])
    for key in ("dg", "dG", "ddg", "Ddg"):
        if key in values:
            result["rosetta_dG"] = float(values[key])
            break
    return result


def load_csv_rows(path: str | None) -> list[dict]:
    if path is None:
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_target(rows: list[dict], metrics: list[str], valid_only: bool = True):
    """Per-target means. Never silently treat unknown validity as success."""
    group = defaultdict(list)
    for r in rows:
        group[r["id"]].append(r)
    out = []
    for target, candidates in sorted(group.items()):
        included = [r for r in candidates if r.get("valid") is True] if valid_only else candidates
        record = {"id": target, "n_candidates": len(candidates),
                  "n_valid": sum(r.get("valid") is True for r in candidates),
                  "n_validity_unknown": sum(r.get("valid") is None for r in candidates)}
        for metric in metrics:
            vals = [float(r[metric]) for r in included if r.get(metric) is not None]
            record[metric] = statistics.mean(vals) if vals else None
            record[f"n_{metric}"] = len(vals)
        out.append(record)
    return out
