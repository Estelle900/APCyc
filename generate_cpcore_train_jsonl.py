import csv
import json
from pathlib import Path

source = Path("CPCore/CPCore/CPCore_properties/CPCore_Basic.tsv")
output = Path("cpcore_train.jsonl")
type_map = {
    "DISULFIDE": "SS",
    "HEADTAIL": "HT",
    "ISOPEPTIDE": "ISO",
}

with source.open(newline="") as infile, output.open("w") as outfile:
    for row in csv.DictReader(infile, delimiter="\t"):
        length = int(row["length"])
        record = {
            "sample_id": row["id"],
            "peptide_length": length,
            "c_star": type_map[row["cyclic_type"].upper()],
            "gt_pair": [0, length - 1],
        }
        outfile.write(json.dumps(record) + "\n")

print(f"Wrote {output}")