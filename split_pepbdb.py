from collections import defaultdict
from pathlib import Path
import random

root = Path("datasets/pepbdb")
groups = defaultdict(list)
with (root / "processed/index.txt").open() as index_file:
  for line in index_file:
    sample_id = line.split("\t", 1)[0]
    pdb_id = sample_id.split("_", 1)[0]
    groups[pdb_id].append(sample_id)

pdb_ids = sorted(groups)
random.Random(12).shuffle(pdb_ids)
train_end = int(0.8 * len(pdb_ids))
valid_end = int(0.9 * len(pdb_ids))
splits = {
  "train": pdb_ids[:train_end],
  "valid": pdb_ids[train_end:valid_end],
  "test": pdb_ids[valid_end:],
}
for split_name, split_pdb_ids in splits.items():
  with (root / f"{split_name}.txt").open("w") as split_file:
    for pdb_id in split_pdb_ids:
      split_file.write("".join(f"{sample_id}\n" for sample_id in groups[pdb_id]))