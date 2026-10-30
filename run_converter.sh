APCYC_LNR=CPSea/Model_Generation_and_Evaluation/TestSet/LNR
python -m scripts.apcyc_prepare_lnr_sampling \
  --receptor_dir "$APCYC_LNR/LNR_receptors/LNR_receptors" \
  --separated_dir "$APCYC_LNR/LNR_separated/LNR_separated" \
  --out_complex_dir datasets/apcyc_lnr/complexes \
  --out_index datasets/apcyc_lnr/index.txt \
  --out_meta datasets/apcyc_lnr/meta.jsonl \
  --rec_chain R \
  --lig_chain L