GPU=0 bash scripts/run_exp_pipe.sh \
  training_run \
  configs/apcyc/apcyc_ae.yaml \
  configs/apcyc/apcyc_ldm.yaml \
  configs/apcyc/setup_latent_guidance.yaml \
  configs/apcyc/lnr_local.yaml \
  0100