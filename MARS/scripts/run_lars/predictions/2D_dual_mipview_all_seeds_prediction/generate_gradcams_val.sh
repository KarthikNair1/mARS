cd /gpfs/home/kn2347/MARS

python MARS/run_lars/select_scans_for_inspection.py /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/predictions/2D_dual_mipview_all_seeds_prediction/generate_gradcams_val.yaml
python MARS/run_lars/generate_gradcams.py /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/predictions/2D_dual_mipview_all_seeds_prediction/generate_gradcams_val.yaml