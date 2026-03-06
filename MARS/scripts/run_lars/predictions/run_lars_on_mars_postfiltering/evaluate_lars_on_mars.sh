cd /gpfs/home/kn2347/MARS
conda activate myeloma6
python MARS/run_lars/predictions.py MARS/scripts/run_lars/predictions/run_lars_on_mars_postfiltering/config.yaml
python MARS/run_lars/aggregate_corsag.py --pred_file_pattern /gpfs/data/mayerhoeferlab/projects/myeloma/results/run_lars_on_mars_postfiltering/pred_val_0.csv
python MARS/evaluate/calculate_metrics_given_predictions.py \
    --pred_csv /gpfs/data/mayerhoeferlab/projects/myeloma/results/run_lars_on_mars_postfiltering/pred_val_mipview_aggregate.csv \
    --results_save_dir /gpfs/data/mayerhoeferlab/projects/myeloma/results/run_lars_on_mars_postfiltering/
    --train_test_split /gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/train_val_test_split.pkl \
    --which_split val