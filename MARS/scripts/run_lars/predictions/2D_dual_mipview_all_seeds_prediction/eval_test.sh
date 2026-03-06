cd /gpfs/home/kn2347/MARS
#conda activate myeloma6
python /gpfs/home/kn2347/MARS/MARS/run_lars/predictions.py /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/predictions/2D_dual_mipview_all_seeds_prediction/config_test.yaml
python MARS/run_lars/aggregate_corsag.py --pred_file_pattern "/gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_dual_mipview_all_seeds_prediction/test/pred_test_*.csv"
python MARS/evaluate/calculate_metrics_given_predictions.py \
    --pred_csv /gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_dual_mipview_all_seeds_prediction/test/pred_test_mipview_aggregate.csv \
    --results_save_dir /gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_dual_mipview_all_seeds_prediction/test \
    --train_test_split /gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/train_val_test_split.pkl \
    --which_split test