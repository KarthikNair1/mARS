python /gpfs/home/kn2347/MARS/MARS/preprocess_dicoms/pick_out_scans.py \
    --input_csv /gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/datafiles/data_scan.csv \
    --pixel_size_csv /gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/datafiles/data_count_axial_pixel_size.csv \
    --output_dir /gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/datafiles \
    --faulty_scan_csv /gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/data_faulty_scan.csv \
    --thumbnail_save_dir /gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/image_thumbnails_png_curated \
> '/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/datafiles/pick_out_scans.log'
