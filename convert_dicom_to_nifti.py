#!/usr/bin/env python
"""
Convert DICOM CT/PT pairs to nnUNet-format NIfTI files.
Can be run in parallel via SLURM.

Usage:
  python convert_dicom_to_nifti.py <subject_dir> <output_dir>

Example:
  python convert_dicom_to_nifti.py /path/to/2938443 /path/to/nibabel_rewrite_for_autopet3_flat
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk


def _load_dicom_series(series_dir):
    series_dir = Path(series_dir)
    dicom_paths = sorted(series_dir.glob('*.dcm'))
    if not dicom_paths:
        raise FileNotFoundError(f'No DICOM files found in {series_dir}')

    datasets = [pydicom.dcmread(str(path)) for path in dicom_paths]
    first = datasets[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    row_cosines = orientation[:3]
    col_cosines = orientation[3:]
    slice_cosines = np.cross(row_cosines, col_cosines)

    positions = np.array([np.asarray(ds.ImagePositionPatient, dtype=float) for ds in datasets])
    sort_order = np.argsort(positions @ slice_cosines)
    datasets = [datasets[index] for index in sort_order]
    positions = positions[sort_order]
    return datasets, row_cosines, col_cosines, slice_cosines, positions


def _parse_dicom_time(time_value):
    if time_value is None:
        return None
    text = str(time_value).strip()
    if not text:
        return None
    text = text.split('.')[0].zfill(6)[:6]
    return datetime.strptime(text, '%H%M%S')


def _pet_suv_factor_from_dicom_series(series_dir):
    series_dir = Path(series_dir)
    dicom_paths = sorted(series_dir.glob('*.dcm'))
    if not dicom_paths:
        raise FileNotFoundError(f'No DICOM files found in {series_dir}')

    first = pydicom.dcmread(str(dicom_paths[0]), stop_before_pixels=True)
    if not hasattr(first, 'RadiopharmaceuticalInformationSequence'):
        raise ValueError(f'Missing RadiopharmaceuticalInformationSequence in {series_dir}')

    rph = first.RadiopharmaceuticalInformationSequence[0]
    if not hasattr(first, 'PatientWeight'):
        raise ValueError(f'Missing PatientWeight in {series_dir}')
    if not hasattr(rph, 'RadionuclideTotalDose'):
        raise ValueError(f'Missing RadionuclideTotalDose in {series_dir}')
    if not hasattr(rph, 'RadionuclideHalfLife'):
        raise ValueError(f'Missing RadionuclideHalfLife in {series_dir}')

    scan_time = _parse_dicom_time(getattr(first, 'SeriesTime', None))
    injection_time = _parse_dicom_time(getattr(rph, 'RadiopharmaceuticalStartTime', None))
    if scan_time is None or injection_time is None:
        raise ValueError(f'Missing scan/injection time in {series_dir}')

    dt_seconds = (scan_time - injection_time).total_seconds()
    if dt_seconds < 0:
        dt_seconds += 24 * 3600

    half_life = float(rph.RadionuclideHalfLife)
    total_dose = float(rph.RadionuclideTotalDose)
    decay = np.exp(-np.log(2.0) * dt_seconds / half_life)
    injected_dose_atscan = total_dose * decay
    if injected_dose_atscan <= 0:
        raise ValueError(f'Computed non-positive injected dose for {series_dir}')

    return (float(first.PatientWeight) * 1e3) / injected_dose_atscan


def _sitk_from_dicom_series(series_dir):
    series_dir = Path(series_dir)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not dicom_names:
        raise FileNotFoundError(f'No DICOM files found in {series_dir}')
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def _resample_moving_to_reference(moving_image, reference_image, interpolator=sitk.sitkLinear, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving_image)


def _sitk_to_nifti(image):
    array = sitk.GetArrayFromImage(image).transpose((2, 1, 0)).astype(np.float32)
    affine = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
    affine = np.column_stack((affine * np.array(image.GetSpacing(), dtype=float), np.asarray(image.GetOrigin(), dtype=float)))
    affine = np.vstack((affine, [0.0, 0.0, 0.0, 1.0]))
    nii = nib.Nifti1Image(array, affine)
    nii.set_qform(nii.affine, code=1)
    nii.set_sform(nii.affine, code=1)
    return nii


def _nnunet_name(series_dir):
    series_name = Path(series_dir).name
    if '_ct_' in series_name:
        return f'{series_name.split("_ct_")[0]}_0000.nii.gz'
    if '_pt_' in series_name:
        return f'{series_name.split("_pt_")[0]}_0001.nii.gz'
    raise ValueError(f'Cannot infer nnUNet modality suffix from {series_name}')


def convert_nnunet_pair_from_dicoms(ct_series_dir, pt_series_dir, out_dir):
    out_dir = Path(out_dir)
    ct_path = out_dir / _nnunet_name(ct_series_dir)
    pt_path = out_dir / _nnunet_name(pt_series_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_image = _sitk_from_dicom_series(ct_series_dir)
    pt_image = _sitk_from_dicom_series(pt_series_dir)
    pt_resampled = _resample_moving_to_reference(pt_image, ct_image, interpolator=sitk.sitkLinear, default_value=0.0)

    suv_factor = _pet_suv_factor_from_dicom_series(pt_series_dir)
    pt_array = sitk.GetArrayFromImage(pt_resampled).astype(np.float32) * suv_factor
    pt_array = np.clip(pt_array, 0.0, None)
    pt_resampled_suv = sitk.GetImageFromArray(pt_array)
    pt_resampled_suv.CopyInformation(ct_image)

    nib.save(_sitk_to_nifti(ct_image), str(ct_path))
    nib.save(_sitk_to_nifti(pt_resampled_suv), str(pt_path))
    return ct_path, pt_path


def process_subject(subject_dir, output_dir):
    """Process a single subject's DICOM data to NIfTI."""
    subject_dir = Path(subject_dir)
    output_dir = Path(output_dir)
    subject_name = subject_dir.name

    ct_file = output_dir / f'{subject_name}_0000.nii.gz'
    pt_file = output_dir / f'{subject_name}_0001.nii.gz'

    if ct_file.exists() and pt_file.exists():
        print(f'[{subject_name}] Output files already exist, skipping')
        return True

    print(f'[{subject_name}] Processing...')

    # Find CT and PT subdirectories
    ct_dirs = list(subject_dir.glob('*_ct_*'))
    pt_dirs = list(subject_dir.glob('*_pt_*'))

    if len(ct_dirs) != 1:
        print(f'[{subject_name}] ERROR: Expected 1 CT directory, found {len(ct_dirs)}')
        return False
    if len(pt_dirs) != 1:
        print(f'[{subject_name}] ERROR: Expected 1 PT directory, found {len(pt_dirs)}')
        return False

    try:
        ct_dir = ct_dirs[0]
        pt_dir = pt_dirs[0]
        ct_path, pt_path = convert_nnunet_pair_from_dicoms(ct_dir, pt_dir, output_dir)
        print(f'[{subject_name}] SUCCESS: {ct_path.name}, {pt_path.name}')
        return True
    except Exception as e:
        print(f'[{subject_name}] ERROR: {e}')
        return False


def main():
    if len(sys.argv) != 3:
        print('Usage: python convert_dicom_to_nifti.py <subject_dir> <output_dir>')
        sys.exit(1)

    subject_dir = sys.argv[1]
    output_dir = sys.argv[2]

    success = process_subject(subject_dir, output_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
