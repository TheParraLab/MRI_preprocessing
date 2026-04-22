"""
Shared test fixtures and helpers for MRI preprocessing tests.

Provides utilities to create minimal and realistic DICOM files for testing
the scanDicom pipeline without requiring real patient data.
"""

import datetime
import os
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid


def make_minimal_dcm(path, modality='MR', series_number=1, patient_id='P1'):
    """Create a minimal, modern-format DICOM file for tests.

    Uses a modern TransferSyntaxUID and avoids deprecated pydicom attributes.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = str(patient_id)
    ds.Modality = modality
    ds.SeriesNumber = series_number
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.save_as(path, write_like_original=False, enforce_file_format=True)
    return path


def make_realistic_mr_dcm(path, **kwargs):
    """Create a realistic MR DICOM file with common MRI-specific attributes.

    This helper creates a DICOM file that closely mimics real MRI scanner output,
    including attributes commonly accessed by DICOMextract and other pipeline steps.

    Args:
        path (str): File path to write the DICOM file to.
        **kwargs: Optional attributes to override defaults:
            - modality (str): DICOM modality (default: 'MR')
            - series_number (int): Series number (default: 1)
            - patient_id (str): Patient ID (default: 'TEST001')
            - patient_name (str): Patient name (default: 'Test^Patient')
            - patient_birthdate (str): Patient birth date in YYYYMMDD format
            - study_date (str): Study date in YYYYMMDD format
            - study_time (str): Study time in HHMMSS format
            - series_description (str): Series description (default: 'Test Series')
            - repetition_time (float): Repetition Time in ms (default: 500.0 -> T1)
            - echo_time (float): Echo Time in ms (default: 25.0)
            - num_slices (int): Number of slices (default: 32)
            - slice_thickness (float): Slice thickness in mm (default: 3.0)
            - image_orientation_patient (list): 6 floats for orientation
            - laterality (str): Laterality code ('L', 'R', 'B')
            - diffusion_b_value (int): DWI b-value (default: 0)
            - acquisition_time (str): Acquisition time in HHMMSS format
            - series_time (str): Series time in HHMMSS format
            - trigger_time (str): Trigger time in HHMMSS format or 'Unknown'
            - manufacturer (str): Scanner manufacturer (default: 'TEST')
            - modality_specific (dict): Additional modality-specific attributes

    Returns:
        pydicom.Dataset: The created DICOM dataset.
    """
    # Defaults
    defaults = {
        'modality': 'MR',
        'series_number': 1,
        'patient_id': 'TEST001',
        'patient_name': 'Test^Patient',
        'patient_birthdate': '19900101',
        'study_date': datetime.datetime.now().strftime('%Y%m%d'),
        'study_time': datetime.datetime.now().strftime('%H%M%S'),
        'series_description': 'Test Series',
        'repetition_time': 500.0,  # T1-weighted (default < 780ms)
        'echo_time': 25.0,
        'num_slices': 32,
        'slice_thickness': 3.0,
        'image_orientation_patient': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        'laterality': None,
        'diffusion_b_value': 0,
        'acquisition_time': datetime.datetime.now().strftime('%H%M%S'),
        'series_time': datetime.datetime.now().strftime('%H%M%S'),
        'trigger_time': 'Unknown',
        'manufacturer': 'TEST',
        'content_time': datetime.datetime.now().strftime('%H%M%S'),
        'injection_time': None,
        'image_type': ['ORIGINAL', 'PRIMARY'],
        'patient_sex': 'O',
        'study_instance_uid': generate_uid(),
        'series_instance_uid': generate_uid(),
        'sop_instance_uid': generate_uid(),
        'sop_class_uid': pydicom.uid.MRImageStorage,
    }
    defaults.update(kwargs)

    # Create file meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = defaults['sop_class_uid']
    file_meta.MediaStorageSOPInstanceUID = defaults['sop_instance_uid']
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    # Create dataset
    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Required DICOM attributes
    ds.SOPClassUID = defaults['sop_class_uid']
    ds.SOPInstanceUID = defaults['sop_instance_uid']

    # Patient data
    ds.PatientID = defaults['patient_id']
    ds.PatientName = defaults['patient_name']
    ds.PatientBirthDate = defaults['patient_birthdate']
    ds.PatientSex = defaults['patient_sex']

    # Study data
    ds.StudyDate = defaults['study_date']
    ds.StudyTime = defaults['study_time']
    ds.StudyInstanceUID = defaults['study_instance_uid']
    ds.AccessionNumber = f'ACC{defaults["patient_id"]}'

    # Series data
    ds.SeriesNumber = defaults['series_number']
    ds.SeriesInstanceUID = defaults['series_instance_uid']
    ds.SeriesDescription = defaults['series_description']

    # MR-specific attributes
    ds.Modality = defaults['modality']
    ds.RepetitionTime = defaults['repetition_time']
    ds.EchoTime = defaults['echo_time']
    ds.NumSlices = defaults['num_slices']
    ds.SliceThickness = defaults['slice_thickness']
    ds.ImageOrientationPatient = defaults['image_orientation_patient']
    ds.AcquisitionTime = defaults['acquisition_time']
    ds.SeriesTime = defaults['series_time']
    # TriggerTime (VR DS) requires numeric; skip when 'Unknown'
    try:
        float(defaults['trigger_time'])
        ds.TriggerTime = defaults['trigger_time']
    except (ValueError, TypeError):
        pass
    ds.Manufacturer = defaults['manufacturer']

    if defaults.get('content_time'):
        ds.ContentTime = defaults['content_time']
    if defaults.get('injection_time'):
        ds.InjectionTime = defaults['injection_time']

    # DWI attributes
    ds.DiffusionBValue = defaults['diffusion_b_value']

    # Image type
    ds.ImageType = defaults['image_type']

    # Laterality
    if defaults.get('laterality'):
        ds.Laterality = defaults['laterality']

    # Additional modality-specific attributes
    if kwargs.get('modality_specific'):
        for key, value in kwargs['modality_specific'].items():
            setattr(ds, key, value)

    ds.save_as(path, write_like_original=False, enforce_file_format=True)
    return ds


def make_t1_mr_dcm(path, **kwargs):
    """Create a T1-weighted MR DICOM file.

    Args:
        path (str): File path to write.
        **kwargs: Additional attributes to override.

    Returns:
        pydicom.Dataset: The created DICOM dataset.
    """
    return make_realistic_mr_dcm(path, repetition_time=450.0, **kwargs)


def make_t2_mr_dcm(path, **kwargs):
    """Create a T2-weighted MR DICOM file.

    Args:
        path (str): File path to write.
        **kwargs: Additional attributes to override.

    Returns:
        pydicom.Dataset: The created DICOM dataset.
    """
    return make_realistic_mr_dcm(path, repetition_time=850.0, **kwargs)


def make_dwi_mr_dcm(path, b_value=1000, **kwargs):
    """Create a DWI MR DICOM file.

    Args:
        path (str): File path to write.
        b_value (int): DWI b-value (default: 1000).
        **kwargs: Additional attributes to override.

    Returns:
        pydicom.Dataset: The created DICOM dataset.
    """
    return make_realistic_mr_dcm(path, diffusion_b_value=b_value,
                                  series_description='DWI', **kwargs)


def create_test_dicom_directory(base_path, files_config):
    """Create a directory structure with multiple DICOM files for testing.

    Args:
        base_path (str): Root directory to create.
        files_config (list): List of dicts, each describing one DICOM file.
            Each dict supports the same kwargs as make_realistic_mr_dcm()
            plus a 'filename' key.

    Returns:
        str: Path to the created directory.
    """
    os.makedirs(base_path, exist_ok=True)

    for cfg in files_config:
        filename = cfg.pop('filename')
        filepath = os.path.join(base_path, filename)
        make_realistic_mr_dcm(filepath, **cfg)

    return base_path


def create_test_study_structure(tmp_path, studies_config):
    """Create a multi-study directory structure for testing.

    Args:
        tmp_path (pytest.Path): pytest temporary path fixture.
        studies_config (dict): Dict mapping study subdirectory names to their
            file configurations (list of dicts for create_test_dicom_directory).

    Returns:
        str: Path to the root data directory.
    """
    root = tmp_path / "test_study"
    root.mkdir(parents=True)

    for study_name, files_config in studies_config.items():
        study_dir = root / study_name
        create_test_dicom_directory(str(study_dir), files_config)

    return str(root)
