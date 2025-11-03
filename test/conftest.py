import datetime
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian


def make_minimal_dcm(path, modality='MR', series_number=1, patient_id='P1'):
    """Create a minimal, modern-format DICOM file for tests.

    Uses an explicit TransferSyntaxUID on the file_meta and avoids setting
    deprecated FileDataset attributes to silence pydicom deprecation warnings.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    # Set a Transfer Syntax UID instead of setting dataset endian/VR attributes
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Populate required/commonly-used tags
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = str(patient_id)
    ds.Modality = modality
    ds.SeriesNumber = series_number
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')

    # Write using modern API flag to avoid write_like_original deprecation
    ds.save_as(path, write_like_original=False, enforce_file_format=True)
    return path