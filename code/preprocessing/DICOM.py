import numpy as np
import pydicom as pyd
import glob
import logging
from typing import Union
import os
import pandas as pd
import re

class DICOMextract:
    UNKNOWN = 'Unknown'

    def __init__(self, file_path: str, debug: int = 0):
        self.debug = debug
        self.metadata = pyd.dcmread(file_path)
        self.metadata.filepath = file_path
    
    def log_error(self, message, exception=None):
        if self.debug > 1 and exception:
            logging.error(f'{message}: {exception}')
        elif self.debug > 0:
            logging.error(message)

    def Orientation(self) -> Union[int, str]:
        """Attempts to extract the orientation of the scan"""
        try:
            tmp = self.metadata.ImageOrientationPatient
            imgOri = np.concatenate([np.reshape(tmp[:3], (1,3)), np.reshape(tmp[3:], (1,3))])
            indOri = np.where(np.abs(imgOri) == np.max(np.abs(imgOri)))[-1]
            tmp = set([0,1,2])- set(indOri)
            for ii in tmp:
                # 0 = sagittal, 1 = coronal, 2 = axial (1 less than Matlab)
                oriIndicator = ii #remove value from set, add 1 to fix 0 index
            return oriIndicator
        except Exception as e:
            self.log_error('Unable to read ImageOrientationPatient', e)
            return self.UNKNOWN
        
    def ID(self) -> str:
        """Attempts to extract the ID of the scan"""
        try:
            return self.metadata.PatientID
        except Exception as e:
            self.log_error('Unable to read PatientID', e)
            return self.UNKNOWN
        
    def Date(self) -> str:
        """Attempts to extract the date of the scan"""
        try:
            return self.metadata.StudyDate
        except Exception as e:
            if self.debug > 1:
                print(f'Error: {e}')
            elif self.debug > 0:
                print('Unable to read StudyDate')
            return self.UNKNOWN
        
    def Desc(self) -> str:
        """Attempts to extract the description of the scan"""
        try:
            return self.metadata.SeriesDescription
        except Exception as e:
            self.log_error('Unable to read SeriesDescription', e)
            return self.UNKNOWN
        
    def Modality(self) -> str:
        """Attempts to extract the modality of the scan"""
        try:
            if self.metadata.RepetitionTime >= 780:
                modality = 'T2'
            else:
                modality = 'T1'
            return modality
        except Exception as e:
            self.log_error('Unable to read RepetitionTime', e)
            return self.UNKNOWN
        
    def Acq(self) -> str:
        """Attempts to extract the acquisition time of the scan"""
        try:
            return self.metadata.AcquisitionTime
        except Exception as e:
            self.log_error('Unable to read AcquisitionTime', e)
            return self.UNKNOWN
        
    def Srs(self) -> str:
        """Attempts to extract the series time of the scan"""
        try:
            return self.metadata.SeriesTime
        except Exception as e:
            self.log_error('Unable to read SeriesTime', e)
            return self.UNKNOWN
        
    def Con(self) -> str:
        """Attempts to extract the content time of the scan"""
        try:
            return self.metadata.ContentTime
        except Exception as e:
            self.log_error('Unable to read ContentTime', e)
            return self.UNKNOWN
        
    def Stu(self) -> str:
        """Attempts to extract the study time of the scan"""
        try:
            return self.metadata.StudyTime
        except Exception as e:
            self.log_error('Unable to read StudyTime', e)
            return self.UNKNOWN
        
    def Tri(self) -> str:
        """Attempts to extract the trigger time of the scan"""
        try:
            return self.metadata.TriggerTime
        except Exception as e:
            self.log_error('Unable to read TriggerTime', e)
            return self.UNKNOWN
        
    def Inj(self) -> str:
        """Attempts to extract the injection time of the scan"""
        try:
            return self.metadata.InjectionTime
        except Exception as e:
            self.log_error('Unable to read InjectionTime', e)
            return self.UNKNOWN
        
    def LR(self) -> str:
        """Attempts to extract the laterality of the scan"""
        # TODO: Does metadata contain the filepath or is the file path required as an argument?

        # Find all adjacent files in the directory
        directory = os.path.dirname(self.metadata.filepath)
        files = glob.glob(os.path.join(directory, '*.dcm'))
        try:
            rcsCoordX1 = self.metadata.ImageOrientationPatient[0]
            metadata = pyd.dcmread(files[-1])
            rcsCoordX2 = metadata.ImageOrientationPatient[0]
            if np.mean([rcsCoordX1, rcsCoordX2]) > 0:
                laterality = 'left'
            elif np.mean([rcsCoordX1, rcsCoordX2]) < 0:
                laterality = 'right'
            return laterality
        except Exception as e:
            self.log_error('Unable to read ImageOrientationPatient', e)
            return self.UNKNOWN
        
    def Thickness(self) -> Union[float, str]:
        """Attempts to extract the slice thickness of the scan [in mm]"""
        try:
            return self.metadata.SliceThickness
        except Exception as e:
            self.log_error('Unable to read SliceThickness', e)
            return self.UNKNOWN
        
    def DWI(self) -> Union[float, str]:
        """Attempts to extract the DWI b-value of the scan"""
        try:
            return self.metadata.DiffusionBValue
        except Exception as e:
            self.log_error('Unable to read DiffusionBValue', e)
            return self.UNKNOWN
        
    def Type(self) -> str:
        """Attempts to extract the type of the scan"""
        try:
            return self.metadata.ImageType
        except Exception as e:
            self.log_error('Unable to read ImageType', e)
            return self.UNKNOWN
        
    def Series(self) -> Union[int, str]:
        """Attempts to extract the series number of the scan"""
        try:
            return self.metadata.SeriesNumber
        except Exception as e:
            self.log_error('Unable to read SeriesNumber', e)
            return self.UNKNOWN
        
    def NumSlices(self) -> Union[int, str]:
        """Attempts to extract the number of slices in the scan"""
        try:
            files = glob.glob('/'.join(self.metadata.filepath.split('/')[:-1])+'/*.dcm')
            n_slices = len(files)
            return n_slices
        except Exception as e:
            self.log_error('Unable to read number of slices', e)
            return self.UNKNOWN
        
    def BreastSize(self) -> Union[float, str]:
        """Attempts to extract the breast size of the scan"""
        try:
            if self.Thickness() == self.UNKNOWN or self.NumSlices() == self.UNKNOWN:
                return self.UNKNOWN
            return self.Thickness()*self.NumSlices()
        except Exception as e:
            self.log_error('Unable to read breast size', e)
            return self.UNKNOWN
        
class DICOMfilter():
    def __init__(self, dicom_table: pd.DataFrame, logger: logging.Logger = None, debug: int = 0):
        self.debug = debug
        self.logger = logger or logging.getLogger(__name__)
        self.dicom_table = dicom_table
        self.Session_ID = self.dicom_table['SessionID'].unique()
        self.SIDE = self.majorSide()
        self.removed = {}
        assert self.Session_ID.size == 1, 'Multiple Session_IDs found in the table'
        self.logger.debug(f'Analyzing {self.Session_ID}')

    def removeT2(self):
        """Removes T2 scans from the table"""
        self.removed['T2'] = self.dicom_table[self.dicom_table['Modality'] == 'T2']
        self.logger.debug(f'Removed {len(self.removed["T2"])} T2 scans | {self.Session_ID}')
        self.dicom_table = self.dicom_table[self.dicom_table['Modality'] != 'T2']
        return self.dicom_table
    
    def removeImplants(self):
        """Removes scans with implants from the table"""
        self.removed['Implants'] = []
        to_remove = []
        for i in range(len(self.dicom_table)):
            try:
                if ((float(self.dicom_table['BreastSize'][i]) < 60) or (float(self.dicom_table['BreastSize'][i]) > 198)):
                    to_remove.append(i)
            except Exception as e:
                logging.error(f'unable to read BreastSize | {e}')
                to_remove.append(i)
        self.removed['Implants'] = self.dicom_table.iloc[to_remove]
        self.logger.debug(f'Removed {len(self.removed["Implants"])} scans with implants | {self.Session_ID}')
        self.dicom_table = self.dicom_table.drop(to_remove)
        return self.dicom_table
    
    def majorSide(self):
        """Determines the major side of the breast"""
        mode_series = self.dicom_table['Lat'].mode()
        if not mode_series.empty:
            self.SIDE = mode_series[0]
            self.logger.debug(f'Major side of the breast is {self.SIDE} | {self.Session_ID}')
        else:
            self.SIDE = 'Unknown'
            self.logger.warning(f'No major side found | {self.Session_ID}')
        return self.SIDE
    
    def removeSide(self):
        """Removes scans from the minor side of the breast"""
        self.removed['Side'] = self.dicom_table[self.dicom_table['Lat'] != self.SIDE]
        self.dicom_table = self.dicom_table[self.dicom_table['Lat'] == self.SIDE]
        self.logger.debug(f'Removed {len(self.removed["Side"])} scans from the minor side | {self.Session_ID}')
        return self.dicom_table
    
    def majorSlices(self):
        """Determine the number of slices within the majority of the scans"""
        mode_series = self.dicom_table['NumSlices'].mode()
        if not mode_series.empty:
            self.N_SLICES = mode_series[0]
            self.logger.debug(f'Majority number of slices is {self.N_SLICES} | {self.Session_ID}')
        else:
            self.N_SLICES = 0
            self.logger.warning(f'No majority number of slices found | {self.Session_ID}')
        return self.N_SLICES
    
    def split_scan(self, i):
        """Splits a scan with multiple of N_SLICES slices into multiple scans"""
        directory = os.path.dirname(self.dicom_table['PATH'].iloc[i])
        files = glob.glob(os.path.join(directory, '*.dcm'))

        # Function to extract all numeric parts from a filename
        def extract_numbers(filename):
            return [int(num) for num in re.findall(r'\d+', filename)]
        
        # Analyze the first few filenames to determine the differing numeric part
        def find_differing_index(filenames, num_samples=3):
            if len(filenames) < num_samples:
                num_samples = len(filenames)
            sample_numbers = [extract_numbers(filenames[i]) for i in range(num_samples)]
            for index in range(len(sample_numbers[0])):
                values = [sample[index] for sample in sample_numbers]
                if len(set(values)) == num_samples:  # All values are different
                    return index
            return -1
        
        try:
            # Sort files based on the identified incremental numeric part
            incremental_index = find_differing_index(files)
            if incremental_index == -1:
                raise ValueError("Unable to determine the incremental part of the filenames.")
        
            def extract_incremental_value(filename):
                numbers = extract_numbers(filename)
                return numbers[incremental_index] if incremental_index < len(numbers) else float('inf')
            
            f_copy = files.copy()
            files = sorted(files, key=extract_incremental_value)

            n_actual = len(files)
            n_expected = self.N_SLICES
            n_scans = n_actual // n_expected
            self.logger.debug(f'Expecting to split scan with {n_actual} slices into {n_scans} scans of {n_expected} slices| {self.Session_ID}')
            new_subtable = {}                           
            for j in range(n_scans):
                file_group = files[(j*n_expected):((j+1)*n_expected)]
                self.logger.debug(f'Scanning {file_group[0]} as start of sub_scan {j} | {self.Session_ID}')
                extract = DICOMextract(file_group[0])
                result = {
                    'PATH': file_group[0],
                    'Orientation': extract.Orientation(),
                    'ID': extract.ID(),
                    'DATE': extract.Date(),
                    'Series_desc': extract.Desc(),
                    'Modality': extract.Modality(),
                    'AcqTime': extract.Acq(),
                    'SrsTime': extract.Srs(),
                    'ConTime': extract.Con(),
                    'StuTime': extract.Stu(),
                    'TriTime': extract.Tri(),
                    'InjTime': extract.Inj(),
                    'Lat': 'Unknown',
                    'NumSlices': n_expected,
                    'Thickness': extract.Thickness(),
                    'BreastSize': extract.BreastSize(),
                    'DWI': extract.DWI(),
                    'Type': extract.Type(),
                    'Series': extract.Series(),
                    'SessionID': self.Session_ID[0]
                }
                for key, item in result.items():
                    if key not in new_subtable.keys():
                        new_subtable[key] = []
                    new_subtable[key].append(item)
            new_scan = pd.DataFrame(new_subtable)
            self.dicom_table = self.dicom_table.drop(self.dicom_table[self.dicom_table['PATH'] == f_copy[0]].index)  # Remove the original scan
            self.dicom_table = pd.concat([self.dicom_table, new_scan], ignore_index=True)  # Add the new scans
            self.logger.debug(f'Split scan with {n_actual} slices into {n_scans} scans | {self.Session_ID}')
        except ValueError as e:
            self.logger.warning(f"Skipping scan due to error: {e} | {self.Session_ID}")
        return self.dicom_table
    
    def removeSlices(self):
        """Removes scans with a different number of slices"""
        self.N_SLICES = self.majorSlices()
        self.removed['Slices'] = self.dicom_table[self.dicom_table['NumSlices']%self.N_SLICES != 0]
        self.dicom_table = self.dicom_table[self.dicom_table['NumSlices']%self.N_SLICES == 0]

        dicom_copy = self.dicom_table.copy()
        for i in range(len(dicom_copy)):
            if dicom_copy['NumSlices'].iloc[i] != self.N_SLICES:
                self.logger.debug(f'Identified scan with multiple of {self.N_SLICES} slices | {self.Session_ID}')
                self.logger.debug('Separating into multiple scans')
                self.split_scan(i)
        self.logger.debug(f'Removed {len(self.removed["Slices"])} scans with a different number of slices | {self.Session_ID}')
        return self.dicom_table

    def removeComputed(self, flags: list):
        """Removes scans with compsuted flags in Series_desc list"""
        removed = []
        for flag in flags:
            # Identify rows to be removed
            to_remove = self.dicom_table[self.dicom_table['Type'].str.contains(flag.upper(), na=False)]
            # Append to the removed list
            removed.append(to_remove)
            # Filter out the identified rows from the dicom_table
            self.dicom_table = self.dicom_table[~self.dicom_table.index.isin(to_remove.index)]
        
        # Concatenate all removed rows into a single DataFrame
        self.removed['Computed'] = pd.concat(removed, ignore_index=True)
        
        self.logger.debug(f'Removed {len(self.removed["Computed"])} scans with computed descriptions | {self.Session_ID}')
        
        return self.dicom_table

    def removeTimes(self, filter_columns: list):
        """Removes scans with computed flags"""
        removed = []
        for column in filter_columns:
            self.dicom_table = self.dicom_table[self.dicom_table[column] != 'Unknown']
            removed.append(self.dicom_table[self.dicom_table[column] == 'Unknown'])
        self.removed['Times'] = pd.concat(removed)
        self.logger.debug(f'Removed {len(self.removed["Times"])} scans with unknown times | {self.Session_ID}')
        return self.dicom_table

    def removeDWI(self):
        """Removes DWI scans from the table"""
        self.dicom_table = self.dicom_table[self.dicom_table['DWI'] == 0]
        self.removed['DWI'] = self.dicom_table[self.dicom_table['DWI'] != 1]
        self.logger.debug(f'Removed {len(self.removed["DWI"])} DWI scans | {self.Session_ID}')
        return self.dicom_table
    
class DICOMorder():

    def __init__(self, dicom_table: pd.DataFrame, debug: int = 0):
        self.debug = debug
        self.dicom_table = dicom_table
        self.Session_ID = self.dicom_table['SessionID'].unique()
        if self.Session_ID.size > 1:
            #TODO: Implement multiple Session_IDs (non-parallel implementation)
            print('Multiple Session_IDs found in the table')
            print('Not currently implemented, please remake with a single Session_ID')
            return None
    
    def order(self, timing_param):
        """Orders the scans based on the provided timing parameter"""
        # Separate rows with 'UNKNOWN' values
        self.timing_param = timing_param
        unknown_rows = self.dicom_table[self.dicom_table[timing_param].astype(str).str.lower() == 'unknown']
        valid_rows_index = self.dicom_table[self.dicom_table[timing_param].astype(str).str.lower() != 'unknown'].index
        if self.debug > 0:
            print(f'Analyzing {self.dicom_table["SessionID"].unique()}')
            print(f'Found {len(unknown_rows)} rows with unknown {timing_param} values')
            print(f'Found {len(valid_rows_index)} rows with known {timing_param} values')
        # Convert the timing_param column to integers for valid rows
        self.dicom_table.loc[valid_rows_index, timing_param] = self.dicom_table.loc[valid_rows_index, timing_param].astype(int)

        # Sort the valid rows
        valid_rows = self.dicom_table.loc[valid_rows_index].sort_values(by=[timing_param])
        self.n_post = len(valid_rows)

        # Add a 'Major' column to the valid rows
        self.dicom_table.loc[valid_rows.index, 'Major'] = np.linspace(1, len(valid_rows), int(len(valid_rows)))
        self.dicom_table.loc[unknown_rows.index, 'Major'] = np.zeros(len(unknown_rows))

        return self.dicom_table
    
    def alternate_pre(self):
        if self.debug > 0:
            print(f'Attemting to solve for pre scan for {self.dicom_table["SessionID"].unique()}')
        # Find the scans with unknown timing parameters
        unknown_rows = self.dicom_table[self.dicom_table[self.timing_param].astype(str).str.lower() == 'unknown']
        # if there is only one unknown value, assume it is a pre scan
        if len(unknown_rows) == 1:
            if self.debug > 0:
                print(f'Found a single unknown value for {self.dicom_table["SessionID"].unique()}')
                print(f'Assuming this is a pre scan')
        # return index of unknown row
        return unknown_rows.index              

    def findPre(self):

        series_numbers = self.dicom_table['Series'][self.dicom_table['Major'] > 0].astype(int)
        if len(series_numbers) == 0:
            if self.debug > 0:
                print(f'No series_numbers found for {self.dicom_table["SessionID"].unique()}')
            #clear dicom table
            self.dicom_table = pd.DataFrame()
            return self.dicom_table
        indx = self.dicom_table[self.dicom_table['Major'] > 0].index

        # sort series numbers
        series_numbers = series_numbers.sort_values()
        pre_value = series_numbers.iloc[0] - 1

        pre_indx = self.dicom_table[self.dicom_table['Series'] == pre_value].index
        if len(pre_indx) == 0:
            if self.debug > 0:
                print(f'No pre scan found for {self.dicom_table["SessionID"].unique()}')
                print(f'WARNING: REMOVING SESSION')
                pre_indx = self.alternate_pre()
            #clear dicom table
            return self.dicom_table
        
        indx = np.append(indx, pre_indx)
        self.dicom_table = self.dicom_table.loc[indx]
        return self.dicom_table
