import numpy as np
import pydicom as pyd
import glob
import logging
from typing import Union
import os
import pandas as pd
import re
import shutil

class DICOMextract:
    """
    Class for extracting relevant metadata from DICOM files.
    """
    UNKNOWN = 'Unknown'

    def __init__(self, file_path: str, debug: int = 0):
        """
        Initialize the extractor with a DICOM file path.

        Args:
            file_path (str): The path to the DICOM file.
            debug (int): Debug level for logging.

        TODO: Consider lazy loading or selective tag reading if parsing thousands
              of massive files. `stop_before_pixels=True` helps, but further pydicom
              optimizations exist (e.g., `specific_tags`).
        """
        self.debug = debug
        self.metadata = pyd.dcmread(file_path, stop_before_pixels=True)
        self.metadata.filepath = file_path
    
    def log_error(self, message, exception=None):
        if self.debug > 1 and exception:
            logging.error(f'{message}: {exception}')
        elif self.debug > 0:
            logging.error(message)

    def Orientation(self) -> Union[int, str]:
        """
        Attempts to extract the orientation of the scan.

        Returns:
            Union[int, str]: Integer representing orientation (0 = sagittal, 1 = coronal,
                             2 = axial) or UNKNOWN.
        """
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
    
    def Accession(self) -> str:
        """Attempts to extract the accession number of the scan"""
        try:
            return self.metadata.AccessionNumber
        except Exception as e:
            self.log_error('Unable to read AccessionNumber', e)
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
        # Acquisition Time isthe time when acquisition of the scans data occurred
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
        """
        Attempts to extract the laterality of the scan.

        Returns:
            str: 'left', 'right', 'bilateral', or UNKNOWN.

        TODO: Edge case handling for lateralities. What if series descriptions conflict
              with the explicit 'Laterality' tag? Should one take precedence?
              Currently, it favors 'Laterality' tag first.
        """
        # Adding in multiple robust mechanisms to attempt to determine if the scan is left or right
        # Find all adjacent files in the directory
        ###### Old method using ImageOrientationPatient, not reliable ######
        #directory = os.path.dirname(self.metadata.filepath)
        #files = glob.glob(os.path.join(directory, '*.dcm'))
        #try:
        #    rcsCoordX1 = self.metadata.ImageOrientationPatient[0]
        #    metadata = pyd.dcmread(files[-1])
        #    rcsCoordX2 = metadata.ImageOrientationPatient[0]
        #    if np.mean([rcsCoordX1, rcsCoordX2]) > 0:
        #        laterality = 'left'
        #    elif np.mean([rcsCoordX1, rcsCoordX2]) < 0:
        #        laterality = 'right'
        #    return laterality
        #except Exception as e:
        #    self.log_error('Unable to read ImageOrientationPatient', e)
        #    return self.UNKNOWN
        ######## New method using Laterality tag if available ######
        laterality = getattr(self.metadata, 'Laterality', '').strip().lower()
        if laterality in ['l', 'left']:
            return 'left'
        elif laterality in ['r', 'right']:
            return 'right'
        elif laterality in ['b', 'bilateral']:
            return 'bilateral'
        # Secondary method using SeriesDescription
        desc = getattr(self.metadata, 'SeriesDescription', '').lower()
        if 'bilateral' in desc or ('left' in desc and 'right' in desc):
            return 'bilateral'
        elif 'left' in desc:
            return 'left'
        elif 'right' in desc:
            return 'right'
        # Fallback to using ImageOrientationPatient where needed
        try:
            rcsCoordX1 = self.metadata.ImageOrientationPatient[0]
            directory = os.path.dirname(self.metadata.filepath)
            files = sorted(glob.glob(directory, '*.dcm'))
            rcsCoordX2 = pyd.dcmread(files[-1], stop_before_pixels=True).ImageOrientationPatient[0]
            if np.mean([rcsCoordX1, rcsCoordX2]) > 0:
                return 'left'
            elif np.mean([rcsCoordX1, rcsCoordX2]) < 0:
                return 'right'
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
        """
        Attempts to extract the number of slices in the scan.

        Returns:
            Union[int, str]: Number of slices or UNKNOWN.

        TODO: Performance bottleneck. `glob.glob` on the directory for every single
              file processing can drastically slow down extraction, particularly on NFS.
              Consider passing the slice count directly if it is already known or
              caching directory sizes.
        """
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
    
    def ScanDur(self) -> Union[float, str]:
        """Attempts to extract the scan duration of the scan, in microseconds"""
        try:
            return self.metadata[('0019','105A')].value
        except Exception as e:
            self.log_error('Unable to read AcquisitionDuration', e)
            return self.UNKNOWN

    def Name(self) -> str:
        """Attempts to extract the name of the scan"""
        try:
            return self.metadata.PatientName
        except Exception as e:
            self.log_error('Unable to read PatientName', e)
            return self.UNKNOWN
        
    def DOB(self) -> str:
        """Attempts to extract the date of birth of the patient"""
        try:
            return self.metadata.PatientBirthDate
        except Exception as e:
            self.log_error('Unable to read PatientBirthDate', e)
            return self.UNKNOWN
        
class DICOMfilter():
    """
    Class to filter a DataFrame of DICOM metadata corresponding to a single SessionID.
    """
    def __init__(self, dicom_table: pd.DataFrame, logger: logging.Logger = None, debug: int = 0, tmp_save: str='/FL_system/data/tmp/'):
        """
        Initialize the filter with a data subset for a single session.

        Args:
            dicom_table (pd.DataFrame): The DICOM data to filter.
            logger (logging.Logger): Logger instance.
            debug (int): Debug level.
            tmp_save (str): Path to temporary save directory for relocations.
        """
        self.debug = debug
        self.logger = logger or logging.getLogger(__name__)
        self.dicom_table = dicom_table
        self.Session_ID = self.dicom_table['SessionID'].unique()
        self.SIDE = self.majorSide()
        self.removed = {}
        self.tmp_save = tmp_save
        self.temporary_relocations = []
        self.multiple_lat = False
        if 'Pre_scan' not in self.dicom_table.columns:
            self.dicom_table['Pre_scan'] = 0
        if 'Post_scan' not in self.dicom_table.columns:
            self.dicom_table['Post_scan'] = 0
        assert self.Session_ID.size == 1, 'Multiple Session_IDs found in the table'
        self.logger.debug('='*50)
        self.logger.debug(f'Analyzing {self.Session_ID}')
        self.removeT2()
        # Add Valid column

    def print_table(self, df: pd.DataFrame = None, columns: list = None):
        """Prints the current state of the dicom table"""
        if df is None:
            df = self.dicom_table
        if columns is None:
            columns = df.columns
        # Filter to only columns that exist
        columns = [col for col in columns if col in df.columns]
        if not columns:
            self.logger.debug(f"No matching columns found in df")
            return
        missing = [col for col in columns if col not in df.columns]
        if missing:
            self.logger.debug(f"Skipping missing columns: {missing}")
        self.logger.debug(f"\n{df[columns].to_string()}")

    def removeT2(self):
        """Removes T2 scans from the table"""
        self.removed['T2'] = self.dicom_table[self.dicom_table['Modality'].isin(['T2', 'Unknown'])]
        self.dicom_table = self.dicom_table[self.dicom_table['Modality'].isin(['T1'])]
        self.logger.debug(f'Removed {len(self.removed["T2"])} T2 scans | {self.Session_ID}')
        #self.dicom_table['Remove_T2'] = self.dicom_table['Modality'].apply(lambda x: 1 if x == 'T1' else 0)
        #self.update_valid('Remove_T2')
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
    
    # def removeSlices(self):
    #     """Removes scans with a different number of slices"""
    #     self.N_SLICES = self.majorSlices()
    #     self.removed['Slices'] = self.dicom_table[self.dicom_table['NumSlices']%self.N_SLICES != 0]
    #     self.dicom_table = self.dicom_table[self.dicom_table['NumSlices']%self.N_SLICES == 0]
    #     #self.dicom_table['Remove_Slices'] = np.where(self.dicom_table['NumSlices']%self.N_SLICES == 0, 0, 1)
    #     dicom_copy = self.dicom_table.copy()
    #     for i in range(len(dicom_copy)):
    #         if dicom_copy['NumSlices'].iloc[i] != self.N_SLICES:
    #             self.logger.debug(f'Identified scan with multiple of {self.N_SLICES} slices | {self.Session_ID}')
    #             splitter = DICOMsplit(dicom_copy.iloc[[i]], logger=self.logger, debug=self.debug, tmp_save=self.tmp_save)
    #             self.temporary_relocations.append(splitter.temporary_relocations)
    #             # remove the original row entry
    #             self.dicom_table = self.dicom_table.drop(self.dicom_table[self.dicom_table['PATH'] == dicom_copy['PATH'].iloc[i]].index)
    #             # add the new rows
    #             self.dicom_table = pd.concat([self.dicom_table, splitter.output_table], ignore_index=True)
    #             self.logger.debug(f'Separated scan with {dicom_copy["NumSlices"].iloc[i]} slices into {len(splitter.output_table)} scans | {self.Session_ID}')
    #     self.logger.debug(f'Removed {len(self.removed["Slices"])} scans with a different number of slices | {self.Session_ID}')
    #     #self.update_valid('Remove_Slices')
    #     return self.dicom_table

    def apply_slices(self, use: str = 'pre') -> pd.DataFrame:
        """
        Filters scans to only retain those matching the expected slice count for 'pre' or 'post'.

        Args:
            use (str): Specifies whether to filter based on 'pre' slices or 'post' slices.

        Returns:
            pd.DataFrame: The filtered dataframe.

        TODO: Edge case: what if both pre and post slice lists exist but are completely
              disjoint from the actual available scans? The logic removes everything and
              returns an empty DataFrame. Consider raising a specific warning to trigger
              manual review rather than silently dropping the session.
        """
        if self.dicom_table.empty:
            return self.dicom_table

        if (not hasattr(self, 'pre_slices')) & (not hasattr(self, 'post_slices')):
            self.logger.error(f'Expected slices not defined for pre or post, cannot apply slice filtering.  Try running pre and post detection first | {self.Session_ID}')
            return self.dicom_table
        if use.lower() == 'pre' and hasattr(self, 'pre_slices'):
            self.logger.debug(f'Filtering scans using determined pre slices')
            self.removed['invalid_slices'] = self.dicom_table[~self.dicom_table['NumSlices'].apply(lambda x: any(x % s == 0 for s in self.pre_slices))]
            self.dicom_table = self.dicom_table[self.dicom_table['NumSlices'].apply(lambda x: any(x % s == 0 for s in self.pre_slices))]
            self.logger.debug(f'Removed {len(self.removed["invalid_slices"])} scans with invalid number of slices [expected {self.pre_slices}] | {self.Session_ID}')
        elif use.lower() == 'post' and hasattr(self, 'post_slices'):
            self.logger.debug(f'Filtering scans using determined post slices')
            self.removed['invalid_slices'] = self.dicom_table[~self.dicom_table['NumSlices'].apply(lambda x: any(s % x == 0 for s in self.post_slices))]
            self.dicom_table = self.dicom_table[self.dicom_table['NumSlices'].apply(lambda x: any(s % x == 0 for s in self.post_slices))]
            self.logger.debug(f'Removed {len(self.removed["invalid_slices"])} scans with invalid number of slices [expected {self.post_slices}] | {self.Session_ID}')
        elif use.lower() == 'pre':
            self.logger.warning(f'Pre slices not defined, cannot apply slice filtering | {self.Session_ID}')
        elif use.lower() == 'post':
            self.logger.warning(f'Post slices not defined, cannot apply slice filtering | {self.Session_ID}')
        else:
            self.logger.warning(f'Invalid use parameter: {use}. Must be either "pre" or "post" | {self.Session_ID}')
        return self.dicom_table

    def Types(self, flags: list):
        """
        Filters the table by removing scans with flags in the Image Type tag, which often contains markers of computed scans

        args:
        flags: list of strings, which will be searched for in the Image Type tag. If any of the flags are present, the scan will be removed
        """
        removed = []
        for flag in flags:
            # Identify rows to be removed
            to_remove = self.dicom_table[self.dicom_table['Type'].str.contains(flag.upper(), na=False)]
            # Append to the removed list
            removed.append(to_remove)
            # Filter out the identified rows from the dicom_table
            self.dicom_table = self.dicom_table[~self.dicom_table.index.isin(to_remove.index)]
            # Flag computed scans
            #self.dicom_table['Remove_Computed'] = np.where(self.dicom_table['Type'].str.contains(flag.upper(), na=False), 1, 0)
        # Concatenate all removed rows into a single DataFrame
        self.removed['Computed'] = pd.concat(removed, ignore_index=True)
        self.logger.debug(f'Removed {len(self.removed["Computed"])} scans with computed descriptions | {self.Session_ID}')
        #self.update_valid('Remove_Computed')
        return self.dicom_table

    def Description(self, flags: list):
        desc_pattern = '|'.join(map(re.escape, flags))
        desc_matches = self.dicom_table['Series_desc'].fillna('').str.lower().str.contains(desc_pattern, na=False)
        self.removed['Description'] = self.dicom_table[desc_matches]
        self.dicom_table = self.dicom_table[~desc_matches]
        self.logger.debug(f'Removed {len(self.removed["Description"])} scans for containing flagged descriptions | {self.Session_ID}')
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
    
    def enforcePrimary(self):
        """Removes all scans without a PRIMary Tag"""
        removed = []
        self.dicom_table = self.dicom_table.loc[self.dicom_table['Type'].str.contains('PRIMARY', na=False)]
        self.removed['Not_Primary'] = self.dicom_table.loc[self.dicom_table['Type'].str.contains('PRIMARY', na=False) == False]
        return self.dicom_table
    
    def isolate_DISCO(self):
        self.disco_table = self.dicom_table.loc[self.dicom_table['IS_DISCO'] == True]
        self.dicom_table = self.dicom_table.loc[self.dicom_table['IS_DISCO'] == False]
        return

    def detect_DISCO(self):
        """Detects and removes DISCO scans if steady state scans are also present"""
        disco_pattern = re.compile(r'disco', re.IGNORECASE)
        self.dicom_table['IS_DISCO'] = self.dicom_table['Series_desc'].str.contains(disco_pattern, na=False)
        is_disco = self.dicom_table.loc[self.dicom_table['IS_DISCO'] == True]
        not_disco = self.dicom_table.loc[self.dicom_table['IS_DISCO'] == False]
        slices_not_disco = not_disco['NumSlices'].unique()
        slices_disco = is_disco['NumSlices'].unique()
        if not_disco['Pre_scan'].sum() >= 1:
            not_disco = not_disco.loc[(not_disco['Pre_scan'] == 1)| (not_disco['Post_scan'] == 1)]

        if len(not_disco) > 2 and len(is_disco) > 0:
            self.logger.debug(f'Detected {len(is_disco)} DISCO scans and {len(not_disco)} non-DISCO scans, selected NON_DISCO | {self.Session_ID}')
            # Need to check if non-disco represent a full sequence
            self.removed['DISCO'] = is_disco
            self.dicom_table = not_disco
            #print('------')
            #print('NON DISCO SELECTED')
            #print('------')
            #print('DISCO')
            #print(is_disco)
            #print("NON DISCO")
            #print(not_disco)
            #input('Please review DISCO detection')
        elif len(not_disco) < 3:
            self.logger.debug(f'Detected {len(is_disco)} DISCO scans and {len(not_disco)} non-DISCO scans, selected DISCO | {self.Session_ID}')
            self.removed['NON_DISCO'] = not_disco
            self.dicom_table = is_disco
            #print('------')
            #print('DISCO SELECTED')
            #print('------')
            #print('DISCO')
            #print(is_disco)
            #print("NON DISCO")
            #print(not_disco)
            #input('Please review DISCO detection')
        return

    def determine_n_slices(self, df):
        # This funciton is intended to identify the actual number of slices per scan.
        # This is usually the mode of all scans available, but some scans are concatenated across the entire sequence
        n_slices = df['NumSlices'].unique()

    def detect_pre(self, action: str = 'check'):
        '''
        Attempt to differentiate pre and post scans through various techniques

        args:
        - action: string, either 'check' or 'apply', determines whether the function should simply check for the presence of a valid pre scan detection, or also apply filtering to isolate the detected pre scans. Default is 'check'
        '''
        assert action.lower() in ['check', 'apply'], 'Invalid action for detect_pre()'

        def isolate_primary(arr, action: str):
            assert action in ['check', 'apply'], 'Invalid action for isolate_primary()'
            if action.lower() == 'check':
                primary = len(self.dicom_table.loc[arr & (self.dicom_table['Type'].str.contains('PRIMARY', na=False))])
                if (primary == 1) or ((primary == 2 and self.multiple_lat)):
                    return True
                else:
                    return False
            elif action.lower() == 'apply':
                self.dicom_table = self.dicom_table.loc[arr | (self.dicom_table['Type'].str.contains('PRIMARY', na=False))]
                self.removed['Not_primary'] = self.dicom_table.loc[arr & (~self.dicom_table['Type'].str.contains('PRIMARY', na=False))]
                self.logger.debug(f'Isolated primary pre scan | {self.Session_ID}')

        def pre_trigger_time(cumulative: bool = False):
            # Trigger time approach
            if cumulative:
                mask = self.dicom_table['Pre_scan'] == 1
                self.dicom_table.loc[mask & (self.dicom_table['Post_scan'] == 0), 'Pre_scan'] = self.dicom_table.loc[mask & (self.dicom_table['Post_scan'] == 0), 'TriTime'].str.lower() == 'unknown'
            else:
                self.dicom_table.loc[self.dicom_table['Post_scan'] == 0, 'Pre_scan'] = self.dicom_table.loc[self.dicom_table['Post_scan'] == 0, 'TriTime'].str.lower() == 'unknown'
            pre_found = self.dicom_table['Pre_scan'].to_numpy().astype(bool)
            self.logger.debug(f'Trigger time pre scan detection found {pre_found.sum()} pre scans | {self.Session_ID}')
            return pre_found

        def pre_series_desc(cumulative: bool = False):
            # Series Description approach
            series_lower = self.dicom_table.loc[self.dicom_table['Post_scan'] == 0, 'Series_desc'].astype(str).str.lower()
            contains_pre = series_lower.str.contains('pre', na=False).astype(int)

            if cumulative:
                mask = self.dicom_table['Pre_scan'] == 1
                self.dicom_table.loc[mask & (self.dicom_table['Post_scan'] == 0), 'Pre_scan'] = contains_pre[mask].astype(bool)
            else:             
                self.dicom_table.loc[self.dicom_table['Post_scan'] == 0, 'Pre_scan'] = contains_pre
            pre_found = self.dicom_table['Pre_scan'].to_numpy().astype(bool)
            self.logger.debug(f'Series Description pre scan detection found {pre_found.sum()} pre scans | {self.Session_ID}')
            return pre_found

        def check_array(arr, name: str, action: str = 'check'):
            '''
            This funciton checks the array resulting from a given filter to determine if it is likely to represent a valid pre scan detection. 
            The function checks that there is at least one candidate, and that if there are two candidates, this is consistent with the presence of multiple laterality (self.multiple_lat = True)
            
            args: 
            - arr: numpy array, the boolean array resulting from a given filter (e.g. series description or trigger time)
            - name: string, the name of the filter being checked, used for logging purposes

            output:
            - boolean, True if the array is consistent with a valid pre scan detection, False otherwise
            '''
            if arr.sum() == 0:
                self.logger.debug(f'{name} detection found no pre candidates | {self.Session_ID}')
                return False
            if (arr.sum() == 1) | ((arr.sum() == 2) & self.multiple_lat):
                self.logger.debug(f'{name} detection found single pre candidate [per side if multiple lat] | {self.Session_ID}')
                return True
            elif isolate_primary(arr, 'check'):
                self.logger.debug(f'{name} detection found single pre candidate [per side if multiple lat] after isolating primary | {self.Session_ID}')
                if action.lower() == 'apply':
                    self.logger.debug(f'=== Removed non-primary entries from {name} detection | {self.Session_ID}')
                    isolate_primary(arr, 'apply')
                return True
            else:
                return False
        
        def check_duplicates(arr, action: str = 'check'):
            '''
            This funciton checks if the 2 filter candidates are adjacent series numbers
            If so, will keep the largest post
            
            args:
            - arr: numpy array, the boolean array resulting from a given filter (e.g. series description or trigger time), which has already been determined to contain 2 candidates
            - action: string, either 'check' or 'apply', determines whether the function should simply check for the presence of adjacent series numbers, or also apply filtering to keep the newer scan and remove the older scan. Default is 'check'
            
            TODO: Should this be replaced by preferentially selecting fat saturated?
            '''
            assert arr.sum() == 2, 'Too many candidates to check for duplicates'
            series_nums = self.dicom_table.loc[arr, 'Series'].values
            if np.abs(series_nums[0] - series_nums[1]) <= 2:
                self.logger.debug(f'Found semi-adjacent series numbers among candidates, likely duplicate scans | {self.Session_ID}')
                # Keep the newer scan (higher series number)
                if action.lower() == 'apply':
                    to_keep = np.argmax(series_nums)
                    to_remove = np.argmin(series_nums)
                    self.dicom_table.loc[arr, 'Pre_scan'] = False
                    self.dicom_table.loc[arr, 'Pre_scan'].iloc[to_keep] = True
                    self.removed[f'Adjacent Series'] = self.dicom_table.loc[arr].iloc[[to_remove]]
                    self.logger.debug(f'=== Removed older duplicate scan based on series number | {self.Session_ID}')
                return True
            else:
                return False
        
        def check_slices(arr, action: str = 'check'):
            '''
            This function will test for differences in number of slices in detected scans
            
            '''
            unique_slices = self.dicom_table.loc[arr, 'NumSlices'].unique()
            if len(unique_slices) == 1:
                self.logger.debug(f'Single number of slices among filtered pre scans | {self.Session_ID}')
                self.pre_slices = unique_slices
                return True
            elif len(unique_slices) == 2:
                self.logger.debug(f'Two unique slice numbers found, setting multiple_lat to True | {self.Session_ID}')
                self.multiple_lat = True
                self.pre_slices = unique_slices
                return False
        
        if hasattr(self, 'dicom_post'):
            self.logger.debug(f'Post scans seperated already, filtering by slice numbers')
            self.apply_slices(use='post')
            
        pre_found_sd = pre_series_desc()
        if check_array(pre_found_sd, 'Series Description', action) and check_slices(pre_found_sd, action):
            self.dicom_table['Pre_scan'] = pre_found_sd
            return True
        elif (pre_found_sd).sum() == 2:
            if check_slices(pre_found_sd, action):
                self.dicom_table['Pre_scan'] = pre_found_sd
                return True
            elif check_duplicates(pre_found_sd, action):
                self.dicom_table['Pre_scan'] = pre_found_sd
                return True
                
        pre_found_tt = pre_trigger_time()
        if check_array(pre_found_tt, 'Trigger Time', action) and check_slices(pre_found_tt, action):
            self.logger.debug(f'Succesfully detected pre scan using trigger time, no need for further filtering | {self.Session_ID}')
            self.dicom_table['Pre_scan'] = pre_found_tt
            return True
        elif (pre_found_tt).sum() == 2:
            # Check if multiple lat is possible but ambigious
            unique_slices = self.dicom_table.loc[pre_found_tt, 'NumSlices'].unique()
            if len(unique_slices) == 2:
                self.multiple_lat = True
                self.logger.debug(f'Multiple pre scans detected with different number of slices, setting multiple_lat to True | {self.Session_ID}')
                return True
            else:
                self.logger.debug(f'Multiple pre scans detected with same number of slices, multiple_lat remains False | {self.Session_ID}')
        
        pre_found_tt = pre_trigger_time()
        cumul = np.array(pre_found_sd) & np.array(pre_found_tt)
        if check_array(cumul, 'Series Description and Trigger Time combined', action):
            self.logger.debug(f'Series Desc and Tri Time combined provide single pre [per side], no need for further filtering | {self.Session_ID}')
            self.dicom_table['Pre_scan'] = cumul
            return True
        elif (cumul.sum()) == 2:
            # Check if multiple lat is possible but ambigious
            unique_slices = self.dicom_table.loc[cumul, 'NumSlices'].unique()
            if len(unique_slices) == 2:
                self.multiple_lat = True
                self.logger.debug(f'Multiple pre scans detected with different number of slices, setting multiple_lat to True | {self.Session_ID}')
                return True
            else:
                self.logger.debug(f'Multiple pre scans detected with same number of slices, multiple_lat remains False | {self.Session_ID}')
        
        self.logger.error(f'Existing filtering failed to capture pre scan for given session | {self.Session_ID}')
        if action.lower() == 'apply':
            self.removed['Pre_Failure'] = self.dicom_table.copy()
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
        return False
    
    def detect_post(self, action: str = 'check'):
        '''Attempt to identifypost scans through various techniques
        Currently, the idea is to progressively apply filtering to achieve
        '''
        assert action.lower() in ['check', 'apply'], 'Invalid action for detect_post()'

        def isolate_primary(action: str):
            assert action in ['check', 'apply'], 'Invalid action for isolate_primary()'
            if action.lower() == 'check': self.logger.debug(f'Attempting to isolate primary scans for improved detection | {self.Session_ID}')
            primary = self.dicom_table.loc[(self.dicom_table['Post_scan'] == 1) & (self.dicom_table['Type'].str.contains('PRIMARY', na=False))]
            if action.lower() == 'check':
                if check_slices(primary, action=action) and check_orientation(primary, action=action):
                    return True
                else:
                    return False
            elif action.lower() == 'apply':
                if check_slices(primary, action=action) and check_orientation(primary, action=action):
                    self.dicom_table = self.dicom_table.loc[(self.dicom_table['Post_scan'] == 0) | (self.dicom_table['Type'].str.contains('PRIMARY', na=False))]
                    self.removed['Not_primary_post'] = self.dicom_table.loc[(self.dicom_table['Post_scan'] == 1) & (~self.dicom_table['Type'].str.contains('PRIMARY', na=False))]
                    self.logger.debug(f'=== Removed {len(self.removed["Not_primary_post"])} non-primary post scans | {self.Session_ID}')
                    return True
            
        def post_trigger_time(cumulative: bool = False, action: str = 'check'):
            # Trigger time approach
            assert action.lower() in ['check', 'apply'], 'Invalid action for post_trigger_time()'
            if cumulative:
                self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'Post_scan'] = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1, 'TriTime'].str.lower() != 'unknown'
            else:
                self.dicom_table['Post_scan'] = self.dicom_table['TriTime'].str.lower() != 'unknown'
            post_found = self.dicom_table['Post_scan'].to_numpy()
            if action.lower() == 'check': self.logger.debug(f'Trigger time post scan detection found {post_found.sum()} post scans | {self.Session_ID}')
            return post_found

        def post_series_desc(cumulative: bool = False, action: str = 'check'):
            # Series Description approach
            assert action.lower() in ['check', 'apply'], 'Invalid action for post_series_desc()'

            series_lower = self.dicom_table['Series_desc'].astype(str).str.lower()
            contains_post = series_lower.str.contains('post', na=False).astype(int)

            if cumulative:
                mask = self.dicom_table['Post_scan'] == 1
                self.dicom_table.loc[mask, 'Post_scan'] = contains_post[mask]
            else:             
                self.dicom_table['Post_scan'] = contains_post
            post_found = self.dicom_table['Post_scan'].to_numpy()
            self.logger.debug(f'Series Description post scan detection found {post_found.sum()} post scans | {self.Session_ID}')
            return post_found
        
        def check_slices(tmp_table: pd.DataFrame, action: str = 'check'):
            '''
            Checks that only a single quantity of slices is represented across post sequence
            May be two quantities if multiple laterality are present
                - Will pass if self.multiple_lat is True (expecting two)
                - Will fail if self.multiple_lat is False (expecting one)
            
            When two quantites and self.multiple_lat is false, self.detect_pre() should be run to set correct expectation
            '''
            assert action.lower() in ['check', 'apply'], 'Invalid action for check_slices()'
            available_slices = tmp_table['NumSlices'].unique()
            self.post_slices = available_slices
            if len(available_slices) == 0:
                return False
            if (len(available_slices) == 1) | ((len(available_slices) == 2) & self.multiple_lat):
                if action.lower() == 'check': self.logger.debug(f'Detected single slice number estimation for post sequence [2 if both sides present] {available_slices} | {self.Session_ID}')
                return True
            else:
                # Check if laterality is ambigious but existing, suggest running pre detection to detect optimal slice
                if action.lower() == 'check': self.logger.debug(f'Detected post sequence has different slices among scans {available_slices} | {self.Session_ID}')
                return False
        
        def check_orientation(tmp_table: pd.DataFrame, action: str = 'check'):
            '''
            Checks if there is a single orientation represented within the post scans
            '''
            assert action.lower() in ['check', 'apply'], 'Invalid action for check_slices()'
            available_orientations = tmp_table['Orientation'].unique()
            if len(available_orientations) == 0:
                return False
            if len(available_orientations) == 1:
                if action.lower() == 'check': self.logger.debug(f'Detected single orientation for post series | {self.Session_ID}')
                return True
            else:
                if action.lower() == 'check': self.logger.debug(f'Detected multiple orientations within post sequence | {self.Session_ID}')
                return False

        ########
        post_found_tt = post_trigger_time(action = action)
        if post_found_tt.sum() >= 1:
            if check_slices(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1], action=action) and check_orientation(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1], action=action):
                return True
            elif isolate_primary(action = action):
                return True
            else:
                if action.lower() == 'check': self.logger.debug(f'Multiple post candidates lack consistent slice numbering [trigger time detection]... | {self.Session_ID}')

        post_found_sd = post_series_desc(action = action)
        if post_found_sd.sum() >= 1:
            if check_slices(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1], action=action) and check_orientation(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1], action=action):
                return True
            elif isolate_primary(action = action):

                return True
            else:
                if action.lower() == 'check': self.logger.debug(f'Multiple post candidates lack consistent slice numbering [series desc detection]... | {self.Session_ID}')

        
        if action.lower() == 'check': self.logger.error(f'Trigger time and series desc failed to find any post scans | {self.Session_ID}')
        if action.lower() == 'apply':
            self.removed['Post_Failure'] = self.dicom_table
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
        return False
        ########

    def isolate_sequence(self) -> bool:
        """
        Aims to isolate the MRI sequence from the given session.

        Uses detect_pre and detect_post to determine filtering.
        Successfully detected pre/post are extracted out of self.dicom_table
        into placeholder tables.

        Returns:
            bool: True if sequence isolation was successful, False otherwise.

        TODO: Structural recommendation: This function is very long and has complex
              branching logic. Breaking it down into smaller, testable sub-functions
              (e.g., `resolve_laterality`, `resolve_slice_counts`) would improve
              maintainability and testability.
        """
        # SHOULD THIS FUNCTION FLOW DEPEND ON SESSION PROTOCOL [16-328, 19-093, 20-425]
        self.print_table(columns=['Session_ID', 'Series_desc', 'NumSlices', 'Lat', 'Orientation', 'TriTime', 'Type', 'Series'])

        ### Fixing laterality from series description
        mask_unknown = self.dicom_table['Lat'].fillna('').str.lower() == 'unknown'
        self.dicom_table.loc[mask_unknown & self.dicom_table['Series_desc'].fillna('').str.lower().str.contains('left', na=False), 'Lat'] = 'left'
        self.dicom_table.loc[mask_unknown & self.dicom_table['Series_desc'].fillna('').str.lower().str.contains('right', na=False), 'Lat'] = 'right'
        self.dicom_table.loc[mask_unknown & self.dicom_table['Series_desc'].fillna('').str.lower().str.contains('bilateral', na=False), 'Lat'] = 'bilateral'
        if self.dicom_table['Lat'].nunique() > 1:
            self.logger.debug(f'Multiple laterality represented in dicom data, need to seperate... | {self.Session_ID}')
            self.multiple_lat = True

        self.dicom_table['Post_scan'] = 0
        self.dicom_table['Pre_scan'] = 0

        # FINDING POST SEQUENCE
        post_success = self.detect_post('check')
        if (not post_success) and (not self.multiple_lat):
            # If post unsuccesfull and single laterality suggested, check pre to identify hidden multiple lateralities
            self.logger.debug(f'Post scan detection failure, unable to isolate sequence, checking if solvable with pre... | {self.Session_ID}')
            self.dicom_table['Post_scan'] = 0 #reset post scan detection
            pre_success = self.detect_pre('check')
            if pre_success and self.multiple_lat:
                # If pre detection succeeded, filter by detected number of slices
                self.apply_slices(use='pre')
                post_success = self.detect_post('check')
                if post_success:
                    self.logger.debug(f'Post sequence detection solved by finding pre first... | {self.Session_ID}')
                else:
                    self.logger.error(f'Unable to solve post detection by solving pre first... | {self.Session_ID}')
                    self.removed['Post_Failure'] = self.dicom_table
                    self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
                    return False
            elif pre_success:
                self.logger.debug(f'Pre scan detection passed before post, no change in laterality detection... | {self.Session_ID}')
                self.apply_slices(use='pre')
                post_success = self.detect_post('check')
                if post_success:
                    self.logger.debug(f'Post sequence detection solved by finding pre first... | {self.Session_ID}')
                else:
                    self.logger.error(f'Unable to solve post detection by solving pre first... | {self.Session_ID}')
                    self.removed['Post_Failure'] = self.dicom_table
                    self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
                    return False
            elif self.multiple_lat:
                # If detect_pre flipped multiple_lat
                self.logger.debug(f'Multiple laterality discerned from pre status, pre failure...| {self.Session_ID}')
                self.dicom_table['Pre_scan'] = 0
                post_success = self.detect_post('check')
                if post_success:
                    # If post detection now workd, continue to applying post detection
                    self.logger.debug(f'Post detection failure ameliorated through laterality separation | {self.Session_ID}')
                    self.detect_post('apply')
                    self.dicom_post = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1]
                    self.dicom_post['Post_scan'] = True
                    self.dicom_table = self.dicom_table.loc[self.dicom_table['Post_scan'] == 0]
                    self.dicom_table['Post_scan'] = False
                    self.apply_slices(use='post')
                    self.logger.debug(f'Successfully detected post sequence | {self.Session_ID}')
                    self.print_table(self.dicom_post, columns=['Session_ID', 'Series_desc', 'NumSlices', 'Lat', 'Orientation', 'TriTime', 'Type', 'Series', 'Post_scan'])
                else:
                    # If post detection still fails, remove
                    self.logger.error(f'Failure to solve by performing pre first | {self.Session_ID}')
                    self.removed['Post_Failure'] = self.dicom_table
                    self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
                    return False
            else:
                self.logger.error(f'Pre detection fails... | {self.Session_ID}')
                self.removed['Pre_Failure'] = self.dicom_table
                self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
                return False


        elif (not post_success):
            # Post failure with multiple lateralites detected, cuurrently unable to continue
            self.logger.error(f'Failure in detecting post sequence, clearing entry... | {self.Session_ID}')
            self.removed['Post_Failure'] = self.dicom_table
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
            return False
        
        # Post sequence can be determined immediately, detect and filter
        self.detect_post('apply')
        self.apply_slices(use='post')
        self.dicom_post = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1]
        self.dicom_post['Post_scan'] = True
        self.dicom_table = self.dicom_table.loc[self.dicom_table['Post_scan'] == 0]
        self.dicom_table['Post_scan'] = False
        self.logger.debug(f'Successfully detected post sequence | {self.Session_ID}')
        self.print_table(self.dicom_post, columns=['Session_ID', 'Series_desc', 'NumSlices', 'Lat', 'Orientation', 'TriTime', 'Type', 'Series', 'Post_scan'])
        #self.print_table(self.dicom_table, columns=['Session_ID', 'Series_desc', 'NumSlices', 'Lat', 'Orientation', 'TriTime', 'Type', 'Series', 'Post_scan'])
    


        # ISOLATE NON_POST FOR PRE DETECTION
        
        pre_success = self.detect_pre('check')
        if not pre_success:
            self.logger.error(f'Failure in detecting pre sequence | {self.Session_ID}')
            self.removed['Pre_Failure'] = self.dicom_table
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
            return False
        elif pre_success:
            self.detect_pre('apply')
            self.dicom_pre = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1]
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
            self.logger.debug(f'Successfully detected pre sequence | {self.Session_ID}')
            self.print_table(self.dicom_pre, columns=['Session_ID', 'Series_desc', 'NumSlices', 'Lat', 'Orientation', 'TriTime', 'Type', 'Series', 'Pre_scan'])

        self.dicom_pre['Pre_scan'] = True
        self.dicom_post['Post_scan'] = True
        self.dicom_table = pd.concat([self.dicom_pre, self.dicom_post])

        # FINDING NUMBER OF SLICES - not needed anymore? solved by .apply_slices()?
        expected_slices = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].unique()
        expected_slices = [x for x in expected_slices if not any((x != y) and (x % y == 0) for y in expected_slices)] # new addition
        self.removed['invalid_slices'] = self.dicom_table[~self.dicom_table['NumSlices'].apply(lambda x: any(s % x == 0 for s in expected_slices))]
        self.dicom_table = self.dicom_table[self.dicom_table['NumSlices'].apply(lambda x: any(s % x == 0 for s in expected_slices))]

        if len(expected_slices) > 2:
            self.logger.debug(f'Multiple post scans with different number of slices detected {expected_slices} | {self.Session_ID}')
            self.removed['Multiple_post_slices'] = self.dicom_table.copy()
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
            return False
        elif (len(expected_slices) == 2):
            if not self.multiple_lat:
                self.logger.debug(f'Only one detected side {self.dicom_table.loc[self.dicom_table["Post_scan"] == 1, "Lat"].unique()} for multiple expected slice numbers {expected_slices} | {self.Session_ID}')
                # Seperate into distinct unknowns based on laterality
                self.dicom_table.loc[self.dicom_table['NumSlices'].apply(lambda x: expected_slices[0] % x == 0), 'Lat'] = 'Unknown_A'
                self.dicom_table.loc[self.dicom_table['NumSlices'].apply(lambda x: expected_slices[1] % x == 0), 'Lat'] = 'Unknown_B'
            
            sides = self.dicom_table['Lat'].unique()
            if len(sides) <= 2:
                self.logger.debug(f'Solved laterality for multiple expected slice numbers {expected_slices} | {self.Session_ID}')

            elif len(sides) > 2:
                self.logger.debug(f'Too many lateralities detected {sides} | {self.Session_ID}')
                self.removed['Multiple_laterality'] = self.dicom_table.copy()
                self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
                return False

        elif len(expected_slices) == 1:
            self.logger.debug(f'Detected single slice number expectation {expected_slices[0]} for post scans | {self.Session_ID}')

        # Detecting scan orientation based on post scans
        orientations = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'Orientation'].unique()
        if len(orientations) == 1:
            self.logger.debug(f'Single orientation detected {orientations[0]} for post scans | {self.Session_ID}')
        elif len(orientations) > 1:
            self.logger.debug(f'Miltiple orientations detected {orientations} for post scans | {self.Session_ID}')
            major_orientation = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'Orientation'].mode()[0]
            self.logger.debug(f'Keeping major orientation {major_orientation} for post scans | {self.Session_ID}')
            self.removed['invalid_orientations'] = self.dicom_table.loc[(self.dicom_table['Orientation'] != major_orientation)]
            self.dicom_table = self.dicom_table.loc[(self.dicom_table['Orientation'] == major_orientation)]

        non_post = len(self.dicom_table)
        if non_post == 1 or (non_post == 2 and self.multiple_lat):
            self.logger.debug(f'Single non-post scan detected per detected side, keeping for now... | {self.Session_ID}')
            self.dicom_table['Pre_scan'] = 1
        else:
            self.logger.debug(f'Too many scans remaining after post detection, attempting to isolate pre... | {self.Session_ID}')
            self.detect_pre()
            if self.dicom_table.empty:
                self.logger.debug(f'Pre scan detection failure, unable to isolate sequence | {self.Session_ID}')
                return False
        self.dicom_table = pd.concat([self.dicom_post, self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1]])

        # self.dicom_table = self.dicom_table.loc[(self.dicom_table['Post_scan'] == 1)|(self.dicom_table['Pre_scan'] == 1)]
        laterality = self.dicom_table.loc[self.dicom_table['Pre_scan']  == 1, 'Lat'].unique()
        if len(laterality) > 1:
            self.logger.debug(f'Multiple sides detected {laterality}, appending Session_ID with laterality | {self.Session_ID}')
            SessionID = self.Session_ID
            for lat in laterality:
                if lat.lower() == 'left':
                    self.dicom_table.loc[self.dicom_table['Lat'].str.lower() == lat.lower(), 'SessionID'] = str(SessionID[0]) + '_l'
                elif lat.lower() == 'right':
                    self.dicom_table.loc[self.dicom_table['Lat'].str.lower() == lat.lower(), 'SessionID'] = str(SessionID[0]) + '_r'
                elif lat.lower() == 'unknown_a':
                    self.dicom_table.loc[self.dicom_table['Lat'].str.lower() == lat.lower(), 'SessionID'] = str(SessionID[0]) + '_a'
                elif lat.lower() == 'unknown_b':
                    self.dicom_table.loc[self.dicom_table['Lat'].str.lower() == lat.lower(), 'SessionID'] = str(SessionID[0]) + '_b'
                elif lat.lower() == 'bilateral':
                    self.logger.error(f'Multiple laterality detected including bilateral, unable to append Session_ID [unexpected behavior] | {self.Session_ID}')



        self.print_table(columns=['Series_desc', 'NumSlices', 'Lat', 'Orientation', 'TriTime', 'Type', 'Series', 'IS_DISCO', 'Pre_scan', 'Post_scan'])
        return True



class DICOMsplit():
    """
    Class used to split a single directory with multiple scans into multiple directories
    of a single scan each.
    """
    def __init__(self, dicom_table: pd.DataFrame,  logger: logging.Logger = None, debug: int = 0, tmp_save: str='/FL_system/data/tmp/'):
        """
        Initialize the splitter. The received DataFrame should contain a single
        SessionID and directory.

        Args:
            dicom_table (pd.DataFrame): Data containing a single SessionID.
            logger (logging.Logger): Logger instance.
            debug (int): Debug flag.
            tmp_save (str): Path to temporary save directory.

        TODO: Memory/Performance: Loading and parsing glob on potentially massive
              directories multiple times can be slow. A file index or caching mechanism
              should be implemented if processing scales up.
        """
        ## Updated process
        self.scan_path = None
        self.scan_results = None
        self.tmp_save = tmp_save

        self.logger = logger or logging.getLogger(__name__)
        if dicom_table.empty:
            raise ValueError('The provided dicom_table is empty.')
        if dicom_table['SessionID'].nunique() != 1:
            raise ValueError('Multiple Session_IDs found in the table')
        self.dicom_table = dicom_table.reset_index(drop=True)
        self.Session_ID = self.dicom_table['SessionID'].unique()[0]
        # Get the common element of all paths
        self.directory = os.path.commonpath(self.dicom_table['PATH'].tolist())
        self.logger.debug(f'Found common path: {self.directory} | [{self.Session_ID}]')
        # Legacy path-correction removed.
        # Previously this block attempted to rewrite paths for datasets imported from other systems
        # (MSKCC_16-328, RIA_19-093, RIA_20-425). Path normalization should be handled upstream
        # (when constructing the DataFrame) or via a dedicated migration script. If live
        # corrections are required again, reintroduce a small, well-tested helper here.

        # Determine expectations for the scan
        self.scan_path = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'PATH'].values[0]
        # Remove file from path to get directory
        self.scan_path = os.path.dirname(self.scan_path)

        self.pre_slices = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1, 'NumSlices'].unique()[0]

        # Determine if scanning is required
        if all(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'] == self.pre_slices):
            self.logger.debug(f'Pre and post scans have the same number of slices, no need to split | [{self.Session_ID}]')
            self.SCAN = False
        elif (len(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].unique()) == 1) and(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].unique()[0] % self.pre_slices == 0):
            self.logger.debug(f'Post scans have different number of slices, scanning required | [{self.Session_ID}]')
            self.SCAN = True
            self.logger.debug(f'Set scan path to: {self.scan_path} | [{self.Session_ID}]')
            self.num_post_scans = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].values[0] // self.pre_slices
        else:
            self.logger.warning(f'Unable to make sense of pre and post scans, removing session, further logic required | [{self.Session_ID}]')
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
            self.SCAN = False

    def scan_all(self):
        """Scans all files in the directory"""
        # If self.scan path doesnt exist, raise error
        if self.scan_path is None or not os.path.exists(self.scan_path):
            self.logger.error(f'Scan path does not exist: {self.scan_path} | [{self.Session_ID}]')
            return

        files = glob.glob(os.path.join(self.scan_path, '*.dcm'))
        info = {
            'PATH': [],
            'AcqTime': [],
            'SrsTime': [],
            'ConTime': [],
            'StuTime': [],
            'TriTime': [],
            'InjTime': [],
            'Series': [],
        }
        for file in files:
            extractor = DICOMextract(file)
            info['PATH'].append(file)
            info['AcqTime'].append(extractor.Acq())
            info['SrsTime'].append(extractor.Srs())
            info['ConTime'].append(extractor.Con())
            info['StuTime'].append(extractor.Stu())
            info['TriTime'].append(extractor.Tri())
            info['InjTime'].append(extractor.Inj())
            info['Series'].append(extractor.Series())
            del extractor
        self.scan_results = pd.DataFrame(info)
        self.logger.debug(f'Found {len(self.scan_results)} DICOM files in the directory | [{self.Session_ID}]')
        if self.scan_results is None or self.scan_results.empty:
            self.logger.warning(f'Error scanning {self.scan_path} | [{self.Session_ID}]')
            return 
        else:
            if not os.path.exists(f'{self.tmp_save}/directory_scan/'):
                os.makedirs(f'{self.tmp_save}/directory_scan/')
            self.scan_results.to_csv(f'{self.tmp_save}/directory_scan/{self.Session_ID}.csv', index=False)
            return 
    
    def sort_scans(self, scan_results: pd.DataFrame = None):
        """Sorts the scans based on the Series number"""
        self.scan_results['Slice'] = self.scan_results['PATH'].apply(lambda x: str(x).split('-')[-2])
        self.scan_results['Slice'] = self.scan_results['Slice'].apply(lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in str(x)) else -1)
        self.temporary_relocations = []
        # Remove multi-scan entry from data table
        self.dicom_table = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1]
        if self.scan_results['Series'].nunique() == self.num_post_scans:
            self.logger.debug(f'Found expected number of post scans {self.num_post_scans} using Series number | [{self.Session_ID}]')
            self.logger.error(f'Additional logic required to sort based on Series number | [{self.Session_ID}]')
        elif self.scan_results['TriTime'].nunique() == self.num_post_scans:
            self.logger.debug(f'Found expected number of post scans {self.num_post_scans} using Trigger time | [{self.Session_ID}]')
            for i in self.scan_results['TriTime'].unique():
                slices = self.scan_results.loc[self.scan_results['TriTime'] == i, 'Slice'].to_list()
                slices.sort()
                if len(slices) != self.pre_slices:
                    self.logger.warning(f'Unexpected number of slices {len(slices)} found for trigger time {i}, expected {self.pre_slices} | [{self.Session_ID}]')
                    return
                # Add new row to the dicom table
                self.dicom_table = pd.concat([self.dicom_table, self.scan_results.loc[(self.scan_results['TriTime'] == i)&(self.scan_results['Slice'] == slices[0])]], ignore_index=True)
                self.dicom_table.loc[(self.dicom_table['TriTime'] == i)&(self.dicom_table['Slice'] == slices[0]), 'PATH'] = f"{self.tmp_save}/dicom/{self.Session_ID}/{i}/{str(slices[0]).zfill(3)}.dcm"
                self.logger.debug(f'{len(slices)} files found for trigger time {i} | [{self.Session_ID}]')
                for j in slices:
                    initial = self.scan_results.loc[(self.scan_results['TriTime'] == i) & (self.scan_results['Slice'] == j), 'PATH'].values[0]
                    # pad j to a 3 digit number
                    j = str(j).zfill(3)
                    destination = f"{self.tmp_save}dicom/{self.Session_ID}/{i}/{j}.dcm"
                    self.temporary_relocations.append([initial, destination])
            self.dicom_table['SessionID'] = self.Session_ID
        return 

        ## Below is old process, kept for reference
        # if self.scan_results['Series'].nunique() == 1:
        #     self.logger.debug(f'Single series found | [{self.Session_ID}]')
        #     if self.scan_results['TriTime'].nunique() > 1:
        #         self.logger.debug(f' {self.scan_results["TriTime"].nunique()} unique trigger times found | [{self.Session_ID}]')
        #         timing = self.scan_results['TriTime'].unique()
                
        #         for i in range(len(timing)):
        #             if not os.path.exists(f'{self.tmp_save}{self.Session_ID}/{i}'):
        #                os.makedirs(f'{self.tmp_save}{self.Session_ID}/{i}')
        #             new_table = self.scan_results[self.scan_results['TriTime'] == timing[i]]
        #             files = new_table['PATH'].to_list()
        #             self.logger.debug(f'Found {len(files)} files for trigger time {timing[i]} | [{self.Session_ID}]')
        #             for j in range(len(files)):
        #                 shutil.copy(files[j], f"{self.tmp_save}{self.Session_ID}/{i}")
        #                 self.temporary_relocations.append((files[j], f"{self.tmp_save}{self.Session_ID}/{i}"))
        #                 if j == 0:
        #                     grab first scan result for 
        #                     info = {}
        #                     info['PATH'] = f"{self.tmp_save}{self.Session_ID}/{i}"
        #                     extractor = DICOMextract(files[j])
        #                     info['AcqTime'] = (extractor.Acq())
        #                     info['SrsTime'] = (extractor.Srs())
        #                     info['ConTime'] = (extractor.Con())
        #                     info['StuTime'] = (extractor.Stu())
        #                     info['TriTime'] = (extractor.Tri())
        #                     info['InjTime'] = (extractor.Inj())
        #                     info['Series'] = extractor.Series()
        #                     info['ID'] = extractor.ID()
        #                     info['DATE'] = extractor.Date()
        #                     info['SessionID'] = self.Session_ID
        #                     self.output_rows.append(info)
        #             save the first file to the output table
        #         print(self.scan_results)
        #     print(self.scan_results)
        # else:
        #     self.logger.info(f'Multiple series found | [{self.Session_ID}]')
        #     for i in self.scan_results['Series'].unique():
        #         if not os.path.exists(f'{self.tmp_save}{self.Session_ID}/{i}'):
        #            os.makedirs(f'{self.tmp_save}{self.Session_ID}/{i}')
        #         slices = self.scan_results[self.scan_results['Series'] == i]
                
        #         for j in range(len(slices)):            
        #             copy each file to its series folder
        #             file = slices['PATH'].iloc[j]
        #             shutil.copy(file, f"{self.tmp_save}{self.Session_ID}/{i}")
        #             self.temporary_relocations.append((file, f"{self.tmp_save}{self.Session_ID}/{i}"))
        #             if j == 0:
        #                 save the first file to the output table
        #                 info = self.dicom_table.iloc[0].to_dict()
        #                 extractor = DICOMextract(file)
        #                 info['PATH'] = f"{self.tmp_save}{self.Session_ID}/{i}"
        #                 info['NumSlices'] = len(slices)
        #                 info['AcqTime'] = self.scan_results['AcqTime'].iloc[j]
        #                 info['SrsTime'] = self.scan_results['SrsTime'].iloc[j]
        #                 info['ConTime'] = self.scan_results['ConTime'].iloc[j]
        #                 info['StuTime'] = self.scan_results['StuTime'].iloc[j]
        #                 info['TriTime'] = self.scan_results['TriTime'].iloc[j]
        #                 info['InjTime'] = self.scan_results['InjTime'].iloc[j]
        #                 info['Series'] = i
        #                 self.output_rows.append(info)
        # return


class DICOMorder():
    """
    Class to order the DICOM scans sequentially based on acquisition or trigger times.
    """
    def __init__(self, dicom_table: pd.DataFrame, logger: logging.Logger = None, debug: int = 0):
        """
        Initialize the orderer for a given SessionID.

        Args:
            dicom_table (pd.DataFrame): Data containing a single SessionID.
            logger (logging.Logger): Logger instance.
            debug (int): Debug flag.
        """
        self.debug = debug
        self.dicom_table = dicom_table
        self.logger = logger or logging.getLogger(__name__)
        try:
            self.Session_ID = self.dicom_table['SessionID'].unique()
        except KeyError:
            self.logger.error(f'No SessionID column present {self.dicom_table.columns}')
            return  # Exit the constructor if SessionID is missing
        if self.Session_ID.size > 1:
            print('Multiple Session_IDs found in the table')
            print('Not currently implemented, please remake with a single Session_ID')
            return None
    
    def order(self, timing_param: str, secondary_param: str) -> pd.DataFrame:
        """
        Orders the scans based on the provided timing parameter.

        Args:
            timing_param (str): The primary timing parameter to sort by.
            secondary_param (str): Fallback parameter if the primary is missing.

        Returns:
            pd.DataFrame: The ordered dataframe.

        TODO: Edge case: if both timing_param and secondary_param values are corrupted,
              ordering fails entirely and returns an empty table. Consider using
              SeriesNumber as an absolute fallback before giving up.
        """
        # Separate rows with 'UNKNOWN' values
        self.timing_param = timing_param
        unknown_rows = self.dicom_table[self.dicom_table[timing_param].astype(str).str.lower() == 'unknown']
        valid_rows_index = self.dicom_table[self.dicom_table[timing_param].astype(str).str.lower() != 'unknown'].index
        self.logger.debug(f'Found {len(unknown_rows)} rows with unknown {timing_param} values | {self.Session_ID}')
        self.logger.debug(f'Found {len(valid_rows_index)} rows with known {timing_param} values | {self.Session_ID}')
        if len(valid_rows_index) == 0:
            self.logger.debug(f'No valid {timing_param} values found | {self.Session_ID}')
            self.logger.debug(f'Utilizing secondary parameter {secondary_param} for ordering | {self.Session_ID}')
            valid_rows_index_2 = self.dicom_table[self.dicom_table[secondary_param].astype(str).str.lower() != 'unknown'].index
            unknown_rows_2 = self.dicom_table[self.dicom_table[secondary_param].astype(str).str.lower() == 'unknown']
            if len(valid_rows_index_2) == 0:
                self.logger.error(f'Unable to order scans with {self.timing_param} or {secondary_param}, returning empty table | {self.Session_ID}')
                self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
                return self.dicom_table
            elif len(valid_rows_index_2) == len(self.dicom_table):
                self.logger.debug(f'All rows have valid {secondary_param} values, ordering by {secondary_param} | {self.Session_ID}')
                # Convert the secondary_param column to integers
                self.dicom_table[secondary_param] = self.dicom_table[secondary_param].astype(str).str.split('.').str[0]  # Remove decimal part if present
                self.dicom_table[secondary_param] = self.dicom_table[secondary_param].astype(int)
                valid_rows = self.dicom_table.sort_values(by=[secondary_param])
                self.n_post = len(valid_rows)
                self.dicom_table['Major'] = np.arange(0, len(valid_rows)) # Start at 0 since all scans (including pre) are included
                return self.dicom_table
        elif len(valid_rows_index) == len(self.dicom_table) - 1:
            self.logger.debug(f'All rows have valid {timing_param} values [except for pre], ordering by {timing_param} | {self.Session_ID}')

            # Convert the timing_param column to integers for valid rows
            self.dicom_table.loc[valid_rows_index, timing_param] = self.dicom_table.loc[valid_rows_index, timing_param].astype(int)

            # Sort the valid rows
            valid_rows = self.dicom_table.loc[valid_rows_index].sort_values(by=[timing_param])
            self.n_post = len(valid_rows)

            # Add a 'Major' column to the valid rows
            self.dicom_table.loc[valid_rows.index, 'Major'] = np.linspace(1, len(valid_rows), int(len(valid_rows)))
            self.dicom_table.loc[unknown_rows.index, 'Major'] = np.zeros(len(unknown_rows))

            return self.dicom_table
        else:
            self.logger.error(f'Unexpected results for {self.timing_param} values, unable to order scans | {self.Session_ID}')
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
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
        elif len(unknown_rows) == 2:
            print(f'Found two unknown values for {self.dicom_table["SessionID"].unique()}')
            print(f'Analyzing to see if either row includes "FS" in the description')
            for i in range(len(unknown_rows)):
                if 'FS' not in unknown_rows['Series_desc'].iloc[i]:
                    if self.debug > 0:
                        print(f'Found a FS scan for {self.dicom_table["SessionID"].unique()}')
                        print(f'Assuming this is a pre scan')
                    # remove the other row
                    unknown_rows = unknown_rows.drop(unknown_rows.index[i])
                    break
            # check if one of the unknown values is a FS scan
        # return index of unknown row
        return unknown_rows.index              

    def findPre(self):
        indx = self.dicom_table[self.dicom_table['Post_scan'] == 1].index
        pre_indx = self.dicom_table[self.dicom_table['Pre_scan'] == 1].index

        if len(pre_indx) == 1:
            indx = np.append(indx, pre_indx)
            self.dicom_table = self.dicom_table.loc[indx]
            return self.dicom_table
        else:
            self.logger.debug(f'Pre scan detection failed | {self.Session_ID}')
            self.logger.debug(f'Removing session from consideration | {self.Session_ID}')
            self.logger.warning(f'Pre scan should be found, possible error in earlier processing steps | {self.Session_ID}')
            self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
            return self.dicom_table

        # series_numbers = self.dicom_table['Series'][self.dicom_table['Major'] > 0].astype(int)
        # if len(series_numbers) == 0:
        #     if self.debug > 0:
        #         print(f'No series_numbers found for {self.dicom_table["SessionID"].unique()}')
        #     #clear dicom table
        #     self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)
        #     return self.dicom_table
        # indx = self.dicom_table[self.dicom_table['Major'] > 0].index

        # # sort series numbers
        # series_numbers = series_numbers.sort_values()
        # pre_value = series_numbers.iloc[0] - 1

        # pre_indx = self.dicom_table[self.dicom_table['Series'] == pre_value].index
        # if len(pre_indx) == 1:
        #     indx = np.append(indx, pre_indx)
        #     self.dicom_table = self.dicom_table.loc[indx]
        #     return self.dicom_table
        # elif len(pre_indx) == 0:
        #     print(f'No pre scan found for {self.dicom_table["SessionID"].unique()}')
        #     print(f'Attempting alternate pre scan detection')
        #     pre_indx = self.alternate_pre()
        
        # if len(pre_indx) == 1:
        #     indx = np.append(indx, pre_indx)
        #     self.dicom_table = self.dicom_table.loc[indx]
        #     return self.dicom_table
        # else:
        #     print('Alternative pre scan detection failed')
        #     print('No pre scan found')
        #     print('Returning empty dicom table')
        #     self.dicom_table = pd.DataFrame(columns=self.dicom_table.columns)