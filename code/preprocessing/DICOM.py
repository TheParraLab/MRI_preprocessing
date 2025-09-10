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
    UNKNOWN = 'Unknown'

    def __init__(self, file_path: str, debug: int = 0):
        self.debug = debug
        self.metadata = pyd.dcmread(file_path, stop_before_pixels=True)
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
        """Attempts to extract the laterality of the scan"""
        # TODO: Does metadata contain the filepath or is the file path required as an argument?
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
    def __init__(self, dicom_table: pd.DataFrame, logger: logging.Logger = None, debug: int = 0, tmp_save: str='/FL_system/data/tmp/'):
        self.debug = debug
        self.logger = logger or logging.getLogger(__name__)
        self.dicom_table = dicom_table
        self.Session_ID = self.dicom_table['SessionID'].unique()
        self.SIDE = self.majorSide()
        self.removed = {}
        self.tmp_save = tmp_save
        self.temporary_relocations = []
        assert self.Session_ID.size == 1, 'Multiple Session_IDs found in the table'
        self.logger.debug(f'Analyzing {self.Session_ID}')
        # Add Valid column
        self.dicom_table['Valid'] = 1

    def update_valid(self, column: str):
        """Updates the Valid column based on the provided column"""
        if column in self.dicom_table.columns:
            self.dicom_table['Valid'] = np.where(self.dicom_table[column] == 1, 0, 1)
        else:
            self.logger.warning(f'Column {column} not found in the table')
        return self.dicom_table

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
            # Flag computed scans
            #self.dicom_table['Remove_Computed'] = np.where(self.dicom_table['Type'].str.contains(flag.upper(), na=False), 1, 0)
        # Concatenate all removed rows into a single DataFrame
        self.removed['Computed'] = pd.concat(removed, ignore_index=True)
        self.logger.debug(f'Removed {len(self.removed["Computed"])} scans with computed descriptions | {self.Session_ID}')
        #self.update_valid('Remove_Computed')
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

    def isolate_sequence(self):

        ### Remove localizer scans
        self.removed['Localizer'] = self.dicom_table[self.dicom_table['Series_desc'].fillna('').str.lower().str.contains('loc', na=False)]
        self.dicom_table = self.dicom_table[~self.dicom_table['Series_desc'].fillna('').str.lower().str.contains('loc', na=False)]

        ### Sessions must have at least 2 scans to be considered valid
        if len(self.dicom_table) < 2:
            self.logger.debug(f'Not enough scans to isolate sequence | {self.Session_ID}')
            self.removed['Too_few_scans'] = self.dicom_table.copy()
            self.dicom_table = pd.DataFrame()
            return self.dicom_table

        ### Fixing laterality from series description
        mask_unknown = self.dicom_table['Lat'].fillna('').str.lower() == 'unknown'
        self.dicom_table.loc[mask_unknown & self.dicom_table['Series_desc'].fillna('').str.lower().str.contains('left', na=False), 'Lat'] = 'left'
        self.dicom_table.loc[mask_unknown & self.dicom_table['Series_desc'].fillna('').str.lower().str.contains('right', na=False), 'Lat'] = 'right'
        self.dicom_table.loc[mask_unknown & self.dicom_table['Series_desc'].fillna('').str.lower().str.contains('bilateral', na=False), 'Lat'] = 'bilateral'
        ### Use Series Description to identify pre and post scans   
        self.dicom_table['Post_scan'] = self.dicom_table['Series_desc'].apply(lambda x: 1 if 'post' in str(x).lower() else 0)
        self.dicom_table['Pre_scan'] = self.dicom_table['Series_desc'].apply(lambda x: 1 if 'pre' in str(x).lower() else 0)

        mask_both = (self.dicom_table['Post_scan'].astype(int) == 1) & (self.dicom_table['Pre_scan'].astype(int) == 1)
        if mask_both.sum() > 0:
            self.logger.debug(f'Scans identified as both pre and post scans | {self.Session_ID}')
            mask_tri_not_unknown = mask_both & (self.dicom_table['TriTime'] != 'Unknown')
            mask_tri_unknown = mask_both & (self.dicom_table['TriTime'] == 'Unknown')
            self.dicom_table.loc[mask_tri_not_unknown, 'Pre_scan'] = 0
            self.dicom_table.loc[mask_tri_unknown, 'Post_scan'] = 0
            mask_both_after = (self.dicom_table['Post_scan'].astype(int) == 1) & (self.dicom_table['Pre_scan'].astype(int) == 1)
            if mask_both_after.sum() > 0:
                self.logger.debug(f'Scans still identified as both pre and post scans after correction | {self.Session_ID}')
                self.removed['Pre_post_conflict'] = self.dicom_table.copy()
                self.dicom_table = pd.DataFrame()
                return self.dicom_table
            elif mask_both_after.sum() == 0:
                self.logger.debug(f'Trigger time correction successfully removed pre and post scan conflict | {self.Session_ID}')

        ### Post scan processing
        if (self.dicom_table['Post_scan'] == 0).all(): # If no post scan
            ## Secondary post sequence detection mechanism
            self.logger.debug(f'No post scan found in Series_desc applying secondary mechanism | {self.Session_ID}')
            self.dicom_table.loc[(self.dicom_table['Post_scan'] == 0)&(self.dicom_table['TriTime'] != 'Unknown'), 'Post_scan'] = 1
            if (self.dicom_table['Post_scan'] == 1).any():
                self.logger.debug(f'Detected post sequence based on trigger time | {self.Session_ID}')
            else:
                self.logger.debug(f'Secondary post scan detection failure | {self.Session_ID}')
                self.removed['No_post'] = self.dicom_table.copy()
                self.dicom_table = pd.DataFrame()
                return self.dicom_table

        expected_slices = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].unique()
        expected_slices = [x for x in expected_slices if not any((x != y) and (x % y == 0) for y in expected_slices)] # new addition

        self.removed['invalid_slices'] = self.dicom_table[~self.dicom_table['NumSlices'].apply(lambda x: any(s % x == 0 for s in expected_slices))]
        self.dicom_table = self.dicom_table[self.dicom_table['NumSlices'].apply(lambda x: any(s % x == 0 for s in expected_slices))]

        if len(expected_slices) > 2:
            self.logger.debug(f'Multiple post scans with different number of slices detected {expected_slices} | {self.Session_ID}')
            self.removed['Multiple_post_slices'] = self.dicom_table.copy()
            self.dicom_table = pd.DataFrame()
            return self.dicom_table

        elif len(expected_slices) == 2:
            laterality = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'Lat'].unique()
            if len(laterality) == 1:
                self.logger.debug(f'Only one detected side {laterality} for multiple expected slice numbers {expected_slices} | {self.Session_ID}')
                # Seperate into distinct unknowns based on laterality
                self.dicom_table.loc[self.dicom_table['NumSlices'].apply(lambda x: expected_slices[0] % x == 0), 'Lat'] = 'Unknown_A'
                self.dicom_table.loc[self.dicom_table['NumSlices'].apply(lambda x: expected_slices[1] % x == 0), 'Lat'] = 'Unknown_B'
            
            sides = self.dicom_table['Lat'].unique()
            if len(sides) <= 2:
                self.logger.debug(f'Solved laterality for multiple expected slice numbers {expected_slices} | {self.Session_ID}')

            elif len(sides) > 2:
                self.logger.debug(f'Too many lateralities detected {sides} | {self.Session_ID}')
                self.removed['Multiple_laterality'] = self.dicom_table.copy()
                self.dicom_table = pd.DataFrame()
                return self.dicom_table

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

        ### Pre scan processing
        if (self.dicom_table['Pre_scan'] == 0).all():
            ## TODO: Possible secondary pre sequence detection mechanism
            self.logger.debug(f'No pre scan found in Series_desc | {self.Session_ID}')
            self.logger.debug(f'Attempting secondary pre scan detection mechanism | {self.Session_ID}')
            # if left and right slices variables exist
            # TODO: Address excessive inclusion of LOC scans here
            if len(self.dicom_table.loc[(self.dicom_table['Pre_scan'] == 0)&(self.dicom_table['Post_scan'] == 0)]) != 0:
                self.dicom_table.loc[(self.dicom_table['Pre_scan'] == 0)&(self.dicom_table['Post_scan'] == 0), 'Pre_scan'] = 1
                self.logger.debug(f'Detected pre sequence based on slice number consistency with post scans | {self.Session_ID}')
            else:
                self.removed['No_pre'] = self.dicom_table.copy()
                self.dicom_table = pd.DataFrame()
                return self.dicom_table
        
        # Checking and solving multiple pre scans
        if (self.dicom_table['Pre_scan'] == 1).sum() == 1:
            self.logger.debug(f'Single pre scan found in Series_desc | {self.Session_ID}')
        elif (self.dicom_table['Pre_scan'] == 1).sum() > 1:
            self.logger.debug(f'Multiple pre scans found in Series_desc | {self.Session_ID}')
            laterality = self.dicom_table.loc[self.dicom_table['Pre_scan']  == 1, 'Lat'].unique()
            if len(laterality) == 1:
                # Check if slice number is different across pre_scans
                self.logger.debug(f'Only one unique side detected, checking for differences in slice numbers | {self.Session_ID}')
                slices = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1, 'NumSlices'].unique()
                if len(slices) == 2:
                    self.logger.debug(f'2 Distinct slice nnumbers detected, assuming different sides | {self.Session_ID}')
                    self.dicom_table.loc[(self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['NumSlices'] == slices[0]), 'Lat'] = 'Unknown_A'
                    self.dicom_table.loc[(self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['NumSlices'] == slices[1]), 'Lat'] = 'Unknown_B'
                elif len(slices) == 1:
                    self.logger.debug(f'single number of slices detected, checking for fat saturation | {self.Session_ID}')
                    # prefer to keep fat saturation as the pre scan
                    descs = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1, 'Series_desc'].values
                    if any('non fat sat' in desc.lower() for desc in descs):
                        self.logger.debug(f'Non-fat sat pre scan found, removing | {self.Session_ID}')
                        to_remove = self.dicom_table[self.dicom_table['Series_desc'].str.lower().str.contains('non fat sat') & (self.dicom_table['Pre_scan'] == 1)]
                        self.removed['Non_fat_sat'] = pd.concat([self.removed.get('Non_fat_sat', pd.DataFrame()), to_remove], ignore_index=True)
                        self.dicom_table = self.dicom_table[~(self.dicom_table['Series_desc'].str.lower().str.contains('non fat sat') & (self.dicom_table['Pre_scan'] == 1))]
                    elif any('fs' in desc.lower() for desc in descs):
                        self.logger.debug(f'Fat sat pre scan found, keeping | {self.Session_ID}')
                        to_remove = self.dicom_table[(~self.dicom_table['Series_desc'].str.lower().str.contains('fs')) & (self.dicom_table['Pre_scan'] == 1)]
                        self.removed['non_fat_sat'] = pd.concat([self.removed.get('non_fat_sat', pd.DataFrame()), to_remove], ignore_index=True)
                        self.dicom_table = self.dicom_table[(self.dicom_table['Series_desc'].str.lower().str.contains('fs') & (self.dicom_table['Pre_scan'] == 1))|self.dicom_table['Post_scan'] == 1]
                else:
                    self.logger.debug(f'Unexpected number of slices detected for pre scans, removing session | {self.Session_ID}')
                    self.removed['Multiple_pre_slices'] = self.dicom_table.copy()
                    self.dicom_table = pd.DataFrame()
                    return self.dicom_table
            elif len(laterality) > 1:
                # if multiple sides detected, check if there is a single pre scan for each side
                for lat in laterality:
                    # When more than 1 for the same side, check fat saturation
                    if (self.dicom_table.loc[(self.dicom_table['Lat'].str.lower() == lat.lower()) & (self.dicom_table['Pre_scan'] == 1)].shape[0] > 1):
                        self.logger.debug(f'Multiple pre scans found for side {lat}, checking fat saturation | {self.Session_ID}')
                        descs = self.dicom_table.loc[(self.dicom_table['Pre_scan'] == 1)&(self.dicom_table['Lat'].str.lower() == lat.lower()), 'Series_desc'].values
                        if any('non fat sat' in desc.lower() for desc in descs):
                            self.logger.debug(f'Non-fat sat pre scan found, removing | {self.Session_ID}')
                            to_remove = self.dicom_table[(self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['Lat'].str.lower() == lat.lower()) & (self.dicom_table['Series_desc'].str.lower().str.contains('non fat sat'))]
                            self.removed['Non_fat_sat'] = pd.concat([self.removed.get('Non_fat_sat', pd.DataFrame()), to_remove], ignore_index=True)
                            self.dicom_table = self.dicom_table[~((self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['Lat'].str.lower() == lat.lower()) & (self.dicom_table['Series_desc'].str.lower().str.contains('non fat sat')))]
                        elif any('fs' in desc.lower() for desc in descs):
                            self.logger.debug(f'Fat sat pre scan found, keeping | {self.Session_ID}')
                            to_remove = self.dicom_table[(~self.dicom_table['Series_desc'].str.lower().str.contains('fs')) & (self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['Lat'].str.lower() == lat.lower())]
                            self.removed['non_fat_sat'] = pd.concat([self.removed.get('non_fat_sat', pd.DataFrame()), to_remove], ignore_index=True)
                            self.dicom_table = self.dicom_table[(self.dicom_table['Series_desc'].str.lower().str.contains('fs') & (self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['Lat'].str.lower() == lat.lower()))|self.dicom_table['Post_scan'] ==0]
                    if (self.dicom_table.loc[(self.dicom_table['Lat'].str.lower() == lat.lower()) & (self.dicom_table['Pre_scan'] == 1)].shape[0] == 1):
                        self.logger.debug(f'Single pre scan found for side {lat} in Series_desc | {self.Session_ID}')
            else:
                self.logger.debug(f'Multiple pre scans found for each side, checking fat saturation | {self.Session_ID}')
                # prefer to keep fat saturation as the pre scan
                descs = self.dicom_table.loc[(self.dicom_table['Pre_scan'] == 1)&(self.dicom_table['Lat'].str.lower() == 'left'), 'Series_desc'].values
                if any('non fat sat' in desc.lower() for desc in descs):
                    self.logger.debug(f'Non-fat sat pre scan found, removing | {self.Session_ID}')
                    to_remove = self.dicom_table[(self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['Lat'].str.lower() == 'left') & (self.dicom_table['Series_desc'].str.lower().str.contains('non fat sat'))]
                    self.removed['Non_fat_sat'] = pd.concat([self.removed.get('Non_fat_sat', pd.DataFrame()), to_remove], ignore_index=True)
                    self.dicom_table = self.dicom_table[~((self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['Lat'].str.lower() == 'left') & (self.dicom_table['Series_desc'].str.lower().str.contains('non fat sat')))]
                
                descs = self.dicom_table.loc[(self.dicom_table['Pre_scan'] == 1)&(self.dicom_table['Lat'].str.lower() == 'right'), 'Series_desc'].values
                if any('non fat sat' in desc.lower() for desc in descs):
                    self.logger.debug(f'Non-fat sat pre scan found, removing | {self.Session_ID}')
                    to_remove = self.dicom_table[(self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['Lat'].str.lower() == 'right') & (self.dicom_table['Series_desc'].str.lower().str.contains('non fat sat'))]
                    self.removed['Non_fat_sat'] = pd.concat([self.removed.get('Non_fat_sat', pd.DataFrame()), to_remove], ignore_index=True)
                    self.dicom_table = self.dicom_table[~((self.dicom_table['Pre_scan'] == 1) & (self.dicom_table['Lat'].str.lower() == 'right') & (self.dicom_table['Series_desc'].str.lower().str.contains('non fat sat')))]

        # Check only 1 pre per side, if nto, remove side with multiple pre scans
        laterality   = self.dicom_table['Lat'].unique()
        for lat in laterality:
            if (self.dicom_table.loc[self.dicom_table['Lat'].str.lower() == lat.lower(), 'Pre_scan'].sum() > 1):
                self.logger.debug(f'Multiple pre scans found for side {lat}, removing side | {self.Session_ID}')
                to_remove = self.dicom_table[self.dicom_table['Lat'].str.lower() == lat.lower()]
                self.removed['Multiple_pre'] = pd.concat([self.removed.get('Multiple_pre', pd.DataFrame()), to_remove], ignore_index=True)
                # Remove all scans for this side
                self.dicom_table = self.dicom_table[self.dicom_table['Lat'].str.lower() != lat.lower()]

        #Isolating scans that are detected as pre or post scans
        self.removed['Not_pre_post'] = self.dicom_table[(self.dicom_table['Post_scan'] == 0) & (self.dicom_table['Pre_scan'] == 0)]
        self.dicom_table = self.dicom_table[(self.dicom_table['Post_scan'] == 1) | (self.dicom_table['Pre_scan'] == 1)]
        #Appending Session_ID with laterality if multiple sides detected
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

        # Removing samples with less than 3 scans per side
        IDs = self.dicom_table['SessionID'].unique()
        for ID in IDs:
            if len(self.dicom_table[self.dicom_table['SessionID'] == ID]) < 2:
                self.logger.debug(f'Not enough scans to isolate sequence for {ID}, removing session | {self.Session_ID}')
                to_remove = self.dicom_table[self.dicom_table['SessionID'] == ID]
                self.removed['Too_few_scans'] = pd.concat([self.removed.get('Too_few_scans', pd.DataFrame()), to_remove], ignore_index=True)
                self.dicom_table = self.dicom_table[self.dicom_table['SessionID'] != ID]
            elif len(self.dicom_table[self.dicom_table['SessionID'] == ID]) == 2:
                if len(self.dicom_table.loc[(self.dicom_table['SessionID'] == ID), 'NumSlices'].unique()) == 1:
                    self.logger.debug(f'Only 2 scans with the same number of slices for {ID}, removing session | {self.Session_ID}')
                    to_remove = self.dicom_table[self.dicom_table['SessionID'] == ID]
                    self.removed['Too_few_scans'] = pd.concat([self.removed.get('Too_few_scans', pd.DataFrame()), to_remove], ignore_index=True)
                    self.dicom_table = self.dicom_table[self.dicom_table['SessionID'] != ID]
                else:
                    self.logger.debug(f'Only 2 scans but with different number of slices for {ID}, preserving session | {self.Session_ID}')
    
        ### If only 1 pre and post sscan, likely that the post scan has been condensed to a single directory
        if (self.dicom_table['Post_scan'].sum() == 1):
            self.logger.debug(f'Detected singlepost scan entry for session, analyzing post for additional hidden scans | {self.Session_ID}')
            pre_slices = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1, 'NumSlices'].values[0]
            post_slices = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].values[0]
            if pre_slices == post_slices:
                self.logger.debug(f'Pre and post scans have the same number of slices, difficulty finding post sequence | {self.Session_ID}')
                self.removed['Single_post'] = self.dicom_table.copy()
                self.dicom_table = pd.DataFrame()
                return self.dicom_table  # Return empty DataFrame if pre and post scans have the same number of slices
            elif post_slices% pre_slices == 0:
                self.logger.debug(f'Post scan has multiple of pre scan slices, likely multiple volumes in a single directory | {self.Session_ID}')
                self.logger.debug(f'Expecting to find {post_slices // pre_slices} post scans in the sequence | {self.Session_ID}')
                ### TODO: Add logic for scanning the post scan directory for additional scans
            else:
                self.logger.debug(f'Unable to make sence of pre and post scans, removing session | {self.Session_ID}')
                self.removed['Pre_post_slices'] = self.dicom_table.copy()
                self.dicom_table = pd.DataFrame()
                return self.dicom_table  # Return empty DataFrame if pre and post scans are not consistent

        ### If 1 pre and multiple post, ensure all post have same number of slices as pre
        elif (self.dicom_table['Post_scan'].sum() > 1) and (self.dicom_table['Pre_scan'].sum() == 1):
            self.logger.debug(f'Detected single pre and multiple post scan entries for session, analyzing post for consistency | {self.Session_ID}')
            pre_slices = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1, 'NumSlices'].values[0]
            post_slices = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].values
            if all(post_slices == pre_slices):
                self.logger.debug(f'All post scans have the same numb er of slices as pre scan, preserving all post scans | {self.Session_ID}')
                return self.dicom_table


class DICOMsplit():
    def __init__(self, dicom_table: pd.DataFrame,  logger: logging.Logger = None, debug: int = 0, tmp_save: str='/FL_system/data/tmp/'):
        '''
        This class is used to split a single directory with multiple scans into multiple directories of a single scan each.
        The received DataFrame should contain a single Session_ID and directory
        '''
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
        ### Temporarily fixing directory from other systems
        if 'MSKCC_16-328' in self.Session_ID:
            self.logger.debug(f'Fixing directory path for MSKCC_16-328 | [{self.Session_ID}]')
            self.directory.replace('/FL_system/data/raw/', '/FL_system/data/raw/16-328/')
            for i in range(len(self.dicom_table)):
                self.dicom_table.at[i, 'PATH'] = self.dicom_table.at[i, 'PATH'].replace('/FL_system/data/raw/', '/FL_system/data/raw/16-328/')
        elif 'RIA_19-093' in self.Session_ID:
            self.logger.debug(f'Fixing directory path for RIA_19-093 | [{self.Session_ID}]')
            self.directory.replace('/mnt/XNAT/19-093/', '/FL_system/data/raw/19-093/')
            for i in range(len(self.dicom_table)):
                self.dicom_table.at[i, 'PATH'] = self.dicom_table.at[i, 'PATH'].replace('/mnt/XNAT/19-093/', '/FL_system/data/raw/19-093/')
        elif 'RIA_20-425' in self.Session_ID:
            self.logger.debug(f'Fixing directory path for RIA_20-425 | [{self.Session_ID}]')
            self.directory.replace('/FL_system/data/raw/', '/FL_system/data/raw/20-425/')
            for i in range(len(self.dicom_table)):
                self.dicom_table.at[i, 'PATH'] = self.dicom_table.at[i, 'PATH'].replace('/FL_system/data/raw/', '/FL_system/data/raw/20-425/')
        else:
            self.logger.error(f'Unable to determine protocol for path correction | [{self.Session_ID}]')

        # Determine expectations for the scan
        self.scan_path = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'PATH'].values[0]
        # Remove file from path to get directory
        self.scan_path = os.path.dirname(self.scan_path)

        self.expected_slices = self.dicom_table.loc[self.dicom_table['Pre_scan'] == 1, 'NumSlices'].unique()[0]

        # Determine if scanning is required
        if all(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'] == self.expected_slices):
            self.logger.debug(f'Pre and post scans have the same number of slices, no need to split | [{self.Session_ID}]')
            self.SCAN = False
        elif (len(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].unique()) == 1) and(self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].unique()[0] % self.expected_slices == 0):
            self.logger.debug(f'Post scans have different number of slices, scanning required | [{self.Session_ID}]')
            self.SCAN = True
            self.logger.debug(f'Set scan path to: {self.scan_path} | [{self.Session_ID}]')
            self.num_post_scans = self.dicom_table.loc[self.dicom_table['Post_scan'] == 1, 'NumSlices'].values[0] // self.expected_slices
        else:
            self.logger.warning(f'Unable to make sense of pre and post scans, removing session, further logic required | [{self.Session_ID}]')
            self.dicom_table = pd.DataFrame()
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
                if len(slices) != self.expected_slices:
                    self.logger.warning(f'Unexpected number of slices {len(slices)} found for trigger time {i}, expected {expected_slices} | [{self.Session_ID}]')
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

    def __init__(self, dicom_table: pd.DataFrame, logger: logging.Logger = None, debug: int = 0):
        self.debug = debug
        self.dicom_table = dicom_table
        self.logger = logger or logging.getLogger(__name__)
        try:
            self.Session_ID = self.dicom_table['SessionID'].unique()
        except KeyError:
            self.logger.error(f'No SessionID column present {self.dicom_table.columns}')
            return  # Exit the constructor if SessionID is missing
        if self.Session_ID.size > 1:
            #TODO: Implement multiple Session_IDs (non-parallel implementation)
            print('Multiple Session_IDs found in the table')
            print('Not currently implemented, please remake with a single Session_ID')
            return None
    
    def order(self, timing_param, secondary_param):
        """Orders the scans based on the provided timing parameter"""
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
                self.dicom_table = pd.DataFrame()
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
            self.dicom_table = pd.DataFrame()
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
            self.dicom_table = pd.DataFrame()
            return self.dicom_table

        # series_numbers = self.dicom_table['Series'][self.dicom_table['Major'] > 0].astype(int)
        # if len(series_numbers) == 0:
        #     if self.debug > 0:
        #         print(f'No series_numbers found for {self.dicom_table["SessionID"].unique()}')
        #     #clear dicom table
        #     self.dicom_table = pd.DataFrame()
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
        #     self.dicom_table = pd.DataFrame()