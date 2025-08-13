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
    
    def removeSlices(self):
        """Removes scans with a different number of slices"""
        self.N_SLICES = self.majorSlices()
        self.removed['Slices'] = self.dicom_table[self.dicom_table['NumSlices']%self.N_SLICES != 0]
        self.dicom_table = self.dicom_table[self.dicom_table['NumSlices']%self.N_SLICES == 0]
        #self.dicom_table['Remove_Slices'] = np.where(self.dicom_table['NumSlices']%self.N_SLICES == 0, 0, 1)
        dicom_copy = self.dicom_table.copy()
        for i in range(len(dicom_copy)):
            if dicom_copy['NumSlices'].iloc[i] != self.N_SLICES:
                self.logger.debug(f'Identified scan with multiple of {self.N_SLICES} slices | {self.Session_ID}')
                splitter = DICOMsplit(dicom_copy.iloc[[i]], logger=self.logger, debug=self.debug, tmp_save=self.tmp_save)
                self.temporary_relocations.append(splitter.temporary_relocations)
                # remove the original row entry
                self.dicom_table = self.dicom_table.drop(self.dicom_table[self.dicom_table['PATH'] == dicom_copy['PATH'].iloc[i]].index)
                # add the new rows
                self.dicom_table = pd.concat([self.dicom_table, splitter.output_table], ignore_index=True)
                self.logger.debug(f'Separated scan with {dicom_copy["NumSlices"].iloc[i]} slices into {len(splitter.output_table)} scans | {self.Session_ID}')
        self.logger.debug(f'Removed {len(self.removed["Slices"])} scans with a different number of slices | {self.Session_ID}')
        #self.update_valid('Remove_Slices')
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

    def isolate_sequence(self):
        """Isolates the sequence of scans based on SeriesNumber"""
        # TODO: should also choose between DISCO or steady state scans based on availability
        self.dicom_table = self.dicom_table.sort_values(by='Series')
        series = self.dicom_table['Series'].values # Assumption: Series is not repeated within a single session
        # TODO: Series number will be repeated for scans with  multiple volumes stacked into a single file, account for this here or by having the table fill in nly pre is stored seperatelyh, w2ith post sequence as 1 file (see 16-328 protocol)
        groups = np.concatenate(([0], np.cumsum(np.diff(series) != 1)))
        self.dicom_table['Series_group'] = groups

        n_groups = np.max(groups) + 1
        # Removiing groups that don't meet the minimum number of images
        for i in range(n_groups):
            n  = np.sum(groups == i)
            if n < 3:
                self.dicom_table = self.dicom_table.loc[~(self.dicom_table['Series_group'] == i)]

        # Now making sure that each sequence has a valid pre scan
        # ASSUMPTION: pre scan contains 'pre' in the Series_desc
        groups = self.dicom_table['Series_group'].unique()
        for i in groups:
            desc = self.dicom_table.loc[self.dicom_table['Series_group'] == i, 'Series_desc']
            pre_desc = [s for s in desc if 'pre' in s.lower()]
            if len(pre_desc) != 1:
                self.logger.error(f'Pre scan detection error for group {i} | {self.Session_ID}')
                # If no pre scan is found, remove the entire group
                if len(pre_desc) == 0:
                    self.logger.error(f'No pre scan found for group {i} | {self.Session_ID}')
                    self.dicom_table = self.dicom_table.loc[~(self.dicom_table['Series_group'] == i)]
                else:
                    # When multiple scans found, log error but keep data for review
                    self.logger.error(f'Multiple pre scans found for group {i} | {self.Session_ID}')
            n = self.dicom_table.loc[(self.dicom_table['Series_group'] == i)*(self.dicom_table['Series_desc'] == pre_desc[0]), 'Series']
            N = self.dicom_table.loc[self.dicom_table['Series_group'] == i, 'Series'].nunique()
            min_N = np.min(N)
            if n != min_N:
                self.logger.error(f'Pre scan is not the first scan in group {i} | {self.Session_ID}')
                # CHeck if more than 3 scans exist starting from the pre scan
                if np.sum( N >= min_N) >= 3:
                    self.logger.error(f'More than 3 scans exist starting from the pre scan in group {i} | {self.Session_ID}')
                    # REmoving scans from before the pre scan
                    self.dicom_table = self.dicom_table.loc[~((self.dicom_table['Series_group'] == i)*(self.dicom_table['Series'] < n))]
                else:
                    self.logger.error(f'Not enough scans exist starting from the pre scan in group {i} | {self.Session_ID}')
                    # Removing the entire group
                    self.dicom_table = self.dicom_table.loc[~(self.dicom_table['Series_group'] == i)]

        groups = self.dicom_table['Series_group'].unique()
        if len(groups) == 1:
            self.logger.debug(f'Single sequence found | {self.Session_ID}')
            return self.dicom_table
        
        # Still has multiple sequences, check for DISCO or steady state\
        # ASSUMPTION: DISCO scans contain 'disco' in the Series_desc
        # If DISCO and steady state scans are both found, prioritize steady state scans
        for i in groups:
            desc = self.dicom_table.loc[self.dicom_table['Series_group'] == i, 'Series_desc']
            if any('disco' in s.lower() for s in desc):
                self.logger.debug(f'DISCO sequence found in group {i} | {self.Session_ID}')
                self.dicom_table = self.dicom_table.loc[~(self.dicom_table['Series_group'] == i)]
            else:
                self.logger.error(f'No DISCO or steady state sequence found in group {i} | {self.Session_ID}')

        groups = self.dicom_table['Series_group'].unique()
        if len(groups) == 1:
            self.logger.debug(f'Single sequence found, returning simplified session data | {self.Session_ID}')
            return self.dicom_table
        else:
            self.logger.error(f'Multiple sequences found after filtering, additional logic required to solve | {self.Session_ID}')
            return self.dicom_table
        
class DICOMsplit():
    def __init__(self, dicom_table: pd.DataFrame,  logger: logging.Logger = None, debug: int = 0, tmp_save: str='/FL_system/data/tmp/'):
        '''
        This class is used to split a single directory with multiple scans into multiple directories of a single scan each.
        The received DataFrame should contain a single Session_ID and directory
        '''
        # Assert dicom table is a single row
        assert dicom_table.shape[0] == 1, 'Multiple rows found in the table'
        assert dicom_table['SessionID'].nunique() == 1, 'Multiple Session_IDs found in the table'
        # Check if the DataFrame is empty
        if dicom_table.empty:
            raise ValueError('The provided dicom_table is empty.')
        self.tmp_save = tmp_save
        if not os.path.exists(self.tmp_save):
            os.makedirs(self.tmp_save)
        self.debug = debug
        self.logger = logger or logging.getLogger(__name__)
        self.dicom_table = dicom_table
        self.Session_ID = self.dicom_table['SessionID'].iloc[0]
        assert isinstance(self.Session_ID, str), 'Session_ID must be a string'
        self.directory = os.path.dirname(self.dicom_table['PATH'].iloc[0])
        assert isinstance(self.directory, str), 'Directory must be a string'
        self.output_rows = []

        # Delay file relocation until all data parsed
        self.temporary_relocations = []

        self.scan_all()
        self.logger.info(f'Found {self.scan_results["Series"].nunique()} unique series | [{self.Session_ID}]')
        self.logger.info(f'Found {self.scan_results["TriTime"].nunique()} unique trigger times | [{self.Session_ID}]')
        self.sort_scans()
        self.output_table = pd.DataFrame(self.output_rows)

    def scan_all(self):
        """Scans all files in the directory"""
        files = glob.glob(os.path.join(self.directory, '*.dcm'))
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
        return 
    
    def sort_scans(self):
        """Sorts the scans based on the Series number"""
        #if not os.path.exists(f'{self.tmp_save}{self.Session_ID}'):
        #    os.makedirs(f'{self.tmp_save}{self.Session_ID}')
        #self.scan_results.to_csv(f'{self.tmp_save}{self.Session_ID}/scan_results.csv')

        if self.scan_results['Series'].nunique() == 1:
            self.logger.debug(f'Single series found | [{self.Session_ID}]')
            if self.scan_results['TriTime'].nunique() > 1:
                self.logger.debug(f' {self.scan_results["TriTime"].nunique()} unique trigger times found | [{self.Session_ID}]')
                timing = self.scan_results['TriTime'].unique()
                
                for i in range(len(timing)):
                    #if not os.path.exists(f'{self.tmp_save}{self.Session_ID}/{i}'):
                    #    os.makedirs(f'{self.tmp_save}{self.Session_ID}/{i}')
                    new_table = self.scan_results[self.scan_results['TriTime'] == timing[i]]
                    files = new_table['PATH'].to_list()
                    self.logger.debug(f'Found {len(files)} files for trigger time {timing[i]} | [{self.Session_ID}]')
                    for j in range(len(files)):
                        #shutil.copy(files[j], f"{self.tmp_save}{self.Session_ID}/{i}")
                        self.temporary_relocations.append((files[j], f"{self.tmp_save}{self.Session_ID}/{i}"))
                        if j == 0:
                            # grab first scan result for 
                            info = {}
                            info['PATH'] = f"{self.tmp_save}{self.Session_ID}/{i}"
                            extractor = DICOMextract(files[j])
                            info['AcqTime'] = (extractor.Acq())
                            info['SrsTime'] = (extractor.Srs())
                            info['ConTime'] = (extractor.Con())
                            info['StuTime'] = (extractor.Stu())
                            info['TriTime'] = (extractor.Tri())
                            info['InjTime'] = (extractor.Inj())
                            info['Series'] = extractor.Series()
                            info['ID'] = extractor.ID()
                            info['DATE'] = extractor.Date()
                            info['SessionID'] = self.Session_ID
                            self.output_rows.append(info)
                    # save the first file to the output table
                #print(self.scan_results)
            #print(self.scan_results)
        else:
            self.logger.info(f'Multiple series found | [{self.Session_ID}]')
            for i in self.scan_results['Series'].unique():
                #if not os.path.exists(f'{self.tmp_save}{self.Session_ID}/{i}'):
                #    os.makedirs(f'{self.tmp_save}{self.Session_ID}/{i}')
                slices = self.scan_results[self.scan_results['Series'] == i]
                
                for j in range(len(slices)):            
                    # copy each file to its series folder
                    file = slices['PATH'].iloc[j]
                    #shutil.copy(file, f"{self.tmp_save}{self.Session_ID}/{i}")
                    self.temporary_relocations.append((file, f"{self.tmp_save}{self.Session_ID}/{i}"))
                    if j == 0:
                        # save the first file to the output table
                        info = self.dicom_table.iloc[0].to_dict()
                        extractor = DICOMextract(file)
                        info['PATH'] = f"{self.tmp_save}{self.Session_ID}/{i}"
                        info['NumSlices'] = len(slices)
                        info['AcqTime'] = self.scan_results['AcqTime'].iloc[j]
                        info['SrsTime'] = self.scan_results['SrsTime'].iloc[j]
                        info['ConTime'] = self.scan_results['ConTime'].iloc[j]
                        info['StuTime'] = self.scan_results['StuTime'].iloc[j]
                        info['TriTime'] = self.scan_results['TriTime'].iloc[j]
                        info['InjTime'] = self.scan_results['InjTime'].iloc[j]
                        info['Series'] = i
                        self.output_rows.append(info)
        return


class DICOMorder():

    def __init__(self, dicom_table: pd.DataFrame, logger: logging.Logger = None, debug: int = 0):
        self.debug = debug
        self.dicom_table = dicom_table
        self.logger = logger or logging.getLogger(__name__)
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
        if len(pre_indx) == 1:
            indx = np.append(indx, pre_indx)
            self.dicom_table = self.dicom_table.loc[indx]
            return self.dicom_table
        elif len(pre_indx) == 0:
            print(f'No pre scan found for {self.dicom_table["SessionID"].unique()}')
            print(f'Attempting alternate pre scan detection')
            pre_indx = self.alternate_pre()
        
        if len(pre_indx) == 1:
            indx = np.append(indx, pre_indx)
            self.dicom_table = self.dicom_table.loc[indx]
            return self.dicom_table
        else:
            print('Alternative pre scan detection failed')
            print('No pre scan found')
            print('Returning empty dicom table')
            self.dicom_table = pd.DataFrame()