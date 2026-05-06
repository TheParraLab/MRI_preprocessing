import os
import json
from datetime import datetime, timezone

start_time = datetime.now(timezone.utc) # Record the start time of the comparison in UTC timezone

scans = os.listdir(os.path.join(os.getcwd(), 'scan_results'))
for i in range(len(scans)):
    print(f'{i}: {scans[i]}')
scan1_index = int(input('Select the primary scan to compare: '))
scan2_index = int(input('Select the secondary scan to compare: '))

scan1_path = os.path.join(os.getcwd(), 'scan_results', scans[scan1_index])
scan2_path = os.path.join(os.getcwd(), 'scan_results', scans[scan2_index])
with open(scan1_path, 'r') as f:
    scan1_data = json.load(f)
with open(scan2_path, 'r') as f:
    scan2_data = json.load(f)

# Compare the two scans and identify differences in file presence and checksums
# When secondary has a directory that primary does not, report it as "Missing in Primary"
# When primary has a directory that secondary does not, report it as "Missing in Secondary"
# When both have the same directory but different files or checksums, report the differences as "Incomplete Matches"
# When both have the same directory and same files with same checksums, report it as "Complete Matches"
report = {
    'missing_in_primary': [],
    'missing_in_secondary': [],
    'incomplete_matches': [],
    'complete_matches': [],
    'imaging_matches': [],
    'metadata_matches': [],
}
for i in scan1_data['results']:
    if i not in scan2_data['results']:
        report['missing_in_secondary'].append(i)
    else:
        # Focus on *.nii and *.json files seperately
        
        json_primary_files = {f['file_name']: f['md5'] for f in scan1_data['results'][i]['files'] if f['file_name'].endswith('.json')}
        json_secondary_files = {f['file_name']: f['md5'] for f in scan2_data['results'][i]['files'] if f['file_name'].endswith('.json')}

        nii_primary_files = {f['file_name']: f['md5'] for f in scan1_data['results'][i]['files'] if f['file_name'].endswith('.nii')}
        nii_secondary_files = {f['file_name']: f['md5'] for f in scan2_data['results'][i]['files'] if f['file_name'].endswith('.nii')}

        if json_primary_files == json_secondary_files and nii_primary_files == nii_secondary_files:
            report['complete_matches'].append(i)
        elif json_primary_files == json_secondary_files and nii_primary_files != nii_secondary_files:
            report['imaging_matches'].append(i)
        elif json_primary_files != json_secondary_files and nii_primary_files == nii_secondary_files:
            report['metadata_matches'].append(i)
        else:
            report['incomplete_matches'].append(i)

for i in scan2_data['results']:
    if i not in scan1_data['results']:
        report['missing_in_primary'].append(i)

stop_time = datetime.now(timezone.utc) # Record the stop time of the comparison in UTC timezone
header = {
    # Take both scan headers
    'primary': {scan1_data['header']},
    'secondary': {scan2_data['header']},
    'analysis': {
        'start_time': start_time,
        'stop_time': stop_time
    }
}
output = {
    'header': header,
    'report': report
}
output_file = f'comparison_report_{scan1_index}_vs_{scan2_index}.json'
output_path = os.path.join(os.getcwd(), 'comparison_findings', output_file)