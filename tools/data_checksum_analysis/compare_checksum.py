import os
import json
from datetime import datetime, timezone

start_time = datetime.now(timezone.utc) # Record the start time of the comparison in UTC timezone

print('Available scans for comparison:')
print('Primary selection will be the source scan, and secondary should be the destination scan to compare against.')
scans = os.listdir(os.path.join(os.getcwd(), 'scan_results'))
for i in range(len(scans)):
    print(f'{i}: {scans[i]}')
scan1_index = int(input('Select the primary scan to compare: '))
scan2_index = int(input('Select the secondary scan to compare: '))

scan1_path = os.path.join(os.getcwd(), 'scan_results', scans[scan1_index])
scan2_path = os.path.join(os.getcwd(), 'scan_results', scans[scan2_index])
with open(scan1_path, 'r') as f:
    scan1_data = json.load(f)
    print(f'Loaded primary scan: {scans[scan1_index]} with {len(scan1_data["results"])} directories')
with open(scan2_path, 'r') as f:
    scan2_data = json.load(f)
    print(f'Loaded secondary scan: {scans[scan2_index]} with {len(scan2_data["results"])} directories')

# Compare files at the individual level across both scans
# Files in primary that also exist in secondary with matching checksums -> marked for deletion from primary
# Files in primary that are missing in secondary or have different checksums -> marked for transfer/replacement
report = {
    'ready_for_deletion': [],
    'need_transfer': [],
}
secondary_file_index = {}
for dir_name, dir_data in scan2_data['results'].items():
    for f in dir_data['files']:
        key = os.path.join(dir_name, f['file_name'])
        secondary_file_index[key] = f['md5']

for dir_name, dir_data in scan1_data['results'].items():
    for f in dir_data['files']:
        key = os.path.join(dir_name, f['file_name'])
        secondary_md5 = secondary_file_index.get(key)
        if secondary_md5 is not None and secondary_md5 == f['md5']:
            report['ready_for_deletion'].append({
                'path': key,
                'md5': f['md5'],
            })
        else:
            report['need_transfer'].append({
                'path': key,
                'primary_md5': f['md5'],
                'secondary_md5': secondary_md5 if secondary_md5 else None,
            })

stop_time = datetime.now(timezone.utc) # Record the stop time of the comparison in UTC timezone
header = {
    # Take both scan headers
    'primary': scan1_data['header'],
    'secondary': scan2_data['header'],
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
with open(output_path, 'w') as f:
    json.dump(output, f, indent=4, default=str)
print(f'Comparison report saved to: {output_path}')
print('-='*20)
print('SUMMARY')
print('-='*20)
print(f'Need Transfer: {len(output['report']['need_transfer'])}')
print(f'Deletion Ready: {len(output['report']['ready_for_deletion'])}')
