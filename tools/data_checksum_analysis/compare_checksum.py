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

# Compare at the session level: any file mismatch flags the entire session for transfer.
# Only sessions where every file matches go to ready_for_deletion.
# Sessions only in secondary are flagged as missing from primary.
report = {
    'ready_for_deletion': [],
    'need_transfer': [],
    'missing_from_primary': [],
}
secondary_file_index = {}
secondary_session_set = set()
for dir_name, dir_data in scan2_data['results'].items():
    secondary_session_set.add(dir_name)
    for f in dir_data['files']:
        key = os.path.join(dir_name, f['file_name'])
        secondary_file_index[key] = f['md5']

primary_session_set = set()
for dir_name, dir_data in scan1_data['results'].items():
    primary_session_set.add(dir_name)
    session_needs_transfer = False

    for f in dir_data['files']:
        key = os.path.join(dir_name, f['file_name'])
        secondary_md5 = secondary_file_index.get(key)
        if secondary_md5 is None or secondary_md5 != f['md5']:
            session_needs_transfer = True
            break

    if session_needs_transfer:
        report['need_transfer'].append({
            'session': dir_name,
            'file_count': len(dir_data['files']),
        })
    else:
        for f in dir_data['files']:
            report['ready_for_deletion'].append({
                'path': os.path.join(dir_name, f['file_name']),
                'md5': f['md5'],
            })

for dir_name in (secondary_session_set - primary_session_set):
    dir_data = scan2_data['results'][dir_name]
    report['missing_from_primary'].append({
        'session': dir_name,
        'file_count': len(dir_data['files']),
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
print(f'Need Transfer: {len(output["report"]["need_transfer"])}')
print(f'Deletion Ready: {len(output["report"]["ready_for_deletion"])}')
print(f'Missing from Primary: {len(output["report"]["missing_from_primary"])}')
