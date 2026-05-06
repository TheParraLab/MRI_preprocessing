##########################################################################################################
# scan_dest.py
#
# This script scans a specified directory for subdirectories (sessions) and files, computes the MD5 hash
# of each file, and saves the results in a JSON file. The JSON file includes metadata such as the scan directory, start time, and stop time.
# Usage:
# 1. Run the script and input the directory to scan when prompted.
# 2. The script will process the files and save the results in a JSON file named 'scan_results_0.json' (or 'scan_results_N.json' if the file already exists).
# Note: Ensure you have the necessary permissions to read the files in the specified directory.
########################################################################################################

# Import necessary libraries
import json
import os
from datetime import datetime, timezone
from hashlib import md5

# Function to compute the MD5 hash of a file
def file_md5(file_path):
    # Compute the MD5 hash of the file at the given path
    hash_md5 = md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): # Read the file in chunks to handle large files efficiently, the lambda function reads 4096 bytes at a time until the end of the file is reached (indicated by an empty byte string).
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

start_time = datetime.now(timezone.utc) # Record the start time of the scan in UTC timezone
scan_dir = input('Please enter the directory to scan: ')
print(f'Scanning directory: {scan_dir}')

results = {}

for root, dirs, files in os.walk(scan_dir):
    # Skip the root directory itself, we only want to process subdirectories (sessions)
    if root == scan_dir:
        continue

    session_id = os.path.basename(root)
    session_files = []

    for file in sorted(files):
        file_path = os.path.join(root, file)
        print(f'Processing file: {file_path}')
        session_files.append({
            'file_name': file,
            'md5': file_md5(file_path),
        })

    if session_files:
        results[session_id] = {
            'files': session_files,
        }

stop_time = datetime.now(timezone.utc)# Record the stop time of the scan in UTC timezone

# Prepare the output dictionary with metadata and results
output = {
    'header': {
        'scan_dir': scan_dir,
        'start_time': start_time.isoformat(),
        'stop_time': stop_time.isoformat(),
    },
    'results': results
}

# Save the output to a JSON file, ensuring we don't overwrite existing files by incrementing the filename if necessary
output_file = 'scan_results_0.json'
output_path = os.path.join(os.getcwd(), 'scan_results', output_file)
if os.path.exists(output_file):
    N = output_file.split('_')[-1].split('.')[0]
    output_file = f'scan_results_{int(N) + 1}.json'
    print(f'Output file already exists. Saving to: {output_file}')

# Write the output dictionary to a JSON file with indentation for readability
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
print(f'Saved JSON results to: {output_file}')
# End of script