#!/usr/bin/env python3

import os
import shutil
import argparse
import subprocess
import re
import pickle

import pandas as pd

# input arguments
parser = argparse.ArgumentParser(description='Prepare for running jobs on HPC')
parser.add_argument('--scan_dir', type=str, required=False, help='Directory containing scans to process')
parser.add_argument('--out_dir', type=str, required=True, help='Directory to save the output')
parser.add_argument('--script', type=str, required=True, help='Script path to run on each scan')
parser.add_argument('--env', type=str, required=True, help='Pyenv path to use for the script')
parser.add_argument('--partition', type=str, default="", help='Partition to use for the job')
parser.add_argument('--load_table', type=str, default="", help='Load table to use for the job')
parser.add_argument('--test', nargs='?', type=int, const=10, help='Number of test directories to process')
parser.add_argument('--prune', '-p', action='store_true', help='Enable the deletion of the source data once processed')

# Get the arguments
args = parser.parse_args()
scan_dir = args.scan_dir
out_dir = args.out_dir
script = args.script
prune = args.prune
# Extract the path to the script
#script = os.path.abspath(script)
working_dir = (os.sep).join(script.split(os.sep)[:-1])
print(f"Working directory: {working_dir}")
env = args.env
if env.split('/')[-1] != 'activate':
    env = os.path.join(env, 'bin', 'activate')
partition = args.partition
def limit_array_size(items):
    # Run the scontrol command and get the output
    output = subprocess.check_output(["scontrol", "show", "config"], text=True)

    # Find the MaxArraySize value in the output
    max_array_size = None
    for line in output.splitlines():
        if "MaxArraySize" in line:
            max_array_size = int(line.split("=")[1].strip())
            break
            
    print(f"MaxArraySize: {max_array_size}")
    print('Limiting array size to received limit')
    # Limit the size of the array to the maximum value
    if len(items) > max_array_size:
        items_out = []
        groupings = len(items) // max_array_size
        # Creating max_array_size sublists of items
        for i in range(0,max_array_size):
            if i < max_array_size - 1:
                items_out.append(items[i*groupings:(i+1)*groupings])
            else:
                items_out.append(items[i*groupings:])
    else:
        items_out = items
    print(f"Number of items: {len(items_out)}")
    return items_out


def get_cluster_resources(partition=""):
    cmd = ["sinfo", "--Format=partition,cpusstate,memory", "--noheader"]
    if partition:
        cmd += ["--partition", partition]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, text=True)
        lines = result.stdout.strip().split("\n")
        total_cpus = 0
        max_mem = 0  # in MB

        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            _, cpus_state, mem_mb = parts
            # cpus_state example: "2/28/0" → allocated/idle/other
            try:
                idle_cpus = int(cpus_state.split("/")[1])
                mem_mb = int(mem_mb)
                total_cpus += idle_cpus
                max_mem = max(max_mem, mem_mb)
            except Exception:
                continue

        return total_cpus, max_mem
    except subprocess.CalledProcessError:
        return None, None

def write_dir_list(items: list, filename: str):
    #with open(filename, 'w') as f:
    #    for item in items:
    #        f.write(f"{item}\n")
    with open(filename, 'wb') as f:
        pickle.dump(items, f)
    print(f"Directory list written to {filename}")

def create_job_script(scan_dir, script, dir_list, extra_flags="", sbatch_args=""):
    job_name = os.path.basename(script).split('.')[0]
    if '05_alignScans.py' in script:
        # Create a directory list file
        niftyreg_inclusion = 'export PATH="$HOME/niftyreg/bin:$PATH"'
    else:
        niftyreg_inclusion = ''

    job_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={working_dir}/logs/{job_name}_%A_%a.out
#SBATCH --error={working_dir}/logs/{job_name}_%A_%a.err
#SBATCH --cpus-per-task={cpus_per_job}
#SBATCH --mem={mem_per_job}G
#SBATCH --array=0-{N-1}

source {env}
{niftyreg_inclusion}

python3 {script} --dir_idx ${{SLURM_ARRAY_TASK_ID}} --dir_list {dir_list} --save_dir {out_dir} {extra_flags}
"""
    print(job_script)
    return job_script

if __name__ == "__main__":
    # Check if the scan directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")
    
    # Perform script-specific processing and checks
    extra_flags = ""
    sbatch_args = ""
    match script.split(os.sep)[-1]:
        case '01_scanDicom.py':
            print('Input script: 01_scanDicom.py')
            if not os.path.exists(scan_dir):
                raise FileNotFoundError(f"Scan directory {scan_dir} does not exist.")
            if not os.path.exists(os.path.join(out_dir, 'tmp/')):
                os.makedirs(os.path.join(out_dir, 'tmp/'))
                print(f"Created tmp directory: {os.path.join(out_dir, 'tmp/')}")
            # Compile a list of directories to process
            print('-' * 20)
            subdirs = os.listdir(scan_dir)
            subdirs = [os.path.join(scan_dir, d) for d in subdirs]
            N = len(subdirs)
            if len(subdirs) <= 10:
                print('WARNING | Less than 10 directories found. Check the scan directory.')
            else:
                print(f"Found {N} directories in {scan_dir}")
            write_dir_list(subdirs, os.path.join(working_dir, 'list.txt'))
            print('-' * 20)

        case '02_parseDicom.py':
            print('Input script: 02_parseDicom.py')
            # Compile a list of IDs to process
            print('-' * 20)
            Input_table = pd.read_csv(args.load_table, sep=',')
            IDs = Input_table['ID'].unique()
            N = len(IDs)
            print(f'Found {N} unique IDs in {args.load_table}')
            write_dir_list(IDs, os.path.join(working_dir, 'list.txt'))
            print('-' * 20)
            extra_flags = f'--load_table {args.load_table}'


        case '04_saveRAS.py':
            print('Input script: 04_saveRAS.py')
            # Compile a list of IDs to process
            if not os.path.exists(scan_dir):
                raise FileNotFoundError(f"Scan directory {scan_dir} does not exist.")
            if not os.path.exists(os.path.join(out_dir, 'tmp/')):
                os.makedirs(os.path.join(out_dir, 'tmp/'))
                print(f"Created tmp directory: {os.path.join(out_dir, 'tmp/')}")
            # Compile a list of directories to process
            print('-' * 20)
            subdirs = os.listdir(scan_dir)
            subdirs = [os.path.join(scan_dir, d) for d in subdirs]
            print(f"Found {len(subdirs)} directories in {scan_dir}")
            # Limit the size of the array to the maximum value
            subdirs = limit_array_size(subdirs)
            N = len(subdirs)
            print(f'Proceeding with {N} number of jobs')
            write_dir_list(subdirs, os.path.join(working_dir, 'list.txt'))
            print('-' * 20)

        case '05_alignScans.py':
            print('Input script: 05_alignScans.py')
            # Compile a list of directories to process
            print('-' * 20)
            if not os.path.exists(scan_dir):
                raise FileNotFoundError(f"Scan directory {scan_dir} does not exist.")

            subdirs = os.listdir(scan_dir)
            subdirs = [os.path.join(scan_dir, d) for d in subdirs]
            subdirs = limit_array_size(subdirs)
            N = len(subdirs)
            if len(subdirs) <= 10:
                print('WARNING | Less than 10 directories found. Check the scan directory.')
            else:
                print(f"Found {N} directories in {scan_dir}")
            write_dir_list(subdirs, os.path.join(working_dir, 'list.txt'))
            if prune:
                print('Pruning enabled. Deleting source data after processing.')
                extra_flags = '--prune'
            print('-' * 20)
    
    # Get cluster-wide idle CPUs and max mem per node
    print('-' * 20)
    idle_cpus, max_mem_mb = get_cluster_resources(partition)
    if idle_cpus == 0 or max_mem_mb == 0:
        raise RuntimeError("No idle CPUs or memory available in the cluster.")

    print(f"Idle CPUs: {idle_cpus}, Max Memory per Node: {max_mem_mb} MB")
    mem_per_job = min(16000, max_mem_mb) // 1024  # Convert to GB
    cpus_per_job = 1 if idle_cpus >= N else max(1, idle_cpus // len(subdirs))
    print(f"CPUs per job: {cpus_per_job}, Memory per job: {mem_per_job} GB")
    print(f"Total jobs: {N}")
    print('-' * 20)

    # Create the job script
    print('-' * 20)
    # Submit the job script
    job_script_path = os.path.join(working_dir, 'send_job.slurm')
    with open(job_script_path, 'w') as f:
        f.write(create_job_script(scan_dir, script, os.path.join(working_dir, 'list.txt'), extra_flags))
    print(f"Job script written to {job_script_path}")
    print('-' * 20)

    # Submit the job script
    print("Submitting job script...")
    submit_cmd = ["sbatch", job_script_path]
    if partition:
        submit_cmd += ["--partition", partition]
    try:
        result = subprocess.run(submit_cmd, stdout=subprocess.PIPE, check=True, text=True)
        print('Job submitted successfully.')
    except:
        print('Error submitting job. Check the job script for errors.')
        raise