import os
import numpy as np
import nibabel as nib

path = input('Please enter the path to the data directory: ').strip()

dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
print(f'Found {len(dirs)} subject directories.')
dirs.append('All subjects')

for i in range(len(dirs)):
    print(f'[{i}] {dirs[i]}')
selection = input(f'Select subject directory [0-{len(dirs) - 1}] (default {len(dirs)-1}): ').strip()

if selection == '':
    selected_dir = dirs[-1]
else:
    try:
        index = int(selection)
        if 0 <= index < len(dirs):
            selected_dir = dirs[index]
        else:
            print('Invalid selection. Defaulting to all subjects.')
            selected_dir = dirs[-1]
    except ValueError:
        print('Invalid input. Defaulting to all subjects.')
        selected_dir = dirs[-1]
print(f'Selected directory: {selected_dir}')
if selected_dir != 'All subjects':
    # Process the selected directory
    print(f'Processing directory: {selected_dir}')
    try:
        pre = nib.load(os.path.join(selected_dir, '00_RAS.nii'))
    except FileNotFoundError:
        pre = nib.load(os.path.join(selected_dir, '00_RAS.nii.gz'))
    p95 = np.nanpercentile(pre.get_fdata(), 95)
    print(f'95th percentile of pre: {p95}')
    fils = [f for f in os.listdir(selected_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    for fil in fils:
        img = nib.load(os.path.join(selected_dir, fil))
        data = img.get_fdata()
        data = data / p95
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, os.path.join(selected_dir, f'NORM_{fil}'))
else:
    # Process all directories
    print('Processing all subject directories.')
    dirs = [d for d in dirs if d != 'All subjects']
    for selected_dir in dirs:
        print(f'Processing directory: {selected_dir}')
        try:
            pre = nib.load(os.path.join(selected_dir, '00_RAS.nii'))
        except FileNotFoundError:
            pre = nib.load(os.path.join(selected_dir, '00_RAS.nii.gz'))
        p95 = np.nanpercentile(pre.get_fdata(), 95)
        print(f'95th percentile of pre: {p95}')
        fils = [f for f in os.listdir(selected_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        for fil in fils:
            img = nib.load(os.path.join(selected_dir, fil))
            data = img.get_fdata()
            data = data / p95
            new_img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(new_img, os.path.join(selected_dir, f'NORM_{fil}'))

print('Processing complete.')