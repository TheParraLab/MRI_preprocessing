import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib  # type: ignore[import-not-found]
import numpy as np


def find_subject_dirs(coreg_dir):
	directory = Path(coreg_dir).expanduser().resolve()
	if not directory.exists():
		raise FileNotFoundError(f'Coreg directory does not exist: {directory}')

	directories = sorted([path for path in directory.iterdir() if path.is_dir()])
	if not directories:
		raise FileNotFoundError(f'No subject directories found in {directory}')
	return directories


def find_nifti_files(subject_dir):
	directory = Path(subject_dir).expanduser().resolve()
	if not directory.exists():
		raise FileNotFoundError(f'Subject directory does not exist: {directory}')

	files = sorted(
		[path for path in directory.iterdir() if path.is_file() and (path.name.endswith('.nii') or path.name.endswith('.nii.gz'))]
	)
	if not files:
		raise FileNotFoundError(f'No NIfTI files found in {directory}')
	return files


def find_reference_file(subject_dir):
	directory = Path(subject_dir).expanduser().resolve()
	for candidate in ('01_RAS.nii.gz', '01_RAS.nii'):
		path = directory / candidate
		if path.exists():
			return path
	raise FileNotFoundError(f'Could not find 01_RAS.nii.gz or 01_RAS.nii in {directory}')


def moving_file_id(path):
	name = Path(path).name
	if name.endswith('.nii.gz'):
		name = name[:-7]
	elif name.endswith('.nii'):
		name = name[:-4]

	if '_' in name:
		return name.split('_', 1)[0]
	return name


def prompt_for_subject_dir(subject_dirs):
	print('\nAvailable subject directories:')
	for index, path in enumerate(subject_dirs):
		print(f'  [{index}] {path.name}')

	while True:
		selection = input(f'Select subject directory [0-{len(subject_dirs) - 1}] (default 0): ').strip()
		if selection == '':
			return subject_dirs[0]
		try:
			index = int(selection)
		except ValueError:
			print('Please enter a valid number.')
			continue

		if 0 <= index < len(subject_dirs):
			return subject_dirs[index]

		print(f'Please choose a number between 0 and {len(subject_dirs) - 1}.')


def load_volume(path):
	image = nib.load(str(path))
	data = np.asanyarray(image.dataobj)
	data = np.squeeze(data)
	if data.ndim == 4:
		data = data[..., 0]
	if data.ndim != 3:
		raise ValueError(f'Expected a 3D volume after squeezing, but got shape {data.shape} for {path.name}')
	return data


def scale_to_uint8(slice_data):
	finite_values = slice_data[np.isfinite(slice_data)]
	if finite_values.size == 0:
		return np.zeros_like(slice_data, dtype=np.uint8)

	lower, upper = np.percentile(finite_values, [1, 99])
	if lower == upper:
		return np.zeros_like(slice_data, dtype=np.uint8)

	normalized = np.clip((slice_data - lower) / (upper - lower), 0, 1)
	return (normalized * 255).astype(np.uint8)


def create_checkerboard(reference_slice, moving_slice, tiles=8):
	reference = scale_to_uint8(reference_slice)
	moving = scale_to_uint8(moving_slice)
	height, width = reference.shape
	row_tiles = max(2, min(tiles, height))
	col_tiles = max(2, min(tiles, width))
	row_edges = np.linspace(0, height, row_tiles + 1, dtype=int)
	col_edges = np.linspace(0, width, col_tiles + 1, dtype=int)

	checkerboard = np.zeros_like(reference, dtype=np.uint8)
	for row_index in range(row_tiles):
		row_start, row_end = row_edges[row_index], row_edges[row_index + 1]
		for col_index in range(col_tiles):
			col_start, col_end = col_edges[col_index], col_edges[col_index + 1]
			if (row_index + col_index) % 2 == 0:
				checkerboard[row_start:row_end, col_start:col_end] = reference[row_start:row_end, col_start:col_end]
			else:
				checkerboard[row_start:row_end, col_start:col_end] = moving[row_start:row_end, col_start:col_end]

	return checkerboard


def get_slice_indices(shape, slice_count=3, border_margin=5):
	safe_start = max(0, border_margin)
	safe_end = min(shape - 1, shape - 1 - border_margin)
	if safe_start >= safe_end:
		return [shape // 2]

	if slice_count <= 1:
		return [shape // 2]

	candidates = np.linspace(0.2, 0.8, slice_count)
	indices = [int(round(candidate * (safe_end - safe_start) + safe_start)) for candidate in candidates]
	indices.append((safe_start + safe_end) // 2)
	return sorted(set(max(safe_start, min(safe_end, index)) for index in indices))


def extract_slice(volume, orientation, slice_index=None):
	if orientation == 'axial':
		axis = 2
		if slice_index is None or slice_index >= volume.shape[axis]:
			slice_index = volume.shape[axis] // 2
		slice_data = volume[:, :, slice_index]
	elif orientation == 'coronal':
		axis = 1
		if slice_index is None or slice_index >= volume.shape[axis]:
			slice_index = volume.shape[axis] // 2
		slice_data = volume[:, slice_index, :]
	elif orientation == 'sagittal':
		axis = 0
		if slice_index is None or slice_index >= volume.shape[axis]:
			slice_index = volume.shape[axis] // 2
		slice_data = volume[slice_index, :, :]
	else:
		raise ValueError(f'Unknown orientation: {orientation}')

	return np.rot90(np.squeeze(slice_data)), slice_index


def render_overlay_figure(reference_volume, moving_volume, title, slice_count=3):
	orientations = ['axial', 'coronal', 'sagittal']
	fig, axes = plt.subplots(len(orientations) * 2, slice_count, figsize=(6 * slice_count, 4.5 * len(orientations) * 2), squeeze=False)
	fig.suptitle(title, fontsize=16)

	for row_index, orientation in enumerate(orientations):
		reference_shape = reference_volume.shape[2] if orientation == 'axial' else reference_volume.shape[1] if orientation == 'coronal' else reference_volume.shape[0]
		moving_shape = moving_volume.shape[2] if orientation == 'axial' else moving_volume.shape[1] if orientation == 'coronal' else moving_volume.shape[0]
		base_shape = min(reference_shape, moving_shape)
		slice_indices = get_slice_indices(base_shape, slice_count=slice_count)
		for col_index in range(slice_count):
			overlay_axis = axes[row_index * 2, col_index]
			checkerboard_axis = axes[row_index * 2 + 1, col_index]
			if col_index >= len(slice_indices):
				overlay_axis.axis('off')
				checkerboard_axis.axis('off')
				continue

			slice_index = slice_indices[col_index]
			reference_slice, _ = extract_slice(reference_volume, orientation, slice_index)
			moving_slice, _ = extract_slice(moving_volume, orientation, slice_index)

			overlay_axis.imshow(scale_to_uint8(reference_slice), cmap='gray', origin='lower', interpolation='nearest')
			overlay_axis.imshow(
				scale_to_uint8(moving_slice),
				cmap='Reds',
				origin='lower',
				alpha=0.35,
				interpolation='nearest',
			)
			overlay_axis.set_title(f'{orientation.title()} slice {slice_index} overlay')
			overlay_axis.axis('off')

			checkerboard_axis.imshow(create_checkerboard(reference_slice, moving_slice), cmap='gray', origin='lower', interpolation='nearest')
			checkerboard_axis.set_title(f'{orientation.title()} slice {slice_index} checkerboard')
			checkerboard_axis.axis('off')

	for row_index, orientation in enumerate(orientations):
		fig.text(0.01, 1 - ((row_index * 2 + 0.5) / (len(orientations) * 2)), f'{orientation.title()} overlay', rotation=90, va='center', ha='left', fontsize=11)
		fig.text(0.01, 1 - ((row_index * 2 + 1.5) / (len(orientations) * 2)), f'{orientation.title()} checkerboard', rotation=90, va='center', ha='left', fontsize=11)

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	return fig


def build_coreg_overlays(subject_dir, slice_count=3):
	directory = Path(subject_dir).expanduser().resolve()
	files = find_nifti_files(directory)
	reference_path = find_reference_file(directory)
	reference_volume = load_volume(reference_path)

	created_paths = []
	for path in files:
		moving_volume = load_volume(path)
		title = f'{directory.name}: {path.name} vs {reference_path.name}'
		fig = render_overlay_figure(reference_volume, moving_volume, title=title, slice_count=slice_count)
		output_path = directory / f'{moving_file_id(path)}_TEST.png'
		fig.savefig(output_path, dpi=200, bbox_inches='tight')
		plt.close(fig)
		created_paths.append(output_path)
		print(f'Saved figure to {output_path}')

	return created_paths


def process_subject_dirs(subject_dirs, slice_count=3):
	all_created_paths = []
	for subject_dir in subject_dirs:
		try:
			all_created_paths.extend(build_coreg_overlays(subject_dir, slice_count=slice_count))
		except FileNotFoundError as error:
			print(f'Skipping {Path(subject_dir).name}: {error}')

	return all_created_paths


def parse_args():
	parser = argparse.ArgumentParser(description='Overlay each NIfTI file against 01_RAS.nii.gz in multiple orientations and slices')
	parser.add_argument('--coreg_dir', type=str, default='/FL_system/data/coreg/', help='Directory containing subject directories with coregistered NIfTI files')
	parser.add_argument('--slice-count', type=int, default=3, help='Number of slices per orientation to display')
	parser.add_argument('--auto', action='store_true', help='Process every subject directory in the coreg directory without prompting')
	return parser.parse_args()


def main():
	args = parse_args()
	subject_dirs = find_subject_dirs(args.coreg_dir)
	if args.auto:
		process_subject_dirs(subject_dirs, slice_count=args.slice_count)
	else:
		subject_dir = prompt_for_subject_dir(subject_dirs)
		build_coreg_overlays(subject_dir, slice_count=args.slice_count)


if __name__ == '__main__':
	main()


