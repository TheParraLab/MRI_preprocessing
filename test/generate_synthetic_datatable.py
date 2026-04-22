import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

NUM_SESSIONS = 20

# Known series descriptions common in real data
COMMON_SERIES = [
    'T1 Sagittal post', 'Loc', 'T1 Sagittal pre', 'T1 non fat sat', 'Axial T1',
    'LOC', 'T2 left breast', 'T2 right breast', 'PJN', 'T2 left', 'T2 right',
    'T1 Axial AP', 'WATER: AX, T2 FS', 'Axial DWI', 'Localization',
    'Axial T1 FS post', 'Axial T1 FS pre', 'Sagittal T2 FS',
    'Axial T2 FS', 'MIP T1', 'T2 Axial FS', 'Axial T1 post', 'T2 FS left',
    'T2 FS right', 'Axial T1 pre', 'STIR', 'T2 FS AXIAL', 'T1 post', 'T1 pre'
]

TYPE_VALUES = [
    "['ORIGINAL', 'PRIMARY', 'OTHER']",
    "['DERIVED', 'PRIMARY', 'DIFFUSION', 'ADC']",
    "['DERIVED', 'PRIMARY', 'DIXON', 'WATER']",
    "['DERIVED', 'PRIMARY', 'OTHER', 'SUBTRACT']",
    "['ORIGINAL', 'PRIMARY', 'PRIMARY', 'NONE']",
    "Unknown"
]

NUM_SLICES_OPTIONS = [240, 156, 30, 40, 34, 46, 44, 176, 160, 144, 166]
THICKNESS_OPTIONS = [3.0, 1.1, 1.5, 1.4, 1.2, 1.0]

# Modality weights: ~76% T1, ~24% T2, ~0.003% Unknown
MODALITY_CHOICES = ['T1', 'T2', 'Unknown']
MODALITY_WEIGHTS = [0.76, 0.24, 0.003]

# Lat distribution: ~88% Unknown, ~5.8% right, ~5.8% left, ~0.002% bilateral
LAT_CHOICES = ['Unknown', 'right', 'left', 'bilateral']
LAT_WEIGHTS = [0.88, 0.058, 0.058, 0.002]

# DWI b-values for non-unknown
DWI_BVALUES = [0, 50, 100, 500, 1000, 1500, 1800]


def calc_breast_size(num_slices, thickness):
    return f"{num_slices * thickness:.1f}"


def random_acq_time():
    hour = random.randint(6, 18)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{hour:02d}{minute:02d}{second:02d}"


def build_session(session_idx):
    id_base = f"SYNTH_{session_idx:02d}"
    accession = 900000 + session_idx
    name = f"TestPat_{session_idx:02d}_{random.randint(100000, 999999):06d}"
    id_full = f"RIA_{id_base}_{session_idx}_{random.randint(100000, 999999):06d}"
    date_str = f"{random.randint(2002, 2023):04d}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
    dob_str = f"{random.randint(1940, 1995):04d}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
    dir_path = f"/FL_system/data/raw/arc001/{accession}/SCANS/6/DICOM"
    img_dir = f"/FL_system/data/raw/{id_base}/arc001/{accession}/SCANS"

    rows = []
    file_idx = 1
    tri_times_post = sorted([random.randint(0, 100000) for _ in range(random.randint(6, 12))])
    num_post = len(tri_times_post)

    # 1. Localization/scout rows (TriTime='Unknown')
    localizer_descriptions = ['Loc', 'LOC', 'Localization']
    num_localizer = random.randint(1, 2)
    for _ in range(num_localizer):
        acq = random_acq_time()
        num_s = random.choice(NUM_SLICES_OPTIONS[:4])
        thick = random.choice(THICKNESS_OPTIONS[:2])
        rows.append({
            'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
            'Orientation': '0',
            'ID': id_full,
            'Accession': str(accession),
            'Name': name,
            'DATE': date_str,
            'DOB': dob_str,
            'Series_desc': random.choice(localizer_descriptions),
            'Modality': 'T1',
            'AcqTime': acq,
            'SrsTime': str(int(acq) - random.randint(0, 5)),
            'ConTime': float(acq),
            'StuTime': float(int(acq) - random.randint(500, 2000)),
            'TriTime': 'Unknown',
            'InjTime': 'Unknown',
            'ScanDur': f"{random.randint(10000000, 500000000):.1f}",
            'Lat': 'Unknown',
            'NumSlices': num_s,
            'Thickness': thick,
            'BreastSize': calc_breast_size(num_s, thick),
            'DWI': 'Unknown',
            'Type': random.choice(TYPE_VALUES[:4]),
            'Series': file_idx,
        })
        file_idx += 1

    # 2. Pre-contrast T1 sequence
    pre_acq = random_acq_time()
    pre_num_s = random.choice(NUM_SLICES_OPTIONS)
    pre_thick = random.choice(THICKNESS_OPTIONS)
    pre_type = random.choice(["['ORIGINAL', 'PRIMARY', 'OTHER']", "['ORIGINAL', 'PRIMARY', 'PRIMARY', 'NONE']"])
    pre_desc = random.choice(['T1 Sagittal pre', 'Axial T1 FS pre', 'Axial T1 pre', 'Axial T1', 'T1 pre'])
    pre_lat = random.choices(['Unknown', 'right', 'left', 'bilateral'], weights=LAT_WEIGHTS, k=1)[0]
    rows.append({
        'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
        'Orientation': random.choice(['0', '1', '2']),
        'ID': id_full,
        'Accession': str(accession),
        'Name': name,
        'DATE': date_str,
        'DOB': dob_str,
        'Series_desc': pre_desc,
        'Modality': 'T1',
        'AcqTime': pre_acq,
        'SrsTime': str(int(pre_acq)),
        'ConTime': float(pre_acq),
        'StuTime': float(int(pre_acq) - random.randint(800, 2500)),
        'TriTime': 'Unknown',
        'InjTime': 'Unknown',
        'ScanDur': f"{random.randint(50000000, 400000000):.1f}",
        'Lat': pre_lat,
        'NumSlices': pre_num_s,
        'Thickness': pre_thick,
        'BreastSize': calc_breast_size(pre_num_s, pre_thick),
        'DWI': 'Unknown',
        'Type': pre_type,
        'Series': file_idx,
    })
    file_idx += 1

    # 3. Optional non-fat-sat T1
    if random.random() < 0.4:
        acq = random_acq_time()
        num_s = random.choice(NUM_SLICES_OPTIONS)
        thick = random.choice(THICKNESS_OPTIONS)
        rows.append({
            'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
            'Orientation': random.choice(['0', '1', '2']),
            'ID': id_full,
            'Accession': str(accession),
            'Name': name,
            'DATE': date_str,
            'DOB': dob_str,
            'Series_desc': 'T1 non fat sat',
            'Modality': 'T1',
            'AcqTime': acq,
            'SrsTime': str(int(acq) - 1),
            'ConTime': float(acq),
            'StuTime': float(int(acq) - random.randint(800, 2500)),
            'TriTime': 'Unknown',
            'InjTime': 'Unknown',
            'ScanDur': f"{random.randint(50000000, 400000000):.1f}",
            'Lat': 'Unknown',
            'NumSlices': num_s,
            'Thickness': thick,
            'BreastSize': calc_breast_size(num_s, thick),
            'DWI': 'Unknown',
            'Type': pre_type,
            'Series': file_idx,
        })
        file_idx += 1

    # 4. Injection time row
    inj_acq = random_acq_time()
    rows.append({
        'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
        'Orientation': random.choice(['0', '1', '2']),
        'ID': id_full,
        'Accession': str(accession),
        'Name': name,
        'DATE': date_str,
        'DOB': dob_str,
        'Series_desc': 'PJN',
        'Modality': 'T1',
        'AcqTime': inj_acq,
        'SrsTime': str(int(inj_acq)),
        'ConTime': float(inj_acq),
        'StuTime': float(int(inj_acq) - random.randint(800, 2500)),
        'TriTime': 'Unknown',
        'InjTime': 'Unknown',
        'ScanDur': f"{random.randint(5000000, 30000000):.1f}",
        'Lat': 'Unknown',
        'NumSlices': random.choice([30, 40, 44]),
        'Thickness': random.choice(THICKNESS_OPTIONS),
        'BreastSize': '330.0',
        'DWI': 'Unknown',
        'Type': "['ORIGINAL', 'PRIMARY', 'PRIMARY', 'NONE']",
        'Series': file_idx,
    })
    file_idx += 1

    # 5. Post-contrast T1 sequences
    post_acq_base = str(int(pre_acq) + random.randint(600, 1200))
    for i, tri_ms in enumerate(tri_times_post):
        acq = str(int(post_acq_base) + i)
        num_s = random.choice(NUM_SLICES_OPTIONS)
        thick = random.choice(THICKNESS_OPTIONS)
        post_desc = random.choice(['T1 Sagittal post', 'Axial T1 FS post', 'Axial T1 post', 'T1 post', 'T1 Axial AP'])
        post_lat = random.choices(['Unknown', 'right', 'left', 'bilateral'], weights=LAT_WEIGHTS, k=1)[0]
        post_type = random.choice(["['ORIGINAL', 'PRIMARY', 'OTHER']", "['DERIVED', 'PRIMARY', 'OTHER', 'SUBTRACT']"])
        # Occasional Unknown modality (~0.3%)
        mod = 'Unknown' if random.random() < 0.003 else 'T1'
        rows.append({
            'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
            'Orientation': random.choice(['0', '1', '2']),
            'ID': id_full,
            'Accession': str(accession),
            'Name': name,
            'DATE': date_str,
            'DOB': dob_str,
            'Series_desc': post_desc,
            'Modality': mod,
            'AcqTime': str(acq),
            'SrsTime': str(acq),
            'ConTime': float(acq),
            'StuTime': float(int(acq) - random.randint(800, 2500)),
            'TriTime': str(tri_ms),
            'InjTime': 'Unknown',
            'ScanDur': f"{random.randint(50000000, 400000000):.1f}",
            'Lat': post_lat,
            'NumSlices': num_s,
            'Thickness': thick,
            'BreastSize': calc_breast_size(num_s, thick),
            'DWI': 'Unknown',
            'Type': post_type,
            'Series': file_idx,
        })
        file_idx += 1

    # 6. Optional MIP reconstruction
    if random.random() < 0.6:
        acq = random_acq_time()
        num_s = random.choice(NUM_SLICES_OPTIONS[:4])
        thick = random.choice(THICKNESS_OPTIONS[:2])
        # Use a post tri_times for MIP
        mip_tri = random.choice(tri_times_post)
        rows.append({
            'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
            'Orientation': '2',
            'ID': id_full,
            'Accession': str(accession),
            'Name': name,
            'DATE': date_str,
            'DOB': dob_str,
            'Series_desc': 'MIP T1',
            'Modality': 'T1',
            'AcqTime': acq,
            'SrsTime': str(int(acq)),
            'ConTime': float(acq),
            'StuTime': float(int(acq) - random.randint(800, 2500)),
            'TriTime': str(mip_tri),
            'InjTime': 'Unknown',
            'ScanDur': f"{random.randint(20000000, 100000000):.1f}",
            'Lat': 'Unknown',
            'NumSlices': num_s,
            'Thickness': thick,
            'BreastSize': calc_breast_size(num_s, thick),
            'DWI': 'Unknown',
            'Type': "['DERIVED', 'PRIMARY', 'PROJECTION IMAGE', 'IVI']",
            'Series': file_idx,
        })
        file_idx += 1

    # 7. Optional T2 sequence
    if random.random() < 0.5:
        is_bilateral = random.random() < 0.5
        t2_acq = random_acq_time()
        num_s = random.choice(NUM_SLICES_OPTIONS)
        thick = random.choice(THICKNESS_OPTIONS)
        t2_desc_base = 'T2 left breast' if not is_bilateral else 'T2 Axial FS'
        t2_type = "['ORIGINAL', 'PRIMARY', 'OTHER', 'NONE']"
        if not is_bilateral:
            side = random.choice(['left', 'right'])
            t2_lat = side
        else:
            t2_lat = 'bilateral'
            t2_desc_base = random.choice(['WATER: AX, T2 FS', 'Sagittal T2 FS', 'T2 FS AXIAL'])

        rows.append({
            'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
            'Orientation': random.choice(['1', '2']),
            'ID': id_full,
            'Accession': str(accession),
            'Name': name,
            'DATE': date_str,
            'DOB': dob_str,
            'Series_desc': t2_desc_base,
            'Modality': 'T2',
            'AcqTime': t2_acq,
            'SrsTime': str(int(t2_acq)),
            'ConTime': float(t2_acq),
            'StuTime': float(int(t2_acq) - random.randint(800, 2500)),
            'TriTime': 'Unknown',
            'InjTime': 'Unknown',
            'ScanDur': f"{random.randint(100000000, 400000000):.1f}",
            'Lat': t2_lat,
            'NumSlices': num_s,
            'Thickness': thick,
            'BreastSize': calc_breast_size(num_s, thick),
            'DWI': 'Unknown',
            'Type': t2_type,
            'Series': file_idx,
        })
        file_idx += 1

        # If unilateral, add the other side as well
        if not is_bilateral:
            other_side = 'right' if t2_lat == 'left' else 'left'
            t2_acq2 = str(int(t2_acq) + random.randint(500, 2000))
            t2_side_desc = f"T2 {other_side}"
            rows.append({
                'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
                'Orientation': random.choice(['1', '2']),
                'ID': id_full,
                'Accession': str(accession),
                'Name': name,
                'DATE': date_str,
                'DOB': dob_str,
                'Series_desc': t2_side_desc,
                'Modality': 'T2',
                'AcqTime': t2_acq2,
                'SrsTime': str(int(t2_acq2)),
                'ConTime': float(t2_acq2),
                'StuTime': float(int(t2_acq2) - random.randint(800, 2500)),
                'TriTime': 'Unknown',
                'InjTime': 'Unknown',
                'ScanDur': f"{random.randint(100000000, 400000000):.1f}",
                'Lat': other_side,
                'NumSlices': num_s,
                'Thickness': thick,
                'BreastSize': calc_breast_size(num_s, thick),
                'DWI': 'Unknown',
                'Type': t2_type,
                'Series': file_idx,
            })
            file_idx += 1

    # 8. Optional Dixon water image
    if random.random() < 0.3:
        acq = random_acq_time()
        num_s = random.choice(NUM_SLICES_OPTIONS[:4])
        thick = random.choice(THICKNESS_OPTIONS)
        dixon_type = "['DERIVED', 'PRIMARY', 'DIXON', 'WATER']"
        dixon_desc = random.choice(['WATER: AX, T2 FS', 'Axial T1 FS post'])
        if 'T1' in dixon_desc:
            d_lat = 'Unknown'
        else:
            d_lat = random.choices(['Unknown', 'bilateral'], weights=[0.9, 0.1], k=1)[0]
        rows.append({
            'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
            'Orientation': '2',
            'ID': id_full,
            'Accession': str(accession),
            'Name': name,
            'DATE': date_str,
            'DOB': dob_str,
            'Series_desc': dixon_desc,
            'Modality': 'T2',
            'AcqTime': acq,
            'SrsTime': str(int(acq)),
            'ConTime': float(acq),
            'StuTime': float(int(acq) - random.randint(800, 2500)),
            'TriTime': 'Unknown',
            'InjTime': 'Unknown',
            'ScanDur': f"{random.randint(20000000, 150000000):.1f}",
            'Lat': d_lat,
            'NumSlices': num_s,
            'Thickness': thick,
            'BreastSize': calc_breast_size(num_s, thick),
            'DWI': 'Unknown',
            'Type': dixon_type,
            'Series': file_idx,
        })
        file_idx += 1

    # 9. Optional DWI sequence
    if random.random() < 0.35:
        dwi_acq = random_acq_time()
        dwi_num_s = random.choice([30, 40, 44])
        dwi_thick = random.choice([3.0, 3.0, 3.0])
        dwi_desc = 'Axial DWI'
        dwi_lat = 'bilateral'
        bvalue = random.choice(DWI_BVALUES)
        dwi_type = "['DERIVED', 'PRIMARY', 'DIFFUSION', 'ADC']"
        rows.append({
            'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
            'Orientation': '2',
            'ID': id_full,
            'Accession': str(accession),
            'Name': name,
            'DATE': date_str,
            'DOB': dob_str,
            'Series_desc': dwi_desc,
            'Modality': 'T2',
            'AcqTime': dwi_acq,
            'SrsTime': str(int(dwi_acq)),
            'ConTime': float(dwi_acq),
            'StuTime': float(int(dwi_acq) - random.randint(800, 2500)),
            'TriTime': str(random.choice(tri_times_post)) if tri_times_post else 'Unknown',
            'InjTime': 'Unknown',
            'ScanDur': f"{random.randint(150000000, 400000000):.1f}",
            'Lat': dwi_lat,
            'NumSlices': dwi_num_s,
            'Thickness': dwi_thick,
            'BreastSize': calc_breast_size(dwi_num_s, dwi_thick),
            'DWI': str(bvalue),
            'Type': dwi_type,
            'Series': file_idx,
        })
        file_idx += 1

        # Optional ADC derivation row
        if random.random() < 0.6:
            adc_acq = str(int(dwi_acq) + random.randint(100, 500))
            rows.append({
                'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
                'Orientation': '2',
                'ID': id_full,
                'Accession': str(accession),
                'Name': name,
                'DATE': date_str,
                'DOB': dob_str,
                'Series_desc': f"ADC (10^-6 mm^2/s):Dec 01 2020 {adc_acq[:2]}-{adc_acq[2:4]}-{adc_acq[4:6]} EST",
                'Modality': 'T2',
                'AcqTime': adc_acq,
                'SrsTime': adc_acq,
                'ConTime': float(adc_acq),
                'StuTime': float(int(adc_acq) - random.randint(800, 2500)),
                'TriTime': str(random.choice(tri_times_post)) if tri_times_post else 'Unknown',
                'InjTime': 'Unknown',
                'ScanDur': f"{random.randint(20000000, 100000000):.1f}",
                'Lat': dwi_lat,
                'NumSlices': dwi_num_s,
                'Thickness': dwi_thick,
                'BreastSize': calc_breast_size(dwi_num_s, dwi_thick),
                'DWI': 'Unknown',
                'Type': "['DERIVED', 'PRIMARY', 'DIFFUSION', 'ADC']",
                'Series': file_idx,
            })
            file_idx += 1

    # 10. Optional STIR sequence
    if random.random() < 0.2:
        stir_acq = random_acq_time()
        num_s = random.choice(NUM_SLICES_OPTIONS)
        thick = random.choice(THICKNESS_OPTIONS)
        rows.append({
            'PATH': f"{dir_path}/{file_idx:04d}/img_{file_idx:04d}.dcm",
            'Orientation': '2',
            'ID': id_full,
            'Accession': str(accession),
            'Name': name,
            'DATE': date_str,
            'DOB': dob_str,
            'Series_desc': 'STIR',
            'Modality': 'T2',
            'AcqTime': stir_acq,
            'SrsTime': str(int(stir_acq)),
            'ConTime': float(stir_acq),
            'StuTime': float(int(stir_acq) - random.randint(800, 2500)),
            'TriTime': 'Unknown',
            'InjTime': 'Unknown',
            'ScanDur': f"{random.randint(100000000, 300000000):.1f}",
            'Lat': 'bilateral',
            'NumSlices': num_s,
            'Thickness': thick,
            'BreastSize': calc_breast_size(num_s, thick),
            'DWI': 'Unknown',
            'Type': "['ORIGINAL', 'PRIMARY', 'OTHER']",
            'Series': file_idx,
        })
        file_idx += 1

    return rows


all_rows = []
for i in range(NUM_SESSIONS):
    session_rows = build_session(i)
    all_rows.extend(session_rows)

df = pd.DataFrame(all_rows)

OUTPUT_PATH = '/mnt/projects/MRI_preprocessing/test/synthetic_Data_table.csv'
df.to_csv(OUTPUT_PATH, index=False)

print(f"Total rows: {len(df)}")
print(f"\nRows per session:")
session_ids = [f"{r['ID']}_{r['DATE']}" for r in all_rows]
unique_sessions = df['ID'].nunique()
print(f"  Unique sessions: {unique_sessions}")
print(f"  Avg rows/session: {len(df) / unique_sessions:.1f}")

print(f"\nModality distribution:")
print(df['Modality'].value_counts().to_string())

print(f"\nSeries_desc distribution:")
print(df['Series_desc'].value_counts().to_string())

print(f"\nTriTime distribution:")
tri_unknown = (df['TriTime'] == 'Unknown').sum()
tri_numeric = len(df) - tri_unknown
print(f"  Unknown: {tri_unknown}")
print(f"  Numeric: {tri_numeric}")

print(f"\nLaterality distribution:")
print(df['Lat'].value_counts().to_string())

print(f"\nFile written to: {OUTPUT_PATH}")
