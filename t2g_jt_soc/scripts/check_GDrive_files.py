from drive_utils import drive_service, get_file_id_by_name, creds
import pandas as pd
import io
from googleapiclient.http import MediaIoBaseDownload

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config.lat_settings import LAT_GD_ID_DIR, LAT_CSV_NAME, LAT_CSV_HEADER

# List all HDF5 files in the folder
def list_hdf5_files(folder_id):
    query = f"'{folder_id}' in parents and trashed = false and name contains '.hdf5'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    return [item['name'] for item in items]

# Read CSV directly from Drive
def get_timestamps_from_drive_csv(folder_id, csv_filename):
    file_id = get_file_id_by_name(csv_filename, folder_id)
    if not file_id:
        raise FileNotFoundError(f"CSV '{csv_filename}' not found in Drive folder '{folder_id}'")

    # Use requests to download as string
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {creds.token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Parse CSV content directly
    csv_content = io.StringIO(response.text)
    df = pd.read_csv(csv_content)

    if 'timestamp' not in df.columns:
        raise ValueError("The CSV does not contain a 'timestamp' column.")
    
    return set(df['timestamp'].astype(str))

# Compare files
def compare_files_and_csv(folder_id, csv_filename):
    hdf5_files = list_hdf5_files(folder_id)
    timestamps_in_csv = get_timestamps_from_drive_csv(folder_id, csv_filename)

    timestamps_in_drive = set(fname.replace('.hdf5', '') for fname in hdf5_files)

    missing_in_drive = timestamps_in_csv - timestamps_in_drive
    missing_in_csv = timestamps_in_drive - timestamps_in_csv

    print("=== Comparison Result ===")
    if missing_in_drive:
        print(f"⚠️ Timestamps in CSV but missing HDF5 files in Drive: {missing_in_drive}")
    else:
        print("✅ All CSV timestamps have corresponding HDF5 files.")
    
    if missing_in_csv:
        print(f"⚠️ HDF5 files in Drive without entries in CSV: {missing_in_csv}")
    else:
        print("✅ All HDF5 files are registered in the CSV.")

def list_empty_hdf5_files(folder_id):
    query = f"'{folder_id}' in parents and trashed = false and name contains '.hdf5'"
    results = drive_service.files().list(
        q=query,
        fields="files(name, size)",
    ).execute()
    files = results.get('files', [])
    empty_files = [f['name'] for f in files if int(f.get('size', 0)) == 0]
    return empty_files

if __name__ == "__main__":
    compare_files_and_csv(LAT_GD_ID_DIR, LAT_CSV_NAME)
    empty_files = list_empty_hdf5_files(LAT_GD_ID_DIR)
    if empty_files:
        print(f"⚠️ Found {len(empty_files)} empty HDF5 files:")
        print(empty_files)
    else:
        print("✅ No empty HDF5 files found.")
