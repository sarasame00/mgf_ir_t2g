import pandas as pd
import io
import requests
import argparse
from pathlib import Path
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# === Setup ===
import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from drive_utils import drive_service, creds, get_file_id_by_name, upload_csv_to_drive

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Compare and clean Drive files vs. CSV entries.")
parser.add_argument('--model', choices=['ss', 'lat'], required=True, help="Choose model: 'ss' or 'lat'")
args = parser.parse_args()

# === Config per Model ===
if args.model == "lat":
    from config.lat_settings import LAT_GD_ID_DIR as GD_ID_DIR, LAT_CSV_NAME as CSV_NAME
    EXTENSION = ".hdf5"
elif args.model == "ss":
    from config.ss_settings import SS_GD_ID_DIR as GD_ID_DIR, SS_CSV_NAME as CSV_NAME
    EXTENSION = ""

# === Helpers ===
def list_drive_files(folder_id, extension=""):
    if extension:
        query = f"'{folder_id}' in parents and trashed = false and name contains '{extension}'"
    else:
        query = f"'{folder_id}' in parents and trashed = false and not name contains '.'"
    results = drive_service.files().list(q=query, fields="files(id, name, size)").execute()
    return results.get('files', [])

def get_csv_dataframe(folder_id, filename):
    file_id = get_file_id_by_name(filename, folder_id)
    if not file_id:
        raise FileNotFoundError(f"CSV '{filename}' not found in Drive folder '{folder_id}'")
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {creds.token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

def delete_drive_file_by_name(folder_id, filename):
    file_id = get_file_id_by_name(filename, folder_id)
    if file_id:
        drive_service.files().delete(fileId=file_id).execute()
        print(f"🗑️ Deleted '{filename}' from Drive.")

# === Main Logic ===
if __name__ == "__main__":
    print("🔍 Scanning Drive and CSV for mismatches and empties...")

    files = list_drive_files(GD_ID_DIR, EXTENSION)
    df = get_csv_dataframe(GD_ID_DIR, CSV_NAME)

    if 'timestamp' not in df.columns:
        raise ValueError("CSV is missing 'timestamp' column.")

    valid_files = set()
    empty_files = []
    for f in files:
        name = f['name']
        if int(f.get('size', 0)) == 0:
            empty_files.append(name)
        if EXTENSION:
            if name.endswith(EXTENSION):
                valid_files.add(name.replace(EXTENSION, ''))
        else:
            valid_files.add(name)

    timestamps_csv = set(df['timestamp'].astype(str))

    missing_in_drive = timestamps_csv - valid_files
    missing_in_csv = valid_files - timestamps_csv

    print("\n=== Comparison Summary ===")
    if missing_in_drive:
        print(f"⚠️ Missing in Drive: {missing_in_drive}")
    if missing_in_csv:
        print(f"⚠️ Missing in CSV: {missing_in_csv}")
    if empty_files:
        print(f"⚠️ Empty files: {empty_files}")
    if not (missing_in_drive or missing_in_csv or empty_files):
        print("✅ All good! CSV and Drive are consistent.")
        sys.exit(0)

    confirm = input("\nDo you want to perform cleanup? This will DELETE files and modify the CSV. (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cleanup aborted by user.")
        exit()

    # === Cleanup ===
    print("\n🧹 Starting cleanup...")

    # 1. Delete missing/empty files from Drive
    to_delete = missing_in_csv.union(empty_files)
    for filename in to_delete:
        if EXTENSION and not filename.endswith(EXTENSION):
            filename += EXTENSION
        delete_drive_file_by_name(GD_ID_DIR, filename)

    # 2. Clean CSV rows
    df_clean = df[df['timestamp'].astype(str).isin(valid_files)].copy()
    df_clean.drop_duplicates(inplace=True)

    # 3. Upload cleaned CSV to Drive
    tmp_path = Path("_temp_cleaned.csv")
    df_clean.to_csv(tmp_path, index=False)
    upload_csv_to_drive(tmp_path, GD_ID_DIR, CSV_NAME)
    tmp_path.unlink()

    print("\n✅ Cleanup completed successfully.")
