from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import numpy as np
import requests
import os
import pandas as pd
from pathlib import Path
import csv


# === CONFIG ===
SERVICE_ACCOUNT_FILE = Path(__file__).parent / "drive_service_account.json"
FOLDER_ID = '1qroO12tPkKu6c3w5Xy-2Reys5XFcbX5L'
SCOPES = ['https://www.googleapis.com/auth/drive']


# Make creds globally accessible
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# Also make drive_service accessible
drive_service = build('drive', 'v3', credentials=creds)

def get_file_id_by_name(filename, parent_id):
    """Query Google Drive folder for a file ID by name."""
    query = f"name = '{filename}' and '{parent_id}' in parents and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    return items[0]['id'] if items else None

def download_txt_file(file_id):
    """Download text file content (CSV or TXT) by file ID."""
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {creds.token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return np.loadtxt(io.StringIO(response.text))

def upload_file_to_drive(filepath, filename=None, parent_id=None, overwrite=True):
    from googleapiclient.http import MediaFileUpload

    filename = filename or os.path.basename(filepath)

    file_metadata = {'name': filename}
    if parent_id:
        file_metadata['parents'] = [parent_id]

    media = MediaFileUpload(filepath, resumable=True)

    # Try to find existing file first
    existing_id = None
    if overwrite and parent_id:
        query = f"name = '{filename}' and '{parent_id}' in parents and trashed = false"
        results = drive_service.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        if files:
            existing_id = files[0]["id"]

    if existing_id:
        # Update file content 
        updated = drive_service.files().update(
            fileId=existing_id,
            media_body=media
        ).execute()
        print(f"🔁 Updated '{filename}' on Drive (ID: {updated['id']})")
        return updated['id']
    else:
        # Upload as new file
        uploaded = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"✅ Uploaded new '{filename}' to Drive (ID: {uploaded['id']})")
        return uploaded['id']

def get_completed_params_from_drive(csv_filename, parent_folder_id):
    """
    Downloads the CSV from Drive and returns a set of completed param tuples.
    """
    file_id = get_file_id_by_name(csv_filename, parent_folder_id)
    if not file_id:
        return set()

    # Download text from Drive (as string)
    response = requests.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
        headers={"Authorization": f"Bearer {creds.token}"}
    )
    response.raise_for_status()

    content = response.text
    reader = csv.reader(io.StringIO(content))
    header = next(reader, None)  # skip header

    completed = set()

    for row in reader:
        if len(row) < 8:
            continue  # skip incomplete rows
        try:
            parsed = tuple(round(float(s), 3) for s in row[:-1])
            completed.add(parsed)
        except ValueError:
            print(f"⚠️ Skipping row with invalid data: {row}")
            continue  # skip bad rows


    return completed


def download_csv_from_drive(local_path, parent_id, filename):
    """Download a CSV file from Drive if exists."""
    file_id = get_file_id_by_name(filename, parent_id)
    if not file_id:
        print(f"⚠️ No existing {filename} found in Drive. Starting fresh.")
        return None
    
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    print(f"    ↓ Downloaded {filename} from Drive.")
    return local_path

def upload_csv_to_drive(local_path, parent_id, filename):
    """Upload (overwrite) CSV to Drive."""
    file_id = get_file_id_by_name(filename, parent_id)
    file_metadata = {'name': filename, 'parents': [parent_id]}
    media = MediaFileUpload(local_path, resumable=True)

    if file_id:
        updated = drive_service.files().update(fileId=file_id, media_body=media).execute()
        print(f"    ↻ Updated '{filename}' in Drive.")
    else:
        uploaded = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"    ✓ Uploaded new '{filename}' to Drive.")


def update_and_upload_csv(new_row: list, local_csv_path, gd_id_dir, csv_name, header):
    """Safely update the model-specific CSV with new rows and upload."""

    # Download existing CSV if it exists
    download_csv_from_drive(local_csv_path, gd_id_dir, csv_name)

    if local_csv_path.exists():
        df = pd.read_csv(local_csv_path)
    else:
        df = pd.DataFrame(columns=header)

    if not new_row:
        print("⚠️ No new rows to add, skipping upload.")
        return

    # 🧠 Detect if it's a single row (list of values) or list of rows
    if isinstance(new_row[0], (int, float, str)):
        # It's a single row → wrap it
        new_row = [new_row]

    # Create DataFrame
    new_df = pd.DataFrame(new_row, columns=header)

    # Concatenate safely
    combined_df = pd.concat([df, new_df], ignore_index=True).drop_duplicates()

    # Save locally
    combined_df.to_csv(local_csv_path, index=False)

    # Upload to Drive
    upload_csv_to_drive(local_csv_path, gd_id_dir, csv_name)




__all__ = [
    "get_file_id_by_name",
    "download_txt_file",
    "upload_file_to_drive",
    "creds",
    "drive_service",
    "get_completed_params_from_drive"
]
