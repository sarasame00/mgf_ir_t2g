from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
import io
import numpy as np
import requests
import os
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

    completed = set(
        tuple(round(float(s), 3) for s in row[:-1])
        for row in reader
        if len(row) >= 8  # make sure row is complete
    )

    return completed



__all__ = [
    "get_file_id_by_name",
    "download_txt_file",
    "upload_txt_file",
    "creds",
    "drive_service",
    "get_completed_params_from_drive"
]
