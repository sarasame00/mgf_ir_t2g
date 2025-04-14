from googleapiclient.discovery import build
from google.oauth2 import service_account
import io
import numpy as np
import requests
import os
from pathlib import Path

# === CONFIG ===
SERVICE_ACCOUNT_FILE = Path(__file__).parent.parent / "drive_service_account.json"
FOLDER_ID = '1qroO12tPkKu6c3w5Xy-2Reys5XFcbX5L'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# ✅ Make creds globally accessible
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# ✅ Also make drive_service accessible
drive_service = build('drive', 'v3', credentials=creds)

# Setup Drive API client
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

def get_file_id_by_name(filename):
    """Query Google Drive folder for a file ID by name."""
    query = f"name = '{filename}' and '{FOLDER_ID}' in parents and trashed = false"
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

__all__ = [
    "get_file_id_by_name",
    "download_txt_file",
    "upload_txt_file",
    "creds",
    "drive_service"
]
