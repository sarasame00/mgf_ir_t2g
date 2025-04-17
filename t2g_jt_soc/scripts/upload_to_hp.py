from huggingface_hub import upload_file, list_repo_files
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Load the .env file
load_dotenv()

# Read from the environment
HF_REPO_ID = "sarasame00/t2g_jt_soc_outputs"
HF_TOKEN = os.getenv("HF_TOKEN")  # Safely loaded!
HF_REPO_TYPE = "dataset"

LOCAL_FILE = ROOT_DIR / "single_site" / "ss_data" / "simulated_values_ss.csv"
FOLDER_PATH = ROOT_DIR / "single_site" / "ss_data" / "ss_results"


def upload_file_if_not_exists(file_path, existing_files):
    filename = os.path.basename(file_path)
    remote_path = f"single_site/{filename}"

    if remote_path in existing_files:
        print(f"⚠️ Skipping {filename} — already exists in repo.")
        return

    try:
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=remote_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
        )
        print(f"✅ Uploaded {filename}")
    except Exception as e:
        print(f"❌ Failed to upload {filename}: {e}")


def upload_folder_to_huggingface(folder_path):
    existing_files = list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN)
    print(f"📂 {len(existing_files)} files already in repo.")

    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, folder_path)
            remote_path = f"single_site/{rel_path.replace(os.sep, '/')}"

            if remote_path in existing_files:
                print(f"⚠️ Skipping {remote_path} — already exists.")
                continue

            try:
                upload_file(
                    path_or_fileobj=full_path,
                    path_in_repo=remote_path,
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                    token=HF_TOKEN,
                )
                print(f"✅ Uploaded {remote_path}")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print("🚫 Rate limit hit! Stopping uploads. Try again in an hour.")
                    return
                else:
                    print(f"❌ Failed to upload {remote_path}: {e}")
            except Exception as e:
                print(f"❌ Failed to upload {remote_path}: {e}")


# Run
if __name__ == "__main__":
    upload_folder_to_huggingface(FOLDER_PATH)
    # You can also add upload_file_if_not_exists(LOCAL_FILE, existing_files) if needed
