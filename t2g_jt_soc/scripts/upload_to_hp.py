from huggingface_hub import upload_file
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Load the .env file
load_dotenv()

# Read from the environment
HF_REPO_ID = "sarasame00/t2g_jt_soc_outputs"
HF_TOKEN = os.getenv("HF_TOKEN")  # Safely loaded!

LOCAL_FILE = ROOT_DIR / "single_site" / "ss_data" / "simulated_values_ss.csv"
FOLDER_PATH = ROOT_DIR / "single_site" / "ss_data" / "ss_results"

def upload_h5_to_huggingface(file_path):
    filename = os.path.basename(file_path)

    try:
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo="single_site/" +filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print(f"✅ Uploaded {filename} to {HF_REPO_ID}")
    except Exception as e:
        print(f"❌ Failed to upload: {e}")


def upload_folder_to_huggingface(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo="single_site/" + file,  
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            print(f"✅ Uploaded {file}")
        except Exception as e:
            print(f"❌ Failed to upload {file}: {e}")



def upload_if_not_exists(local_file_path, remote_path=None):
    if remote_path is None:
        remote_path = os.path.basename(local_file_path)

    # List existing files in the dataset
    existing_files = list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN)

    # Check if the file already exists
    if remote_path in existing_files:
        print(f"⚠️ File already exists on HF: {remote_path} — skipping upload.")
        return

    # Upload if not found
    try:
        upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=remote_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
        )
        print(f"✅ Uploaded: {remote_path}")
    except Exception as e:
        print(f"❌ Failed to upload {remote_path}: {e}")

        
# Run
if __name__ == "__main__":
    upload_folder_to_huggingface(FOLDER_PATH)
    # upload_h5_to_huggingface(LOCAL_FILE)