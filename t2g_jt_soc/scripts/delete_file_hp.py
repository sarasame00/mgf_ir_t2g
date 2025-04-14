from huggingface_hub import delete_file
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

delete_file(
    path_in_repo="single_sitesimulated_values_ss.csv",  
    repo_id="sarasame00/t2g_jt_soc_outputs",
    repo_type="dataset",
    token=HF_TOKEN
)

print("✅ File deleted!")