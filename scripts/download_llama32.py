"""Download Llama 3.2 3B Instruct (con fallback unsloth se gated)."""
import shutil, sys
from huggingface_hub import snapshot_download

def free_gb(): return shutil.disk_usage("/").free / 1e9

CANDIDATES = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
]
print(f"Free disk: {free_gb():.1f} GB")
for repo in CANDIDATES:
    print(f"\nTrying {repo}...")
    try:
        local = snapshot_download(repo_id=repo)
        print(f"  OK cached at {local}")
        print(f"  Free disk after: {free_gb():.1f} GB")
        sys.exit(0)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")

print("ALL FAILED")
sys.exit(1)
