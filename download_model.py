import os
import sys
from huggingface_hub import snapshot_download

def download_qwen_model():
    print("===========================================")
    print("[Optimus Model Loader] HuggingFace Weight Downloader")
    print("===========================================")
    
    repo_id = "cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4"
    local_dir = "./qwen_3_6_27b_dummy"
    
    print(f"[1] Target Repository: {repo_id}")
    print(f"[2] Local Destination: {local_dir}")
    print("[3] Filtering for safetensors to leverage our Bare-Metal Zero-Copy Loader...")
    print("[Download in progress... This may take ~40 minutes depending on your 300mbps connection.]")
    
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
            ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5"],
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n[SUCCESS] Model weights downloaded successfully to: {path}")
        print("You can now execute `qwen_optimus_runner.py` to ignite the engine!")
        
    except Exception as e:
        print(f"\n[Fatal] Failed to download model from HuggingFace: {e}")
        print("Hint: Ensure you have `huggingface_hub` installed in your environment.")
        sys.exit(1)

if __name__ == "__main__":
    download_qwen_model()
