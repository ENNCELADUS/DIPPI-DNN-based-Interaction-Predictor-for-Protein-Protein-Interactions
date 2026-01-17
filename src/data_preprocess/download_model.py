#!/usr/bin/env python3
"""
ESM-3 Model Downloader and Exporter.

This script:
1. Redirects HF cache to a local directory (models/esm3).
2. Authenticates with Hugging Face Hub (if needed).
3. Downloads the ESM-3 open model via the official SDK.
4. Exports the model weights to a standalone .pth file.
"""

import os
import argparse
import torch
from pathlib import Path

try:
    from huggingface_hub import login
    from esm.models.esm3 import ESM3
except ImportError:
    print("Error: Required packages 'huggingface_hub' or 'esm' not installed.")
    print("Please run: conda activate esm")
    exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download and export ESM-3 model weights."
    )
    parser.add_argument(
        "--model", default="esm3_sm_open_v1", help="ESM3 model name to download"
    )
    parser.add_argument(
        "--no-login", action="store_true", help="Skip Hugging Face login"
    )
    parser.add_argument(
        "--no-export", action="store_true", help="Skip exporting to standalone .pth"
    )
    args = parser.parse_args()

    # 1. Setup local cache path
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    local_cache = workspace_root / "models" / "esm3"

    # Ensure directory exists
    local_cache.mkdir(parents=True, exist_ok=True)

    # Set HF_HOME environment variable to redirect SDK cache to our local folder
    os.environ["HF_HOME"] = str(local_cache)

    # 2. Authentication
    if not args.no_login:
        print("Logging in to Hugging Face Hub...")
        login()

    # 3. Download / Load Model
    print(f"Loading ESM-3 model '{args.model}' via SDK...")
    # This will download to models/esm3 or use existing cache
    model = ESM3.from_pretrained(args.model, device=torch.device("cpu"))
    print("Model loaded successfully.")

    # 4. Export to standalone .pth
    if not args.no_export:
        output_path = local_cache / f"{args.model}_full.pth"
        print(f"Exporting state_dict to: {output_path}")

        # Save only the state_dict for maximum portability
        torch.save(model.state_dict(), output_path)

        print("Export complete!")
        print(f"Standalone file size: {output_path.stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
