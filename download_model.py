#!/usr/bin/env python3
"""
Script to download the pre-trained model weights for card rectification.
Run this script to download CRDN1000.pkl from the original repository.
"""

import os
import sys
import urllib.request
from urllib.error import URLError, HTTPError

MODEL_URL = "https://github.com/shakex/card-rectification/raw/master/CRDN1000.pkl"
MODEL_FILENAME = "CRDN1000.pkl"

def download_model():
    """Download the model weights file if it doesn't exist."""
    
    if os.path.exists(MODEL_FILENAME):
        print(f"✓ {MODEL_FILENAME} already exists.")
        return True
    
    print(f"Downloading {MODEL_FILENAME} from {MODEL_URL}...")
    
    try:
        # Download with progress indication
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME, progress_hook)
        print(f"\n✓ Successfully downloaded {MODEL_FILENAME}")
        
        # Check file size
        file_size = os.path.getsize(MODEL_FILENAME)
        print(f"File size: {file_size / (1024*1024):.1f} MB")
        
        return True
        
    except HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: {e.reason}")
        return False
    except URLError as e:
        print(f"\n✗ URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        return False

def main():
    """Main function to download model weights."""
    print("Card Rectification Model Downloader")
    print("=" * 40)
    
    success = download_model()
    
    if success:
        print("\n✓ Model download completed successfully!")
        print("You can now run the Flask application with: python app.py")
    else:
        print("\n✗ Model download failed!")
        print("Please download CRDN1000.pkl manually from:")
        print(MODEL_URL)
        sys.exit(1)

if __name__ == "__main__":
    main()
