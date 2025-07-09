# scripts/test_icloud_access.py

import os
from pathlib import Path

ICLOUD_FOLDER = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess")
TEST_FILE = Path(ICLOUD_FOLDER) / "test_access.txt"

def test_icloud():
    try:
        # Write test
        with open(TEST_FILE, "w") as f:
            f.write("iCloud access test successful.")

        # Read test
        with open(TEST_FILE, "r") as f:
            content = f.read()
        print("Read from iCloud:", content)

        # Cleanup
        os.remove(TEST_FILE)
        print("Cleanup complete. iCloud access verified.")
    except Exception as e:
        print("iCloud access failed:", e)

if __name__ == "__main__":
    test_icloud()
