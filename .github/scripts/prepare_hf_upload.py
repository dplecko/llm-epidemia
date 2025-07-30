import os
import re

# These are the original import paths you want to rewrite
IMPORT_REWRITES = {
    r"from utils\.([a-zA-Z0-9_]+) import": r"from .\1 import"
}

FOLDERS_TO_FLATTEN = ["utils"]

for folder in FOLDERS_TO_FLATTEN:
    if not os.path.exists(folder):
        continue

    for root, _, files in os.walk(folder):
        for file in files:
            if not file.endswith(".py"):
                continue

            full_path = os.path.join(root, file)

            # Read original file
            with open(full_path, "r") as f:
                code = f.read()

            # Rewrite imports
            for pattern, replacement in IMPORT_REWRITES.items():
                code = re.sub(pattern, replacement, code)

            # Compute destination file path at top-level
            relative_path = os.path.relpath(full_path, folder)
            new_filename = os.path.basename(relative_path)
            dest_path = os.path.join("workspace", new_filename)

            # Write modified file to ./workspace/
            with open(dest_path, "w") as f:
                f.write(code)

print("Code flattened and imports rewritten in ./workspace/")
