import os
import re
import shutil

WORKSPACE = "workspace"
UTILS_DIR = os.path.join(WORKSPACE, "utils")

# Files to move/flatten from workspace/utils/ to workspace/
REQUIRED_FILES = [
    "hd_helpers.py",
    "helpers.py",
    "extract_helpers.py",
]

# Regex rewrites to apply inside files under workspace/
# e.g., "from utils.helpers import X" -> "from .helpers import X"
IMPORT_REWRITES = [
    # Match lines like: from utils.module import ...
    (re.compile(r'(?m)^\s*from\s+utils\.([A-Za-z0-9_]+)\s+import\b'), r'from .\1 import'),
]

def rewrite_imports(code: str) -> str:
    for pattern, replacement in IMPORT_REWRITES:
        code = pattern.sub(replacement, code)
    return code

def ensure_workspace_package():
    os.makedirs(WORKSPACE, exist_ok=True)
    init_path = os.path.join(WORKSPACE, "__init__.py")
    if not os.path.exists(init_path):
        # Make workspace a package so "from .helpers" resolves at runtime
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("")  # empty pkg marker

def move_required_utils_files():
    if not os.path.isdir(UTILS_DIR):
        print(f"{UTILS_DIR} does not exist. Nothing to move.")
        return

    for fname in REQUIRED_FILES:
        src = os.path.join(UTILS_DIR, fname)
        dest = os.path.join(WORKSPACE, fname)

        if not os.path.exists(src):
            print(f"Missing: {src} (skipping)")
            continue

        with open(src, "r", encoding="utf-8") as f:
            code = f.read()

        # Apply rewrites while moving
        new_code = rewrite_imports(code)

        # Write to destination (overwrite if exists)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(new_code)

        # Remove the original file so it's truly flattened
        os.remove(src)
        print(f"Moved and rewrote: {src} -> {dest}")

    # Optionally remove utils dir if it becomes empty
    try:
        if os.path.isdir(UTILS_DIR) and not os.listdir(UTILS_DIR):
            os.rmdir(UTILS_DIR)
            print(f"🧹 Removed empty directory: {UTILS_DIR}")
    except OSError:
        # Directory not empty or cannot remove — ignore
        pass

def rewrite_all_workspace_files():
    if not os.path.isdir(WORKSPACE):
        print(f"{WORKSPACE} does not exist; nothing to rewrite.")
        return

    ws_abs = os.path.abspath(WORKSPACE)
    utils_abs = os.path.abspath(UTILS_DIR)

    for root, dirs, files in os.walk(WORKSPACE):
        # Skip any .git directories just in case
        dirs[:] = [d for d in dirs if d != ".git"]

        for file in files:
            if not file.endswith(".py"):
                continue

            full_path = os.path.join(root, file)

            # Skip files still inside utils/ (they will be moved or intentionally left)
            if os.path.abspath(full_path).startswith(utils_abs + os.sep):
                continue

            with open(full_path, "r", encoding="utf-8") as f:
                original = f.read()

            rewritten = rewrite_imports(original)

            if rewritten != original:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(rewritten)
                rel = os.path.relpath(full_path, WORKSPACE)
                print(f"Rewrote imports in: {rel}")

def main():
    ensure_workspace_package()
    # 1) Move selected files from workspace/utils -> workspace (with rewrites)
    move_required_utils_files()
    # 2) Rewrite imports across all files under workspace (in-place)
    rewrite_all_workspace_files()
    print("Done: utils files moved and imports rewritten under ./workspace/")

if __name__ == "__main__":
    main()
