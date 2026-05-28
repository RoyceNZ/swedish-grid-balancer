import os
from pathlib import Path

# Files/folders to explicitly ignore to save context window space
IGNORE_LIST = {
    "venv", ".git", "__pycache__", ".pytest_cache", 
    "outputs", "data", ".env", ".gitignore", "pack_repo.py"
}
IGNORE_EXT = {".png", ".jpg", ".joblib", ".parquet", ".csv", ".gz"}

def pack_repository(output_file="codebase_snapshot.txt"):
    root_dir = Path(".")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== REPOSITORY SNAPSHOT ===\n\n")
        
        # 1. Write out the directory structure tree
        f.write("--- DIRECTORY TREE ---\n")
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_LIST]
            level = len(Path(root).relative_to(root_dir).parts)
            indent = " " * 4 * level
            f.write(f"{indent}📁 {os.path.basename(root)}/\n")
            sub_indent = " " * 4 * (level + 1)
            for file in files:
                if not any(file.endswith(ext) for ext in IGNORE_EXT):
                    f.write(f"{sub_indent}📄 {file}\n")
        
        f.write("\n" + "="*50 + "\n\n")
        
        # 2. Append the actual code content of every valid file
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_LIST]
            for file in files:
                if any(file.endswith(ext) for ext in IGNORE_EXT):
                    continue
                file_path = Path(root) / file
                f.write(f"--- FILE: {file_path.as_posix()} ---\n")
                try:
                    with open(file_path, "r", encoding="utf-8") as code_f:
                        f.write(code_f.read())
                except Exception as e:
                    f.write(f"[Error reading file: {e}]")
                f.write("\n\n" + "-"*30 + "\n\n")

    print(f"🎉 Codebase packaged successfully into: {output_file}")

if __name__ == "__main__":
    pack_repository()