import subprocess
import shutil
import os

# Find all 9x9 SGF files
result = subprocess.run(
    ["grep", "-rl", "SZ\[9\]", ".", "--include=*.sgf"],
    capture_output=True,
    text=True,
)

files = result.stdout.strip().split("\n")
print(f"Found {len(files)} 9x9 games")

# Copy to a new directory
os.makedirs("data/9x9", exist_ok=True)
for i, f in enumerate(files):
    shutil.copy(f, "data/9x9/")
    if i % 10000 == 0:
        print(f"Copied {i}/{len(files)} files")

print("Done")
