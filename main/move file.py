import os
import shutil
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path(r"C:\Users\DUY\Downloads\part1\part1")
TARGET_DIR = Path(r"C:\Users\DUY\Desktop\code\AI gender detection\train\female")

REQUIRED_AGE = 18
REQUIRED_GENDER = 1 # 0 for male

# --- Setup ---

# 1. Create the target directory if it doesn't exist
try:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory '{TARGET_DIR}' ensured.")
except Exception as e:
    print(f"Error creating target directory: {e}")
    exit()

# --- Processing Loop ---

print(f"Starting to process files in '{SOURCE_DIR}'...")
moved_count = 0

# Use .glob('*') to find all files in the source directory
for file_path in SOURCE_DIR.glob('*'):
    # Check if the path points to a file and not a directory
    if file_path.is_file():
        file_name = file_path.name
        
        # UTKFace file names are formatted as: [age]_[gender]_[race]_[date&time].jpg
        try:
            # 1. Strip the file extension (e.g., '.jpg') to get the base name
            base_name = file_path.stem
            
            # 2. Split the base name by the underscore '_'
            parts = base_name.split('_')
            
            # The UTKFace format has at least 4 parts, we only need the first two
            if len(parts) >= 3:
                age = int(parts[0])
                gender = int(parts[1])
                
                # 3. Apply the filtering criteria
                if age >= REQUIRED_AGE and gender == REQUIRED_GENDER:
                    
                    # 4. Construct the destination path
                    destination_path = TARGET_DIR / file_name
                    
                    # 5. Move the file
                    shutil.move(file_path, destination_path)
                    moved_count += 1
                    # print(f"Moved: {file_name} (Age: {age})") # Uncomment for detailed output
            else:
                # Handle files that do not match the expected naming convention
                print(f"Skipping file with unexpected name format: {file_name}")

        except ValueError:
            # Handles cases where age/gender is not a valid integer
            print(f"Skipping file due to non-integer age/gender: {file_name}")
        except Exception as e:
            # Catch other potential errors (e.g., file permission issues)
            print(f"An error occurred while processing {file_name}: {e}")

# --- Summary ---
print("\n--- Summary ---")
print(f"Successfully **moved {moved_count}** male faces (Age >= 18) to: {TARGET_DIR}")