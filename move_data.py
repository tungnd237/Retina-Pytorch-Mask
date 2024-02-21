import os
import shutil

# Paths to your datasets
root_masked_dir = '/Users/macbook/Downloads/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'
root_unmasked_dir = '/Users/macbook/Downloads/self-built-masked-face-recognition-dataset/AFDB_face_dataset'

# New directories where you want to save the flattened dataset
flat_masked_dir = '/Users/macbook/Documents/VinAI/Data/flat_masked_dataset'
flat_unmasked_dir = '/Users/macbook/Documents/VinAI/Data/flat_unmasked_dataset'

if not os.path.exists(flat_masked_dir):
    os.makedirs(flat_masked_dir)

if not os.path.exists(flat_unmasked_dir):
    os.makedirs(flat_unmasked_dir)

def flatten_directory(root_dir, flat_dir):
    for root, dirs, files in os.walk(root_dir):
        print(f"Checking directory: {root}")  # Add a print statement to show current directory being checked
        for file in files:
            if file.lower().endswith('.jpg'):  # Adjust this if your images have a different extension
                original_file_path = os.path.join(root, file)
                new_file_path = os.path.join(flat_dir, file)
                
                if os.path.exists(new_file_path):
                    base, extension = os.path.splitext(file)
                    i = 1
                    new_file_name = f"{base}_{i}{extension}"
                    new_file_path = os.path.join(flat_dir, new_file_name)
                    while os.path.exists(new_file_path):
                        i += 1
                        new_file_name = f"{base}_{i}{extension}"
                        new_file_path = os.path.join(flat_dir, new_file_name)
                
                print(f"Copying file {original_file_path} to {new_file_path}")  # Print statement to show file copy operation
                shutil.copy(original_file_path, new_file_path)
            else:
                print(f"Skipping file {file} as it does not have a .jpg extension")

flatten_directory(root_masked_dir, flat_masked_dir)
flatten_directory(root_unmasked_dir, flat_unmasked_dir)

print("Flattening complete.")

