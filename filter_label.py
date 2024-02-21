# Load the list of occluded images
with open('occluded_images.txt', 'r') as file:
    occluded_images = {line.strip() for line in file}

# Path to your original label file
label_file_path = '/Users/macbook/Documents/VinAI/Pytorch_Retinaface/data/widerface/train/label.txt'

# Path to the new label file that will only include occluded faces
filtered_label_file_path = '/Users/macbook/Documents/VinAI/widerface/filtered_label.txt'

with open(label_file_path, 'r') as label_file, open(filtered_label_file_path, 'w') as filtered_label_file:
    write_line = False
    for line in label_file:
        if line.startswith('#'):  # Image file indicator
            # Extract the image filename from the line
            image_filename = line.strip().split(' ')[1]
            # Determine if this image should be included
            write_line = image_filename in occluded_images
            if write_line:
                # Write the image filename line
                filtered_label_file.write(line)
        elif write_line:
            # Write the bounding box line
            filtered_label_file.write(line)

print(f"Filtered labels have been saved to {filtered_label_file_path}")
