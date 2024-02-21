def parse_label_file(label_file_path, output_file_path):
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

    occluded_images = []
    current_file = ""
    for line in lines:
        if '.jpg' in line:
            if current_file and any_face_occluded:  # If the previous file had any occluded face, save it
                occluded_images.append(current_file)
            current_file = line.strip()  # Update the current file
            any_face_occluded = False  # Reset the occlusion flag for the new file
        else:
            parts = line.split()
            if len(parts) >= 10:  # Checking for at least 10 parts to ensure a full line of data
                occlusion = int(parts[8])
                if occlusion == 1 or occlusion == 2:
                    any_face_occluded = True  # Set the flag if any face is occluded

    if current_file and any_face_occluded:  # Check the last file in the list
        occluded_images.append(current_file)

    # Write the results to the output file, ensuring consistent order
    with open(output_file_path, 'w') as out_file:
        for img in sorted(occluded_images):  # Sort to maintain consistent order
            out_file.write(img + '\n')

# Paths
label_file_path = '/Users/macbook/Documents/VinAI/widerface/wider_face_split/wider_face_train_bbx_gt.txt'  # Replace with your actual path
output_file_path = '/Users/macbook/Documents/VinAI/widerface/wider_face_split/occluded_images.txt'  # Replace with your desired output file path

# Process the label file and save the output
parse_label_file(label_file_path, output_file_path)
print(f"Occluded images have been saved to {output_file_path}")

