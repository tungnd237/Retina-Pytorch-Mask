import csv
import os

# Define the directories where your images are stored
masked_dir = 'Dataset_image/flat_masked_dataset'
unmasked_dir = 'Dataset_image/flat_unmasked_dataset'

# Define the CSV file where you want to save the labels
csv_file = 'dataset_labels.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['image', 'label'])

    # Write the rows for masked images
    for image in os.listdir(masked_dir):
        writer.writerow([os.path.join(masked_dir, image), 1])

    # Write the rows for unmasked images
    for image in os.listdir(unmasked_dir):
        writer.writerow([os.path.join(unmasked_dir, image), 0])
