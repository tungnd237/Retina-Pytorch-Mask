import os
import argparse
from pathlib import Path

def parse_occlusion_file(occlusion_label_path):
    occlusion_info = {}
    with open(occlusion_label_path, 'r') as file:
        lines = file.readlines()
        current_filename = None
        for line in lines:
            line = line.strip()
            if line.endswith('.jpg'):
                current_filename = line
                occlusion_info[current_filename] = []
            else:
                parts = line.split()
                if len(parts) == 10:
                    _, _, _, _, _, _, _, _, occlusion, _ = map(int, parts)
                    mask_flag = 1 if occlusion in [1, 2] else 0
                    occlusion_info[current_filename].append(mask_flag)
                else:
                    print(f"Skipping line due to unexpected format: {line}")
    return occlusion_info

def add_mask_flag_to_original_labels(original_label_path, occlusion_info, output_path):
    with open(original_label_path, 'r') as original_file, open(output_path, 'w') as output_file:
        lines = original_file.readlines()
        current_filename = None
        current_masks = []
        for line in lines:
            if line.startswith('#'):
                output_file.write(line)
                current_filename = line[2:].strip()
                if current_filename in occlusion_info:
                    current_masks = occlusion_info[current_filename]
            else:
                mask_flag = current_masks.pop(0) if current_filename in occlusion_info and current_masks else 0
                output_file.write(f"{line.strip()} {mask_flag}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Combine original labels with occlusion information.")
    parser.add_argument("-i", "--original_label_path", type=Path, required=True, help="Path to the original label file.")
    parser.add_argument("-o", "--occlusion_label_path", type=Path, required=True, help="Path to occlusion label file.")
    parser.add_argument("-out", "--output_path", type=Path, required=True, help="Path for the output file.")
    return parser.parse_args()

def main():
    args = parse_args()
    # Parse occlusion information
    occlusion_info = parse_occlusion_file(args.occlusion_label_path)
    # Combine original labels with occlusion information
    add_mask_flag_to_original_labels(args.original_label_path, occlusion_info, args.output_path)
    print("Completed combining label files.")

if __name__ == "__main__":
    main()
