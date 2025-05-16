import os
from rembg import remove
from PIL import Image

# Paths to the dataset
input_dir = '../combined_dataset'  # The folder containing train, val, and test
output_dir = '../processed_dataset_no_bg'  # Output folder for images with backgrounds removed

# Create output directories
for split in ['train', 'valid', 'test']:
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Process each split
for split in ['train', 'valid', 'test']:
    split_input_dir = os.path.join(input_dir, split)
    split_output_dir = os.path.join(output_dir, split)

    # Process each class folder
    for class_name in os.listdir(split_input_dir):
        class_input_dir = os.path.join(split_input_dir, class_name)
        class_output_dir = os.path.join(split_output_dir, class_name)

        # Create output class directory
        os.makedirs(class_output_dir, exist_ok=True)

        # Process each image in the class folder
        for filename in os.listdir(class_input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(class_input_dir, filename)
                output_path = os.path.join(class_output_dir, filename)

                # Remove background
                with open(input_path, "rb") as input_file:
                    input_data = input_file.read()

                output_data = remove(input_data)

                # Save the output image
                with open(output_path, "wb") as output_file:
                    output_file.write(output_data)

print("Background removal completed for all images.")