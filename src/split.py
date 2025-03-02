from logging import root
import os
import shutil
import argparse
from get_data import get_data

def train_and_test(config_file):
    config = get_data(config_file)
    root_dir = config['raw_data']['data_src']
    dest = config['load_data']['preprocessed_data']

    # Dictionary mapping class indexes to names
    classes = {0: 'no_tumor', 1: 'pituitary_tumor', 2: 'meningioma_tumor', 3: 'glioma_tumor'}

    # Create directories for each class index inside 'train' and 'test'
    os.makedirs(os.path.join(dest, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'test'), exist_ok=True)

    for class_id in classes.keys():  # Using indices as folder names
        os.makedirs(os.path.join(dest, 'train', f'class_{class_id}'), exist_ok=True)
        os.makedirs(os.path.join(dest, 'test', f'class_{class_id}'), exist_ok=True)

    # Copy files from Training directory
    training_dir = os.path.join(root_dir, 'Training')
    for class_id, class_name in classes.items():
        src_dir = os.path.join(training_dir, class_name)  # Use class name to fetch files
        if not os.path.exists(src_dir):
            print(f"Warning: Directory {src_dir} does not exist. Skipping...")
            continue

        files = os.listdir(src_dir)
        print(f"{class_name} (Training) -> {len(files)} images")

        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dest, 'train', f'class_{class_id}', f)  # Saving with "class_" prefix
            shutil.copy(src_path, dst_path)

        print(f"Done copying training data for {class_name} (saved in class_{class_id})")

    # Copy files from Testing directory
    testing_dir = os.path.join(root_dir, 'Testing')
    for class_id, class_name in classes.items():
        src_dir = os.path.join(testing_dir, class_name)  # Use class name to fetch files
        if not os.path.exists(src_dir):
            print(f"Warning: Directory {src_dir} does not exist. Skipping...")
            continue

        files = os.listdir(src_dir)
        print(f"{class_name} (Testing) -> {len(files)} images")

        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dest, 'test', f'class_{class_id}', f)  # Saving with "class_" prefix
            shutil.copy(src_path, dst_path)

        print(f"Done copying testing data for {class_name} (saved in class_{class_id})")


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    train_and_test(config_file=passed_args.config)