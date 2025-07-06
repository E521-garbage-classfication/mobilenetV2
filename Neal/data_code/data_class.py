import os
import random
import shutil

original_data_dir = r"C:\E521C\Dataset\trash_dataset"
output_dir = r"C:\E521C\Dataset\trash_dataset_v2_new_split"

# ===== Clear Folder =====
def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

clear_directory(output_dir)

# ===== Ratio Setting =====
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ===== Create Folders =====
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(train_dir)
os.makedirs(val_dir)
os.makedirs(test_dir)

# ===== Split per class =====
class_names = [folder for folder in os.listdir(original_data_dir) if os.path.isdir(os.path.join(original_data_dir, folder))]
for class_name in class_names:
    class_dir = os.path.join(original_data_dir, class_name)
    images = [img for img in os.listdir(class_dir) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    random.shuffle(images)
    total_images = len(images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size
    
    # Create empty class subfolders
    os.makedirs(os.path.join(train_dir, class_name))
    os.makedirs(os.path.join(val_dir, class_name))
    os.makedirs(os.path.join(test_dir, class_name))

    # Copy images
    for img in images[:train_size]:
        shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))
    for img in images[train_size:train_size + val_size]:
        shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, class_name, img))
    for img in images[train_size + val_size:]:
        shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, class_name, img))

print(f"Dataset split done. Saved at: {output_dir}")
