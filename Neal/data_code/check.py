import os
import tensorflow as tf
from rich.console import Console

console = Console()

# ===== Path Setting =====
data_dir = r"C:\E521C\Dataset\trash_dataset"  

# ===== Categories =====
print("Num of Categories: ", len(os.listdir(data_dir)))
print("Category Name: ", os.listdir(data_dir), "\n")    

# ===== Number of Data =====
sum = 0
for folder in sorted(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, folder)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f'{folder}: {count} ')
        sum += count
print("total:", sum, "\n")

# ===== Check Image Format =====
valid_exts = ['.jpg', '.jpeg', '.png']
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[1].lower()
        if ext not in valid_exts:
            print(f"Invaild Image Format: {file}（Category: {folder}）")
print("No invaild image\n")

bad_files = []
for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if not os.path.isdir(class_path):
        continue
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        try:
            img = tf.io.read_file(file_path)
            img = tf.image.decode_image(img)
        except Exception as e:
            print(f"Unable to Decode: {file_path}")
            bad_files.append(file_path)

print(f"\n There are {len(bad_files)} image can't be decoded")