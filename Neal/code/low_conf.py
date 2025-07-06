import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2
from PIL import Image
from rich.console import Console
import tensorflow as tf

console = Console()

# ===== Path Setting =====
model_path  = r"C:\E521C\Model\model_smoothing\best_val_acc_phase2_run1.h5"
test_dir    = r"C:\E521C\Dataset\trash_dataset_v2_new_split\test"
output_root = r"C:\E521C\Result\low_conf"
image_size = (192, 192)

# ===== Threshold Setting =====
confidence_threshold = 0.65
gap_threshold = 0.2

# ===== Load Model & Category =====
model = load_model(
    model_path,
    custom_objects={'custom_loss': lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing = 0.01)}
)

class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
print(f"\nCategory: {class_names}\n")


# ===== Clear Old Output Data =====
if os.path.exists(output_root):
    for folder in ["low_conf_correct", "low_conf_wrong", "ambiguous", "high_conf_wrong"]:
        dir_path = os.path.join(output_root, folder)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  

# ===== Create Folders =====
output_dirs = {
    "low_conf_correct": os.path.join(output_root, "low_conf_correct"),
    "low_conf_wrong": os.path.join(output_root, "low_conf_wrong"),
    "ambiguous": os.path.join(output_root, "ambiguous"),
    "high_conf_wrong": os.path.join(output_root, "high_conf_wrong")
}

for d in output_dirs.values():
    os.makedirs(d, exist_ok = True)

# ===== Helper Functions =====
def load_image(path):
    img = Image.open(path).convert('RGB').resize(image_size)
    arr = np.array(img)
    arr = mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis = 0)

# ===== Inference and Classification Logic =====
total = 0
counts = {k: 0 for k in output_dirs}

for true_class in class_names:
    class_folder = os.path.join(test_dir, true_class)
    for fname in os.listdir(class_folder):
        if not fname.lower().endswith(('jpg', 'jpeg', 'png')):
            continue

        img_path = os.path.join(class_folder, fname)
        img_tensor = load_image(img_path)

        preds = model.predict(img_tensor, verbose = 0)[0]
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]
        confidence = preds[pred_idx]
        sorted_preds = np.sort(preds)[::-1]
        gap = sorted_preds[0] - sorted_preds[1]

        total += 1

        # ===== Classification Conditions =====
        if confidence < confidence_threshold and pred_class == true_class:
            dest = output_dirs["low_conf_correct"]
            counts["low_conf_correct"] += 1
        elif confidence < confidence_threshold and pred_class != true_class:
            dest = output_dirs["low_conf_wrong"]
            counts["low_conf_wrong"] += 1
        elif confidence >= 0.9 and pred_class != true_class:  
            dest = output_dirs["high_conf_wrong"]
            counts["high_conf_wrong"] += 1
        elif gap < gap_threshold:
            dest = output_dirs["ambiguous"]
            counts["ambiguous"] += 1

        else:
            continue  # normal data

        # ===== Copy File =====
        dst_path = os.path.join(dest, f"{true_class}__{pred_class}__{confidence:.2f}__{fname}") # file name
        shutil.copy(img_path, dst_path)

# ===== Print Result =====
print(f"\n🔎 Analysis, {total} images were processed")
for k, v in counts.items():
    print(f"📁 {k}: {v}")
print(f"\nSaved result to: {output_root}")

# ===== Chart =====
labels = list(counts.keys())
values = list(counts.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color = ['orange', 'crimson', 'skyblue'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha = 'center', va = 'bottom', fontsize = 12)

plt.title("Manual Review Category Counts", fontsize = 14)
plt.ylabel("Number of Images")
plt.tight_layout()
plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)
plt.show()
plt.close()