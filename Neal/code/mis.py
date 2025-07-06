import os
import numpy as np
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2
from PIL import Image
from collections import Counter
from rich.console import Console

console = Console()
# ===== Path Setting =====
model_path = r"C:\E521C\Model\model_smoothing\best_val_acc_phase2_run1.h5"
test_dir = r"C:\E521C\Dataset\trash_dataset_v2_new_split\test"
img_size = (192, 192)

# ===== Category Name =====
class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
print(f"Detected classes: {class_names}\n")

# ===== Load Model =====
model = load_model(
    model_path,
    custom_objects={'custom_loss': lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing = 0.01)}
)   # label smoothing

# ===== Helper Functions =====
def load_image(path):
    img = Image.open(path).convert('RGB').resize(img_size)
    arr = np.array(img)
    arr = mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis = 0), img

# ===== Scan Error Classification =====
wrong_preds = []
for true_class in class_names:
    class_folder = os.path.join(test_dir, true_class)
    for filename in os.listdir(class_folder):
        if not filename.lower().endswith(('jpg', 'jpeg', 'png')):
            continue

        img_path = os.path.join(class_folder, filename)
        img_tensor, img_pil = load_image(img_path)
        preds = model.predict(img_tensor, verbose=0)
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]

        if pred_class != true_class:
            prob = preds[0][pred_idx]
            wrong_preds.append((img_pil, img_path, true_class, pred_class, prob))

wrong_preds = sorted(wrong_preds, key=lambda x: x[4])

# ===== Classification Error Statistics =====
err_stat = Counter([true_cls for _, _, true_cls, pred_cls, _ in wrong_preds])
print("\nTop error counts per class:")
for cls, cnt in err_stat.most_common():
    print(f"{cls}: {cnt}")
    
labels, values = zip(*err_stat.most_common())
plt.figure(figsize = (10, 6))
plt.barh(labels, values, color='tomato')
plt.gca().invert_yaxis()
plt.xlabel('Error Count')
plt.title('Top Error Counts Per Class')
plt.grid(True, axis = 'x')
plt.tight_layout()
plt.show()
plt.close()

# ===== Save Misclassified Images =====
save_dir = r"C:\E521C\Result\misclassified"
os.makedirs(save_dir, exist_ok = True)

# ===== Clear Old Files =====
for f in os.listdir(save_dir):
    f_path = os.path.join(save_dir, f)
    if os.path.isfile(f_path):
        os.remove(f_path)

for i, (_, path, true_cls, pred_cls, prob) in enumerate(wrong_preds):
    filename = os.path.basename(path)
    dst_filename = f"{true_cls}__{pred_cls}__{prob:.2f}__{filename}"
    dst_path = os.path.join(save_dir, dst_filename)
    try:
        shutil.copy(path, dst_path)
    except Exception as e:
        print(f"Failed to copy {path}: {e}")

print(f"\nMisclassified images saved to: {save_dir}")

# ===== Show Wrong Images =====
print(f"\nTotal misclassified: {len(wrong_preds)}")
