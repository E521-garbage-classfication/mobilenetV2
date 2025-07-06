import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import mobilenet_v2
from typing import Tuple, List
from rich.console import Console
from glob import glob

console = Console()

# ===== Setting =====
model_path = r"C:\E521C\Model\model_smoothing\best_val_acc_phase2_run1.h5"
test_dir = r"C:\E521C\Dataset\trash_dataset_v2_new_split\test"
image_size = (192, 192)
batch_size = 32

# ===== Data Loader =====
def load_test_dataset(path: str, image_size: Tuple[int, int], batch_size: int):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size = image_size,
        batch_size = batch_size,
        label_mode = 'categorical',
        shuffle = False
    )
    class_names = ds.class_names
    ds = ds.map(lambda x, y: (mobilenet_v2.preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)
    return ds, class_names

# ===== Evaluation =====
def evaluate_model(model, dataset) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose = 0)
        y_true.extend(np.argmax(labels.numpy(), axis = 1))
        y_pred.extend(np.argmax(preds, axis = 1))
    return np.array(y_true), np.array(y_pred)

def print_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    cm = confusion_matrix(y_true, y_pred)
    accs = cm.diagonal() / cm.sum(axis = 1)
    print()
    for i, acc in enumerate(accs):
        print(f"{class_names[i]} Accuracy: {acc:.2%}")

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize = (8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)
    disp.plot(cmap = 'Blues', values_format = 'd', ax = ax)
    plt.title('Confusion Matrix')
    plt.grid(False)
    if save_path:
        plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()
    plt.close()

# ===== Main =====
def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing = 0.01)

model = tf.keras.models.load_model(model_path, custom_objects={'custom_loss': custom_loss})
print("Model loaded.")

test_ds, class_names = load_test_dataset(test_dir, image_size, batch_size)
print(f"Detected Classes: {class_names}")

y_true, y_pred = evaluate_model(model, test_ds)
accuracy = np.mean(y_true == y_pred)
console.print(f"[cyan]\nOverall Test Accuracy: {accuracy:.2%}[/]")

console.print("\n[magenta]Classification Report:[/]\n")
print(classification_report(y_true, y_pred, target_names = class_names, zero_division = 0))

console.print("[magenta]Per-Class Accuracy:[/]")
print_per_class_accuracy(y_true, y_pred, class_names)
print(" ")

plot_confusion_matrix(y_true, y_pred, class_names)

# ===== Optional: Save low-confidence samples =====
manual_review_dir = r"C:\E521C\Result\traincheck"
os.makedirs(manual_review_dir, exist_ok = True)
confidence_threshold = 0.65         # threshold

if os.path.exists(manual_review_dir):
    for file in os.listdir(manual_review_dir):
        file_path = os.path.join(manual_review_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

file_paths = sorted(glob(os.path.join(test_dir, '*', '*')))
test_ds_with_paths = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = image_size,
    batch_size = 1,
    label_mode = 'categorical',
    shuffle = False
)
test_ds_with_paths = test_ds_with_paths.map(lambda x, y: (mobilenet_v2.preprocess_input(x), y))

for i, (images, labels) in enumerate(test_ds_with_paths):
    image = images[0]
    label = labels[0]
    file_path = file_paths[i]

    pred = model.predict(tf.expand_dims(image, axis = 0), verbose = 0)[0]
    pred_idx = np.argmax(pred)
    true_idx = np.argmax(label.numpy())

    confidence = pred[pred_idx]
    if pred_idx != true_idx and confidence < confidence_threshold:
        filename = os.path.basename(file_path)
        dst_path = os.path.join(manual_review_dir, f"{class_names[true_idx]}__{class_names[pred_idx]}__{confidence:.2f}__{filename}")
        shutil.copy(file_path, dst_path)

print(f"Low-confidence images saved to: {manual_review_dir}")
