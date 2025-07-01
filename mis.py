import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2
import matplotlib.pyplot as plt
from PIL import Image

# ----- 基本設定 -----
model_path = r'C:\E521C\saved_models_v2_new\trash_model_advanced.h5'
test_dir = r'C:\E521C\DataImage\trash_dataset_v2_new_split\test'
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
img_size = (192, 192)

# ----- 載入模型 -----
model = load_model(model_path, custom_objects={'custom_loss': lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.02)})

# ----- 輔助函式 -----
def load_image(path):
    img = Image.open(path).convert('RGB').resize(img_size)
    arr = np.array(img)
    arr = mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0), img

# ----- 掃描錯誤分類 -----
wrong_preds = []

for class_name in os.listdir(test_dir):
    class_folder = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_folder):
        continue

    for filename in os.listdir(class_folder):
        if not filename.lower().endswith(('jpg', 'png', 'jpeg')):
            continue

        img_path = os.path.join(class_folder, filename)
        img_tensor, img_pil = load_image(img_path)
        preds = model.predict(img_tensor, verbose=0)
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]
        true_class = class_name

        if pred_class != true_class:    # 檢查全部分類
        # if true_class == 'plastic' and pred_class != 'plastic': # ' ' 單獨檢查被分錯種類
            wrong_preds.append((img_pil, img_path, true_class, pred_class, preds[0][pred_idx]))

# ----- 顯示錯分圖片 -----
print(f"Total misclassified: {len(wrong_preds)}")
print("Showing top 20 misclassified samples:")

for i, (img, path, true_cls, pred_cls, prob) in enumerate(wrong_preds[:20]):
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'True: {true_cls} | Pred: {pred_cls} ({prob*100:.1f}%)\n{os.path.basename(path)}')
    plt.show()
