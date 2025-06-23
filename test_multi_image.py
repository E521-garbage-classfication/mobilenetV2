import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# 模型路徑
model_path = r'C:\E521C\saved_models_v2_new\trash_model_advanced.h5'

# 測試資料夾路徑（跟訓練時同結構）
test_dir = r'C:\Users\codek\Downloads\test123'

# 參數設定
image_size = (192, 192)
batch_size = 32

# 載入模型
model = tf.keras.models.load_model(model_path)
print("Model loaded.")

# 讀取測試資料集
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int',
    shuffle=False  # 測試集不打亂，方便對應真實標籤
)

class_names = test_ds.class_names
print(f"Classes: {class_names}")

# 預處理函數（與訓練相同）
def preprocess(x, y):
    return mobilenet_v2.preprocess_input(x), y

test_ds = test_ds.map(preprocess)

# 預測所有測試資料
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 計算準確率
accuracy = np.mean(y_true == y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 列印分類報告
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(8, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.grid(False)
plt.show()
